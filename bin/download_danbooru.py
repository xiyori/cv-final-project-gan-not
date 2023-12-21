import argparse
import os
import requests
import json

from requests.exceptions import ConnectionError
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Dict


def main():
    args = parse_args()

    params = dict()
    if args.proxy != "none":
        params["proxies"] = {"http": args.proxy, "https": args.proxy}

    config = load_json(args.config)

    try:
        posts_meta = load_json(config["path"] + "/meta.json")
        print("Saved posts metadata found, skipping downloading")
    except FileNotFoundError:
        posts_meta = download_posts_meta(config, params, args.n_retries)

    n_jobs = min(args.n_jobs, len(posts_meta))
    Parallel(n_jobs=n_jobs)(delayed(download_images)(config, posts_meta[i::n_jobs], params, args.n_retries)
                            for i in range(n_jobs))


def download_posts_meta(config: Dict, params: Dict, n_retries: int = 10):
    path = config["path"]
    tags_and = config["tags_and"]
    tags_or = config["tags_or"]
    ratings = config["ratings"]

    api_address = "https://danbooru.donmai.us/posts.json?"

    posts_meta = []
    tqdm_bar = tqdm(desc=f"Downloading posts metadata")
    for i in range(max(len(tags_or), 1)):
        current_id = None
        while True:
            api_request = f"{api_address}&limit=200&" \
                          f"tags={'+'.join(tags_and + tags_or[i:i + 1])}+rating:{','.join(ratings)}"
            if current_id is not None:
                api_request += f"&page=b{current_id}"
            response = request_with_retries(api_request, params, n_retries)
            next_batch_meta = response.json()
            if len(next_batch_meta) == 0:
                break
            merge(posts_meta, next_batch_meta)
            current_id = next_batch_meta[-1]["id"]
            tqdm_bar.update(1)
    tqdm_bar.close()

    print("Saving posts metadata")
    save_json(path + "/meta.json", posts_meta)

    return posts_meta


def download_images(config: Dict, metadata: List[Dict], params: Dict, n_retries: int = 10):
    path = config["path"]
    file_url_priority = config["file_url_priority"]

    for post in tqdm(metadata, desc="Downloading images"):
        for image_url_key in file_url_priority:
            request: str = post.get(image_url_key, None)
            if request is not None:
                break
        else:
            continue

        extension = request.split('.')[-1]
        image_path = f"{path}/{post['id']}.{extension}"
        if os.path.exists(image_path):
            continue

        response = request_with_retries(request, params, n_retries)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        del response


def request_with_retries(request: str, params: Dict, n_retries: int = 10):
    for i in range(n_retries):
        try:
            response = requests.get(request, **params)
            if response.status_code != 200:
                raise ConnectionError(f"Http request failed, status code {response.status_code}")
            return response
        except ConnectionError as e:
            if i == n_retries - 1:
                raise e


def merge(metadata: List[Dict], next_meta: List[Dict]):
    ids = [post["id"] for post in metadata]
    for post in next_meta:
        if post["id"] not in ids:
            metadata += [post]


def save_json(path: str, metadata: List[Dict]):
    pardir = os.path.dirname(path)
    os.makedirs(pardir, exist_ok=True)
    json_object = json.dumps(metadata, indent=4)
    with open(path, "w") as f:
        f.write(json_object)


def load_json(path: str):
    with open(path, "r") as f:
        json_object = json.load(f)
    return json_object


def parse_args():
    parser = argparse.ArgumentParser(description="Create images dataset from Danbooru tags.")
    parser.add_argument("-c", "--config", metavar="CONFIG", type=str, required=True,
                        help="Config filename (default: %(default)s).")
    parser.add_argument("-p", "--proxy", metavar="PROXY", type=str, default="none",
                        help="Proxy address:port (default: %(default)s).")
    parser.add_argument("--n_jobs", metavar="INT", type=int, default=4,
                        help="Number of parallel jobs for downloading (default: %(default)s).")
    parser.add_argument("--n_retries", metavar="INT", type=int, default=10,
                        help="Number of times to repeat http request in case of failure (default: %(default)s).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
