import argparse
import os

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from src import config
from src.utils import save_image, split_extension


parser = argparse.ArgumentParser(description="Apply transforms to images for correct FID measuring.")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str, default="sat2map",
                    help="Config filename (default: %(default)s).")
parser.add_argument("-d", "--data_dir", type=str, default="auto",
                    help="Path to directory with input images (default: %(default)s).")
parser.add_argument("-s", "--save_dir", type=str, default="auto",
                    help="Where to save images (default: %(default)s).")
parser.add_argument("--unnorm", type=str, choices=["yes", "no"], default="yes",
                    help="Unnormalize images [-1; 1] -> [0, 1] when saving (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)

data_config = config_.DataConfig()
model_config = config_.ModelConfig()
train_config = config_.TrainConfig()


if args.data_dir != "auto":
    data_config.valid_images_dir = args.data_dir

dataset = data_config.valid_dataset
data_loader = DataLoader(
    dataset,
    batch_size=train_config.valid_batch,
    shuffle=False
)


save_dir = args.save_dir
if save_dir == "auto":
    save_dir = f"resources/{args.config}_fid"
os.makedirs(save_dir, exist_ok=True)
for name, (input, target) in tqdm(zip(dataset.ids, iter(dataset)), desc="Processing data"):
    path = f"{save_dir}/{split_extension(os.path.basename(name))[0]}.png"
    save_image(path, target, args.unnorm == "yes")
