import argparse
import os

import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from src import config
from src.utils import image_grid, save_image, split_extension
from bin.download_danbooru import load_json


parser = argparse.ArgumentParser(description="Create grid from output images.")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str, default="sat2map",
                    help="Config filename (default: %(default)s).")
parser.add_argument("-i", "--ids", type=str, default="config/grid_ids.json",
                    help="Path to image id json (default: %(default)s).")
parser.add_argument("-d", "--data_dir", type=str, default="auto",
                    help="Path to directory with input images (default: %(default)s).")
parser.add_argument("-o", "--out_dir", type=str, default="auto",
                    help="Path to directory with output images (default: %(default)s).")
parser.add_argument("-s", "--save_dir", type=str, default="resources/grid",
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


out_dir = args.out_dir
if out_dir == "auto":
    out_dir = f"resources/{args.config}"

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

ids = load_json(args.ids)

transform = transforms.ToTensor()

inputs = []
outputs = []
targets = []
for i, name in enumerate(dataset.ids):
    base = split_extension(os.path.basename(name))[0]
    if base in ids:
        input, target = dataset[i]
        if args.unnorm == "yes":
            input = input / 2 + 0.5
            target = target / 2 + 0.5
        output = cv2.imread(f"{out_dir}/{base}.png")[..., ::-1].copy()
        output = transform(output)
        inputs += [torch.transpose(input, -2, -1)]
        outputs += [torch.transpose(output, -2, -1)]
        targets += [torch.transpose(target, -2, -1)]

inputs = torch.stack(inputs, dim=0)
outputs = torch.stack(outputs, dim=0)
targets = torch.stack(targets, dim=0)

grid = image_grid(inputs, outputs, targets, num_images=len(ids), clip=(0, 1))
grid = torch.transpose(grid, -2, -1)
save_image(f"{save_dir}/{os.path.basename(out_dir)}.png", grid, unnorm=False)
