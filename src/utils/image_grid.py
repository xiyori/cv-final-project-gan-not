import torch
from torch import Tensor


def image_grid(*images, num_images = 1, clip=(-1, 1)) -> Tensor:
    images = [image.repeat(1, 3 // image.shape[1], 1, 1) for image in images]
    horizontal_grid = torch.cat(images, dim=-1)
    images = torch.unbind(horizontal_grid[:num_images], dim=0)
    vertical_grid = torch.cat(images, dim=-2)
    out = torch.clip(vertical_grid, min=clip[0], max=clip[1])
    return out
