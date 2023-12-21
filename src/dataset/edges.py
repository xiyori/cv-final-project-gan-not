import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from .transforms import default_input_post_transform, default_target_post_transform


class EdgesDataset(Dataset):
    """Edges dataset. Read images and apply transforms.

    Args:
        images_dir (str): path to images folder
        pre_transform (torchvision.transforms.transform):
            image transform before edge detection
        input_post_transform (torchvision.transforms.transform):
            input image transform after edge detection
        target_post_transform (torchvision.transforms.transform):
            target image transform after edge detection
        thresholds (tuple): Canny edge detection params

    """

    def __init__(
            self,
            images_dir,
            pre_transform = None,
            input_post_transform = None,
            target_post_transform = None,
            thresholds = (100, 200)
    ):
        self.images_dir = images_dir[:-1] if images_dir[-1] == "/" else images_dir
        self.pre_transform = transforms.Lambda(lambda x: x) if pre_transform is None else pre_transform
        self.input_post_transform = default_input_post_transform() if input_post_transform is None \
                                                                   else input_post_transform
        self.target_post_transform = default_target_post_transform() if target_post_transform is None \
                                                                     else target_post_transform
        self.thresholds = thresholds

        self.ids = [name for name in os.listdir(self.images_dir) if
                    name.lower().endswith('.png') or
                    name.lower().endswith('.jpg') or
                    name.lower().endswith('.jpeg') or
                    # name.lower().endswith('.gif') or
                    name.lower().endswith('.bmp')]
        self.ids.sort()

        self.ids = [self.images_dir + "/" + name for name in self.ids]

    def __getitem__(self, i):
        # Load image
        target = cv2.imread(self.ids[i])
        target = target[..., ::-1]
        # pair = cv2.cvtColor(pair, cv2.COLOR_BGR2GRAY)

        # Apply pre-transforms
        target = Image.fromarray(target, mode="RGB")
        target = self.pre_transform(target)

        # Extract edges
        input = cv2.Canny(np.asarray(target),
                          threshold1=self.thresholds[0],
                          threshold2=self.thresholds[1])
        input = 255 - input

        # Apply post-transforms
        input = self.input_post_transform(input)
        target = self.target_post_transform(target)

        return input, target

    def __len__(self):
        return len(self.ids)
