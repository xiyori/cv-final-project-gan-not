import numpy as np
from PIL import Image


class TopCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img):
        if isinstance(img, Image.Image):
            x = (img.size[0] - self.output_size[1]) // 2
            img = img.crop((x, 0, x + self.output_size[1], self.output_size[0]))
        elif isinstance(img, np.ndarray):
            x = (img.shape[1] - self.output_size[1]) // 2
            img = img[:self.output_size[0], x:x + self.output_size[1]]
        else:
            raise NotImplementedError(f"{self.__class__.__name__} not implemented for images of type {type(img)}")
        return img
