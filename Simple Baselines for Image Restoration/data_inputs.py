import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SIDD_Dataset(Dataset):

    def __init__(self, img_dir, transform=None):

        self.transform = transform
        self.img_dir = os.path.normpath(img_dir)

    def __len__(self):
        '''To get the dataset size'''
        return len([name for name in os.listdir(self.img_dir)])

    def __getitem__(self, idx):

        assert idx <= len(
            self), "Index should be smaller than the dataset size"

        for name in os.listdir(self.img_dir):

            if(int(name[:4])-1 == idx):
                noisy_img = None
                good_img = None

                imgs_path = os.path.join(self.img_dir, name)
                for f in os.listdir(imgs_path):
                    if(f[0] == "G"):
                        good_img = io.imread(os.path.join(imgs_path, f))
                if(f[0] == "N"):
                    noisy_img = io.imread(os.path.join(imgs_path, f))

                output = {'good_img': good_img, "noisy_img": noisy_img}

                if self.transform:
                    output = self.transform(output)

                return output


class Rescale(object):
    """Rescale the images in a sample to a given size.

    Args:
        output_size (tuple ): Desired output size (H*W). Output is
            matched to output_size.
    """

    def __init__(self, output_size=(1000, 1800)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        good_img, noisy_img = sample['good_img'], sample['noisy_img']
        new_h, new_w = self.output_size

        good_img = transform.resize(good_img, (new_h, new_w))
        noisy_img = transform.resize(noisy_img, (new_h, new_w))

        return {'good_img': good_img, "noisy_img": noisy_img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        good_img, noisy_img = sample['good_img'], sample['noisy_img']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        good_img = good_img.transpose((2, 0, 1))
        noisy_img = noisy_img.transpose((2, 0, 1))
        return {'good_img': torch.from_numpy(good_img),
                'noisy_img': torch.from_numpy(noisy_img)}
