import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # Class Filter Fix (Target ID -> 255 for CVAT compatibility)
        mask_np = np.array(mask.convert('L'))
        binary_mask = (mask_np == 255).astype(np.float32)
        mask = Image.fromarray((binary_mask * 255).astype(np.uint8))

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # 1. Class Filtering (Must be performed before augmentation)
        mask_np = np.array(mask.convert('L'))
        binary_mask = (mask_np == 255).astype(np.float32)
        mask = Image.fromarray((binary_mask * 255).astype(np.uint8))

        # 2. Data Augmentation Pipeline
        if self.augment:
            # Horizontal Flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Rotation
            angle = random.uniform(-10, 10)
            img = transforms.functional.rotate(img, angle)
            mask = transforms.functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            # Random Perspective
            if random.random() < 0.3:
                width, height = img.size
                startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, 0.3)
                img = transforms.functional.perspective(img, startpoints, endpoints)
                mask = transforms.functional.perspective(mask, startpoints, endpoints, interpolation=InterpolationMode.NEAREST)

        # 3. Return Tensors
        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }