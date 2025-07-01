import torch
import random

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
class GroupRandomCrop:
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, int) else size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in img_group]


class GroupCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        return [transforms.CenterCrop(self.size)(img) for img in img_group]


class GroupRandomHorizontalFlip:
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        if random.random() < 0.5:
            flipped = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(flipped), 2):
                    flipped[i] = ImageOps.invert(flipped[i])
            return flipped
        return img_group


class GroupScale:
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        return [transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR)(img) for img in img_group]


class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        # Convert PIL to tensor first, shape (C, H, W)
        img_group = [transforms.ToTensor()(img) for img in img_group]
        # Concatenate along time (frame) dimension
        return torch.cat(img_group, dim=0)


class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        c = tensor.size(0)
        rep_mean = self.mean.repeat(c // 3, 1, 1)
        rep_std = self.std.repeat(c // 3, 1, 1)
        return (tensor - rep_mean) / rep_std


def train_transforms():
    return transforms.Compose([
        GroupScale((256, 256)),
        GroupRandomCrop(224),
        GroupRandomHorizontalFlip(),
        Stack(),
        GroupNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def val_transforms():
    return transforms.Compose([
        GroupScale((224, 224)),
        GroupCenterCrop(224),
        Stack(),
        GroupNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
