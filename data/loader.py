from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import torch
import os

base = os.path.dirname(__file__)

transforms_train = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    v2.RandomErasing(p=0.25, scale=(0.02, 0.15))
])
transforms_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
])

cifar10_dataset = CIFAR10(base, train=True, download=True, transform=transforms_train)
cifar10_dataset_test = CIFAR10(base, train=False, download=True, transform=transforms_test)
