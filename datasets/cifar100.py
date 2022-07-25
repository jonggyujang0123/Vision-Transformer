"""
Cifar10 Dataloader implementation, used in CondenseNet
"""
import logging
import numpy as np

import torch
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, Normalize, Resize
from torchvision.datasets import CIFAR100

from torch.utils.data import DataLoader, TensorDataset, Dataset


class CIFAR100DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Cifar100DataLoader")

        self.logger.info("Loading DATA.....")
        normalize = Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675,0.2565, 0.2761))

        train_set = CIFAR100('./data', train=True, download=True,
                            transform=Compose([RandomHorizontalFlip(),
                                               RandomVerticalFlip(),
                                               Resize(config.im_size),
                                               ToTensor(),
                                               normalize]))

        valid_set = CIFAR100('./data', train=False,
                            transform=Compose([Resize(config.im_size),
                                                ToTensor(),
                                                normalize]))

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)
        self.test_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
        self.test_iterations = len(self.test_loader)

    def finalize(self):
        pass
