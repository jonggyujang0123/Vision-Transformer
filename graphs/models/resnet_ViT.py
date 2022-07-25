# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence, TypeVar
import torch.nn as nn
import torch
T = TypeVar('T')
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F

import logging

class ResidualUnit(nn.Module):
    """
    features: int
    strides: Sequence[int] = (1, 1)
     """
    """Bottleneck ResNet block."""
    def __init__(self, in_channels, features: int, stride= (1, 1)):
        super().__init__()
        self.logger = logging.getLogger("ResNet")
        self.features = features
        self.stride = stride
        self.projection = nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, self.features*4, kernel_size=(1,1), stride=self.stride ,bias=False, padding='valid')),
                nn.GroupNorm(num_groups=self.features*4,num_channels=self.features*4))
        self.layer1 = nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, self.features, kernel_size=(1,1), stride=(1,1) ,bias=False, padding='valid')),
                nn.ReLU())
        self.layer2 = nn.Sequential(
                weight_norm(nn.Conv2d(self.features, self.features, kernel_size=(3,3), stride=self.stride ,bias=False, padding='valid')),
                nn.GroupNorm(num_groups=self.features,num_channels=self.features),
                nn.ReLU())
        self.layer3 = nn.Sequential(
                weight_norm(nn.Conv2d(self.features, self.features*4, kernel_size=(1,1), stride=(1,1) ,bias=False, padding='valid')),
                nn.GroupNorm(num_groups=self.features*4,num_channels=self.features*4))


    def forward(self, x):
        needs_projection = (x.size(-1) != self.features*4 or self.strides !=(1,1))
        residual = x
        if needs_projection:
            residual = self.projection(residual)
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        return nn.ReLU(y+residual)

class ResNetStage(nn.Module):
    """A ResNet stage."""
    def __init__(self,in_channels, block_size, nout, first_stride):
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        self.nout= nout
        self.first_stride = first_stride
        self.model = nn.Sequential()
        self.model.append(ResidualUnit(self.in_channels,self.nout, stride=self.first_stride))
        for i in range(1,self.block_size):
            self.model.append(ResidualUnit(self.nout*4, self.nout, stride=(1,1)))
    def forward(self,x):
        x = self.model(x)
        return x
