from torch import nn
from typing import Tuple

import torch

class Jarvis():
    def __init__(self, dim_in: Tuple[int, int], dim_out: Tuple[int, int]):
        self.h = dim_in[0]
        self.w = dim_in[1]
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.features = nn.Sequential(
                            nn.Conv2d(1, 16, kernel_size=3, padding=1), # shape: (1, h, w) -> (16, h, w)
                            nn.ReLU(),
                            nn.BatchNorm2d(num_features=16),
                            nn.Conv2d(16, 32, kernel_size=3, padding=1), # shape: (16, h, w) -> (32, h, w)
                            nn.ReLU(),
                            nn.BatchNorm2d(num_features=32),
                            nn.Conv2d(32, 32, kernel_size=3, padding=1), # shape: (32, h, w) -> (32, h , w)
                            nn.ReLU(),
                            nn.BatchNorm2d(num_features=32),
                            nn.AdaptiveAvgPool2d(output_size=(1, 8)) # shape: (32, h, w) -> (8, h, w)
                        )
        
        self.classifier = nn.Sequential(
                            nn.Linear(256, 64), # shape: (256) -> (64)
                            nn.ReLU(),
                            nn.Linear(64, 1), # shape: (64) -> (1)
                            nn.Sigmoid()
                        )
        
    def foward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # shape: (b, 32, 1, 8) -> (b, 256)
        x = self.classifier(x)
        return x
