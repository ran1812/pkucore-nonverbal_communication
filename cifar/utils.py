import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Sequential):
    def __init__(self, c):
        super().__init__()
        for x, (i, o) in enumerate(zip(c[:-1], c[1:])):
            self.add_module(f"{x}", nn.Conv2d(i, o, kernel_size=3, stride=2, padding=1))
            self.add_module(f"{x}bn", nn.BatchNorm2d(o))
            self.add_module(f"{x}ac", nn.LeakyReLU())
        self.add_module(f"flatten", nn.Flatten())

class Linear(nn.Sequential):
    def __init__(self, *d):
        super().__init__()
        for x, (i, o) in enumerate(zip(d[:-1], d[1:])):
            self.add_module(f"{x}", nn.Linear(i, o))
            self.add_module(f"{x}bn", nn.BatchNorm1d(o))
            self.add_module(f"{x}ac", nn.LeakyReLU())
