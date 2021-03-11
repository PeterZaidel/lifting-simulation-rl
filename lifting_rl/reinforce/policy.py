from typing import Tuple
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        dropout_prob: float = 0.6
    ) -> None:
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.linear2_sigma = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        mu = self.linear2(x)
        sigma_sq = self.linear2_sigma(x)
        return mu, sigma_sq
