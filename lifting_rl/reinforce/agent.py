import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lifting_rl.reinforce.policy import Policy
import math
import torch.nn.utils as utils


def normal(x: torch.Tensor, mu: torch.Tensor, sigma_sq: torch.Tensor):
    device = x.device
    pi = torch.FloatTensor([math.pi]).to(device)
    a = (-1*(x-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a*b


class ReinforceAgent:
    def __init__(
        self,
        env,
        hidden_dim: int = 128,
        lr: float = 0.01,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        self.num_states = env.state_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.device = device
        self.policy = Policy(
            in_dim=self.num_states,
            hid_dim=hidden_dim,
            out_dim=self.num_actions
        ).to(device)
        self.policy.train()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.memory = []

    def reset(self):
        self.memory.clear()

    def get_action(self, state, step):
        device = self.device
        pi = torch.FloatTensor([math.pi]).to(device)
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(device)
        mu, sigma_sq = self.policy(state)
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size()).to(device)
        # calculate the probability
        action = torch.normal(mu, sigma_sq)
        # action = (mu + sigma_sq.sqrt()*eps).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)
        log_prob = prob.log()

        action_val = action.detach().cpu().view(-1).numpy()
        return action_val, (log_prob, entropy, mu.item(), sigma_sq.item())

    def finish_step(self, state, action, new_state, reward, done, meta):
        entry = list(meta[:2]) + [reward]
        self.memory.append(entry)

    def update(self):
        device = self.device
        R = torch.zeros(1, 1).to(device)
        gamma = self.gamma
        loss = 0

        for log_prob, entropy, reward in self.memory:
            R = gamma * R + reward
            loss = loss - (log_prob * R.expand_as(log_prob)).sum() - (0.0001 * entropy).sum()

        loss = loss / len(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.policy.parameters(), 40)
        self.optimizer.step()
        self.memory.clear()
