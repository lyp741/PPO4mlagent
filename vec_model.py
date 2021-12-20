import torch
import torch.nn as nn

from typing import Tuple
from math import floor

class ActorDiscretePPO(nn.Module):
    def __init__(self, vis, vec, action_dim):
        super().__init__()
        mid_dim = 64
        state_dim = vec
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        state = state[1]
        state = torch.tensor(state).float().to(self.device)

        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        a_prob = self.soft_max(self.forward(state))
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state[1].shape[0]).unsqueeze(0)
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        out = self.forward(state)
        a_prob = self.soft_max(out)
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class Critic(nn.Module):
    def __init__(self, vis, vec, _action_dim):
        super().__init__()
        mid_dim = 64
        state_dim = vec
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state):
        state = state[1]
        return self.net(state)  # Advantage value
