import torch
import torch.nn as nn

from typing import Tuple
from math import floor


class ActorDiscretePPO(nn.Module):
    def __init__(self,  vis_shape=0, vec_shape=0,
                 encoding_size=64,
                 output_size=7):
        super().__init__()
        self.vis_shape = vis_shape
        if vis_shape:
            height = vis_shape[0]
            width = vis_shape[1]
            initial_channels = vis_shape[2]
            conv_1_hw = self.conv_output_shape((height, width), 8, 4)
            conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
            self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
            self.conv1 = torch.nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
            self.conv2 = torch.nn.Conv2d(16, 32, [4, 4], [2, 2])
        else:
            self.final_flat = 0
        self.vec_shape = vec_shape
        self.dense1 = torch.nn.Linear(
            self.final_flat+self.vec_shape, encoding_size)
        self.dense2 = torch.nn.Linear(encoding_size, output_size)
        self.action_dim = output_size
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        if self.vis_shape:
            visual_obs = torch.tensor(state[0]).to(self.device)
            visual_obs = visual_obs.float()
            visual_obs = visual_obs.permute(0, 3, 1, 2)
            conv_1 = torch.relu(self.conv1(visual_obs))
            conv_2 = torch.relu(self.conv2(conv_1))
            conv_out = conv_2.reshape([-1, self.final_flat])
        vec_obs = torch.tensor(state[1]).to(self.device)
        vec_obs = vec_obs.float()
        if self.vis_shape:
            concat = torch.cat((conv_out, vec_obs), 1)
        else:
            concat = vec_obs
        hidden = self.dense1(concat)
        hidden = torch.relu(hidden)
        hidden = self.dense2(hidden)
        return hidden

    def get_action(self, state):
        a_prob = self.soft_max(self.forward(state))
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state[1].shape[0])
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        a_prob = self.soft_max(self.forward(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)

    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
            ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
            ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w


class Critic(torch.nn.Module):
    def __init__(
        self,
        vis_shape: Tuple[int, int, int],
        vec_shape=None,
        output_size=7,
        encoding_size=64
    ):
        """
        Creates a neural network that takes as input a batch of images (3
        dimensional tensors) and outputs a batch of outputs (1 dimensional
        tensors)
        """
        super(Critic, self).__init__()
        self.vis_shape = vis_shape
        if vis_shape:
            height = vis_shape[0]
            width = vis_shape[1]
            initial_channels = vis_shape[2]
            conv_1_hw = self.conv_output_shape((height, width), 8, 4)
            conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
            self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
            self.conv1 = torch.nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
            self.conv2 = torch.nn.Conv2d(16, 32, [4, 4], [2, 2])
        else:
            self.final_flat = 0
        self.vec_shape = vec_shape
        self.dense1 = torch.nn.Linear(
            self.final_flat+self.vec_shape, encoding_size)
        self.dense2 = torch.nn.Linear(encoding_size, 1)
        self.action_dim = 1
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        if self.vis_shape:
            visual_obs = torch.tensor(state[0]).to(self.device)
            visual_obs = visual_obs.float()
            visual_obs = visual_obs.permute(0, 3, 1, 2)
            conv_1 = torch.relu(self.conv1(visual_obs))
            conv_2 = torch.relu(self.conv2(conv_1))
            conv_out = conv_2.reshape([-1, self.final_flat])
        vec_obs = torch.tensor(state[1]).to(self.device)
        vec_obs = vec_obs.float()
        if self.vis_shape:
            concat = torch.cat((conv_out, vec_obs), 1)
        else:
            concat = vec_obs
        hidden = self.dense1(concat)
        hidden = torch.relu(hidden)
        hidden = self.dense2(hidden)
        return hidden

    @staticmethod
    def conv_output_shape(
        h_w: Tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
            ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
            ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w
