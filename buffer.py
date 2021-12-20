
from math import e
import numpy as np
import random
from collections import namedtuple, deque
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, rolls=32, agents=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (string): GPU or CPU
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.memory = [[[] for agent in range(agents)] for roll in range(rolls)]
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "a_probs"])
        self.device = device
        self.rolls = rolls
        self.agents = agents
    
    def add(self, state, action, reward, done, a_probs, masks):
        for i in masks:
            roll = i[0]
            agent = i[1]
            vis_obs = state[0]
            vec_obs = state[1]
            if state[0]:
                vis_o = vis_obs[roll][agent]
            else:
                vis_o = None
            e = self.experience((vis_o, vec_obs[roll, agent]), action[roll, agent], reward[roll, agent], done[roll, agent], a_probs[roll, agent])
            if len(self.memory[roll][agent])>2 and self.memory[roll][agent][-1].done and e.done:
                # print('double 2!')
                continue
            self.memory[roll][agent].append(e)
        a = 1

    def sample(self):
        experiences = []
        for i in range(self.rolls):
            for j in range(self.agents):
                cur_experiences = random.sample(self.memory[i][j],k=self.batch_size)
                experiences += cur_experiences
        if experiences[0].state[0]:
            vis_obs = torch.from_numpy(np.array([e.state[0] for e in experiences if e is not None])).float().to(self.device)
        else:
            vis_obs = None
        vec_obs = torch.from_numpy(np.array([e.state[1] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)

        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return ((vis_obs, vec_obs), actions, rewards, dones)

    def sample_one_agent(self, roll, agent):
        memo = self.memory[roll][agent]
        # for i in range(len(memo)-1, -1, -1):
        #     exp = memo[i]
        #     if exp.done:
        #         break

        # if i == 0:
        #     return (None, None, None, None, None)
        # experiences = memo[:i+1]
        # if experiences[-2].done:
        #     print('double 1!')
        # self.memory[roll][agent][:] = memo[i+1:]
        experiences = memo[:]
        if experiences[0].state[0]:
            vis_obs = torch.from_numpy(np.array([e.state[0] for e in experiences if e is not None])).float().to(self.device)
        else:
            vis_obs = None
        vec_obs = torch.from_numpy(np.array([e.state[1] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)

        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        a_probs = torch.from_numpy(np.array([e.a_probs for e in experiences if e is not None])).float().to(self.device)
        return ((vis_obs, vec_obs), actions, rewards.squeeze(), dones, a_probs)

    def clear(self):
        self.memory = [[[] for agent in range(self.agents)] for roll in range(self.rolls)]
        return
        for roll in self.rolls:
            for agent in self.agents:
                memo = self.memory[roll][agent]
                for i in range(len(memo)-1, -1, -1):
                    exp = memo[i]
                    if exp.done:
                        break
                self.memory[roll][agent][:] = memo[i+1:]

    def __len__(self):
        return len(self.memory)