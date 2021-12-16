
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
        self.memory = [[deque(maxlen=int(buffer_size/rolls/agents)) for agent in range(agents)] for roll in range(rolls)]
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
            e = self.experience((vis_obs[roll, agent], vec_obs[roll, agent]), action[roll, agent], reward[roll, agent], done[roll, agent], a_probs[roll, agent])
            self.memory[roll][agent].append(e)
    
    def sample(self):
        experiences = []
        for i in range(self.rolls):
            for j in range(self.agents):
                cur_experiences = random.sample(self.memory[i][j],k=self.batch_size)
                experiences += cur_experiences
        vis_obs = torch.from_numpy(np.array([e.state[0] for e in experiences if e is not None])).float().to(self.device)
        vec_obs = torch.from_numpy(np.array([e.state[1] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)

        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return ((vis_obs, vec_obs), actions, rewards, dones)

    def sample_one_agent(self, roll, agent):
        experiences = self.memory[roll][agent]
        vis_obs = torch.from_numpy(np.array([e.state[0] for e in experiences if e is not None])).float().to(self.device)
        vec_obs = torch.from_numpy(np.array([e.state[1] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)

        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        a_probs = torch.from_numpy(np.array([e.a_probs for e in experiences if e is not None])).float().to(self.device)
        return ((vis_obs, vec_obs), actions, rewards, dones, a_probs)

    def clear(self):
        self.memory = [[deque(maxlen=int(self.buffer_size/self.rolls/self.agents)) for agent in range(self.agents)] for roll in range(self.rolls)]

    def __len__(self):
        return len(self.memory)