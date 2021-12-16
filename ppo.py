import os
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy
from model import ActorDiscretePPO, Critic
from buffer import ReplayBuffer

class AgentPPO:
    def __init__(self):
        super().__init__()
        self.state = None
        self.device = None
        self.action_dim = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

        '''init modify'''
        self.ClassCri = Critic
        self.ClassAct = ActorDiscretePPO

        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.trajectory_list = None

    def init(self, net_dim, vis_obs_shape, vec_obs_shape, action_dim, env, learning_rate=1e-4, if_use_gae=False, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.action_dim = action_dim
        self.cri = self.ClassCri(vis_obs_shape, vec_obs_shape, action_dim).to(self.device)
        self.act = self.ClassAct(vis_obs_shape, vec_obs_shape, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        self.buffer = ReplayBuffer(buffer_size=1e6, batch_size=64, device=self.device, rolls=env.num_rolls, agents=env.num_agents)
        self.num_rolls = env.num_rolls
        self.num_agents = env.num_agents
        del self.ClassCri, self.ClassAct

    def select_action(self, state):
        # states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(state)
        return actions.detach().cpu().numpy(), noises.detach().cpu().numpy()

    def explore_env(self, env, target_step):
        

        state = self.state
        rolls = state[0].shape[0]
        agents= state[0].shape[1]
        last_done = 0
        actions = np.zeros((rolls, agents))
        a_probs = np.zeros((rolls, agents, self.action_dim))
        for i in range(target_step):
            for r in range(rolls):
                action, a_prob = self.select_action((state[0][r], state[1][r]))  # different
                actions[r] = action
                a_probs[r] = a_prob
            next_state, reward, done, _, masks = env.step(actions)  # different
            self.buffer.add(state, actions, reward, done, a_probs, masks)  # different

            for (roll, agent) in masks:
                state[0][roll][agent] = next_state[0][roll][agent]
                state[1][roll][agent] = next_state[1][roll][agent]
        self.state = state

        '''splice list'''
        # trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        # self.trajectory_list \
        #     = trajectory_temp[last_done:]
        return self.buffer


    def update_net(self, batch_size, repeat_times, soft_update_tau):
        for roll in range(self.num_rolls):
            for agent in range(self.num_agents):
                with torch.no_grad():
                    buf_state, buf_action, buf_reward, buf_mask, buf_noise = self.buffer.sample_one_agent(roll, agent)
                    buf_len = buf_state[0].shape[0]
                    # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

                    '''get buf_r_sum, buf_logprob'''
                    bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
                    buf_value = [self.cri_target((buf_state[0][i:i + bs], buf_state[1][i:i+bs])) for i in range(0, buf_len, bs)]
                    buf_value = torch.cat(buf_value, dim=0)
                    buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

                    buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
                    buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)

                '''PPO: Surrogate objective of Trust Region'''
                obj_critic = obj_actor = None
                for _ in range(int(buf_len / batch_size * repeat_times)):
                    indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

                    state = (buf_state[0][indices], buf_state[1][indices])
                    action = buf_action[indices]
                    r_sum = buf_r_sum[indices]
                    logprob = buf_logprob[indices]
                    advantage = buf_advantage[indices]

                    new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                    ratio = (new_logprob - logprob.detach()).exp()
                    surrogate1 = advantage * ratio
                    surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                    obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                    obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                    self.optim_update(self.act_optim, obj_actor)

                    value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                    obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
                    self.optim_update(self.cri_optim, obj_critic)
                    self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

                a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
        self.buffer.clear()
        return obj_critic.item(), obj_actor.item(), a_std_log.mean().item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1.0 - tau))
