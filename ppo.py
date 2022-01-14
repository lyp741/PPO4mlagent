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
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
step = 0


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
        self.lambda_entropy = 0.01  # could be 0.01~0.05
        self.lambda_gae_adv = 0.95  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.trajectory_list = None
        self.replay = None

    def init(self, net_dim, vis_obs_shape, vec_obs_shape, action_dim, env, learning_rate=1e-4, if_use_gae=False, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if not if_use_gae else self.get_reward_sum_raw
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
        # self.clear_buffer()
        self.cumulative_reward = []
        self.rewards = [[[] for _ in range(self.num_agents)] for _ in range(self.num_rolls)]
        del self.ClassCri, self.ClassAct

    def clear_buffer(self):
        self.buffer = [[[] for agent in range(self.num_agents)] for roll in range(self.num_rolls)]




    def select_action(self, state):
        # states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(state)
        return actions.squeeze().detach().cpu().numpy(), noises.detach().cpu().numpy()

    def explore_env(self, env, target_step):
        

        state = self.state
        rolls = state[1].shape[0]
        agents= state[1].shape[1]
        last_done = 0
        actions = np.zeros((rolls, agents))
        a_probs = np.zeros((rolls, agents, self.action_dim))
        for i in range(target_step):
            for r in range(rolls):
                if state[0] is not None:
                    vis_state = state[0][r]
                else:
                    vis_state = None
                action, a_prob = self.select_action((vis_state, state[1][r]))  # different
                actions[r] = action
                a_probs[r] = a_prob
            next_state, reward, done, _, masks, ds, ts = env.step(actions)  # different
            if vis_state is not None:
                vis_state = state[0].copy()
            else:
                vis_state = None
            self.buffer.add((vis_state,state[1].copy()), actions.copy(), reward.copy(), done.copy(), a_probs.copy(), masks)  # different

            for (roll, agent) in masks:
                if state[0] is not None:
                    state[0][roll][agent] = next_state[0][roll][agent]
                state[1][roll][agent] = next_state[1][roll][agent]
                self.rewards[roll][agent].append(reward[roll][agent].copy())
                if done[roll][agent]:
                    self.cumulative_reward.append(sum(self.rewards[roll][agent]))
                    self.rewards[roll][agent] = []
        self.state = state

        '''splice list'''
        # trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        # self.trajectory_list \
        #     = trajectory_temp[last_done:]
        return self.buffer


    def update_net(self, batch_size, repeat_times, soft_update_tau):

        buf_vis_obs = []
        buf_vec_obs = []
        buf_action = []
        buf_r_sum = []
        buf_logprob = []
        buf_advantage = []
        global writer, step
        step+=1
        for roll in range(self.num_rolls):
            for agent in range(self.num_agents):
                with torch.no_grad():
                    buf_state2, buf_action2, buf_reward2, buf_mask2, buf_noise2 = self.buffer.sample_one_agent(roll, agent)
                    # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer
                    if buf_state2 == None or buf_state2[1].shape[0] == 1:
                        continue
                    buf_len = buf_state2[1].shape[0]
                    vis_obs = buf_state2[0]
                    vec_obs = buf_state2[1]
                    '''get buf_r_sum, buf_logprob'''
                    bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.

                    buf_value = self.cri((vis_obs, vec_obs)).squeeze()
                    # buf_value = torch.cat(buf_value, dim=0).squeeze()
                    buf_logprob.append(self.act.get_old_logprob(buf_action2, buf_noise2))

                    buf_r_sum2, buf_advantage2 = self.get_reward_sum_gae(buf_len, buf_reward2, buf_mask2, buf_value)  # detach()
                    # buf_advantage2 = (buf_advantage2 - buf_advantage2.mean()) / (buf_advantage2.std() + 1e-10)
                    buf_vis_obs.append(vis_obs)
                    buf_vec_obs.append(vec_obs)
                    buf_action.append(buf_action2)
                    buf_r_sum.append(buf_r_sum2)
                    buf_advantage.append(buf_advantage2)
        if buf_vis_obs[0] is not None:
            buf_vis_obs = torch.cat(buf_vis_obs, dim=0)
        else:
            buf_vis_obs = None
        buf_vec_obs = torch.cat(buf_vec_obs, dim=0)
        buf_action = torch.cat(buf_action, dim=0)
        buf_r_sum = torch.cat(buf_r_sum, dim=0)
        buf_logprob = torch.cat(buf_logprob, dim=0)
        buf_advantage = torch.cat(buf_advantage, dim=0)
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)

        buf_len = buf_vec_obs.shape[0]
        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for _ in range(3):
            # indices = torch.randint(buf_len, size=(128,), requires_grad=False, device=self.device)
            for i in range(int(buf_len/128)-1):
                indices = torch.randint(buf_len, size=(128,), requires_grad=False, device=self.device)
                if buf_vis_obs is not None:
                    vis = buf_vis_obs[indices]
                else:
                    vis = None
                state = (vis, buf_vec_obs[indices])
                action = buf_action[indices]
                r_sum = buf_r_sum[indices]
                logprob = buf_logprob[indices]
                advantage = buf_advantage[indices]
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = advantage * ratio
                surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate - obj_entropy * self.lambda_entropy
                self.optim_update(self.act_optim, obj_actor)

                value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        # self.buffer.clear()
        writer.add_scalar('PPO/obj_actor', obj_actor.item(), step)
        writer.add_scalar('PPO/obj_critic', obj_critic.item(), step)
        writer.add_scalar('PPO/entropy', obj_entropy.item(), step)
        writer.add_scalar('PPO/advantage', advantage.mean().item(), step)
        if len(self.cumulative_reward)>1:
            self.cumulative_reward = self.cumulative_reward[-32:]
            writer.add_scalar('PPO/rewards', sum(self.cumulative_reward)/len(self.cumulative_reward), step)
        self.buffer.clear()
        
    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        buf_mask = 1 - buf_mask  # mask
        buf_mask = buf_mask*0.99
        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_r_sum = (buf_r_sum.detach() - buf_r_sum.mean ()) / (buf_r_sum.std() + 1e-6)
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value
        ten_mask = 1 - ten_mask  # mask
        ten_mask = ten_mask*0.99
        ten_bool = torch.not_equal(ten_mask, 0).float()

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_bool[i] * (pre_advantage - ten_value[i])  # fix a bug here
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

    def save_model(self, path):
        torch.save(self.act.state_dict(), path)

    def load_model(self, path):
        self.act.load_state_dict(torch.load(path))
