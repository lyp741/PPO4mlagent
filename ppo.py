import os
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy
from vec_model import ActorDiscretePPO, Critic
from buffer import ReplayBuffer
from collections import namedtuple, deque
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
        self.last_obs = None
        self.last_action = None
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "a_probs", "next"])


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
        self.rewards = [[[] for agent in range(self.num_agents)] for roll in range(self.num_rolls)]
        self.cumulative_rewards = []
        self.clear_buffer()
        self.last_obs = [[None for agent in range(self.num_agents)] for roll in range(self.num_rolls)]
        self.last_action = [[None for agent in range(self.num_agents)] for roll in range(self.num_rolls)]
        self.last_a_probs = [[None for agent in range(self.num_agents)] for roll in range(self.num_rolls)]
        del self.ClassCri, self.ClassAct

    def clear_buffer(self):
        self.buffer = [[[] for agent in range(self.num_agents)] for roll in range(self.num_rolls)]




    def select_action(self, state):
        # states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(state)
        return actions.squeeze().detach().cpu().numpy(), noises.detach().cpu().numpy()

    def process_states(self, dss, tss):
        for agent_id_t in tss:
            ts = tss[agent_id_t]
            r = ts.group_id - 1
            a = agent_id_t % self.num_agents
            last_exp = self.experience(
                state=self.last_obs[r][a].copy(),
                action=self.last_action[r][a].copy(),
                reward=ts.reward,
                done=True,
                a_probs=self.last_a_probs[r][a].copy(),
                next=np.concatenate((ts.obs[0], ts.obs[1]))
            )
            self.last_obs[r][a] = None
            self.last_action[r][a] = None
            self.last_a_probs[r][a] = None
            self.buffer[r][a].append(last_exp)
            self.rewards[r][a].append(ts.reward)
            self.cumulative_rewards.append(sum(self.rewards[r][a]))
            self.rewards[r][a] = []
            
        for agent_id_d in dss:
            ds = dss[agent_id_d]
            r = ds.group_id - 1
            a = agent_id_d % self.num_agents
            if self.last_obs[r][a] is not None:
                last_exp = self.experience(
                    state=self.last_obs[r][a].copy(),
                    action=self.last_action[r][a].copy(),
                    reward=ds.reward,
                    done=False,
                    a_probs=self.last_a_probs[r][a].copy(),
                    next=np.concatenate((ds.obs[0], ds.obs[1]))
                )
                self.buffer[r][a].append(last_exp)
                self.rewards[r][a].append(ds.reward)
            self.last_obs[r][a] = (
                np.concatenate((ds.obs[0], ds.obs[1]))
            )

    def explore_env(self, env, target_step):
        

        state = self.state
        
        last_done = 0
        states = np.zeros((self.num_rolls, self.num_agents, env.obs_shape[0]))
        
        actions = np.zeros((self.num_rolls, self.num_agents))
        a_probs = np.zeros((self.num_rolls, self.num_agents, self.action_dim))
        for i in range(target_step):
            for r in range(self.num_rolls):
                for a in range(self.num_agents):
                    if self.last_obs[r][a] is not None:
                        states[r][a] = self.last_obs[r][a]
            for r in range(self.num_rolls):
                action, a_prob = self.select_action((None, states[r]))  # different
                actions[r] = action
                a_probs[r] = a_prob
            next_state, reward, done, _, masks, dss, tss = env.step(actions)  # different
            # self.buffer.add((None,state[1]), actions, reward, done, a_probs, masks)  # different

            # for (roll, agent) in masks:
            #     state[1][roll][agent] = next_state[1][roll][agent]
            for agent_id_d in dss:
                ds = dss[agent_id_d]
                r = ds.group_id - 1
                a = agent_id_d % self.num_agents
                self.last_action[r][a] = actions[r,a].copy()
                self.last_a_probs[r][a] = a_probs[r,a].copy()
            self.process_states(dss, tss)

                
        
        self.state = state

        '''splice list'''
        # trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        # self.trajectory_list \
        #     = trajectory_temp[last_done:]
        return self.buffer

    def sample_one_agent(self, roll, agent):
        memo = self.buffer[roll][agent]
        experiences = memo[:]
        
        vis_obs = None
        vec_obs = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)

        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        a_probs = torch.from_numpy(np.array([e.a_probs for e in experiences if e is not None])).float().to(self.device)
        next  = torch.from_numpy(np.array([e.next for e in experiences if e is not None])).float().to(self.device)
        return ((vis_obs, vec_obs), actions, rewards.squeeze(), dones, a_probs, next)

    def update_net(self, batch_size, repeat_times, soft_update_tau):
        global writer, step
        batch_size = 128
        step+=1
        buf_state = []
        buf_action = []
        buf_r_sum = []
        buf_logprob = []
        buf_advantage = []
        buf_next_obs = []
        buf_reward = []
        buf_done = []
        for roll in range(self.num_rolls):
            for agent in range(self.num_agents):
                with torch.no_grad():
                    buf_state2, buf_action2, buf_reward2, buf_mask2, buf_noise2, next_obs = self.sample_one_agent(roll, agent)
                    # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer
                    if buf_state2 == None or buf_state2[1].shape[0] == 1:
                        continue
                    buf_len = buf_state2[1].shape[0]
                    bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
                    buf_value = self.cri((None, buf_state2[1])).squeeze()
                    next_v = self.cri((None, next_obs)).squeeze()
                    buf_logprob.append(self.act.get_old_logprob(buf_action2, buf_noise2))
                    buf_r_sum2, buf_advantage2 = self.myGAE(buf_len, buf_reward2, buf_mask2, buf_value, next_v)  # detach()
                    # buf_advantage2 = (buf_advantage2 - buf_advantage2.mean()) / buf_advantage2.std()
                    buf_state.append(buf_state2[1])
                    buf_action.append(buf_action2)
                    buf_r_sum.append(buf_r_sum2)
                    buf_advantage.append(buf_advantage2)
                    buf_next_obs.append(next_obs)
                    buf_reward.append(buf_reward2)
                    buf_done.append(buf_mask2)

        buf_state = torch.cat(buf_state, dim=0)
        buf_action = torch.cat(buf_action, dim=0)
        buf_r_sum = torch.cat(buf_r_sum, dim=0)
        buf_logprob = torch.cat(buf_logprob, dim=0)
        buf_advantage = torch.cat(buf_advantage, dim=0)
        buf_next_obs = torch.cat(buf_next_obs, dim=0)
        buf_reward = torch.cat(buf_reward, dim=0)
        buf_done = torch.cat(buf_done, dim=0)
        buf_len = buf_state.shape[0]
        buf_advantage = (buf_advantage / (buf_advantage.std() + 1e-3))
        # buf_r_sum = (buf_r_sum - buf_r_sum.mean()) / (buf_r_sum.std() + 1e-10)
        '''PPO: Surrogate objective of Trust Region'''
                
        obj_critic = obj_actor = None
        for _ in range(3):
            # indices = torch.randint(buf_len, size=(128,), requires_grad=False, device=self.device)
            for i in range(0, buf_len, batch_size):
                indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)
                state = (None, buf_state[indices])
                action = buf_action[indices]
                r_sum = buf_r_sum[indices]
                logprob = buf_logprob[indices]
                advantage = buf_advantage[indices]
                next_obs = buf_next_obs[indices]
                reward = buf_reward[indices]
                done = buf_done[indices]
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = advantage * ratio
                surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate - obj_entropy * self.lambda_entropy
                self.optim_update(self.act_optim, obj_actor, self.act)

                value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                
                obj_critic = self.criterion(value, r_sum.detach()) / (r_sum.std() + 1e-6)
                self.optim_update(self.cri_optim, obj_critic, self.cri)

        # self.buffer.clear()
        writer.add_scalar('PPO/obj_actor', obj_actor.item(), step)
        writer.add_scalar('PPO/obj_critic', obj_critic.item(), step)
        writer.add_scalar('PPO/entropy', obj_entropy.item(), step)
        writer.add_scalar('PPO/advantage', advantage.mean().item(), step)
        if len(self.cumulative_rewards) > 0:
            writer.add_scalar('PPO/rewards', sum(self.cumulative_rewards)/ len(self.cumulative_rewards), step)
            self.cumulative_rewards = self.cumulative_rewards[-32:]
        self.clear_buffer()
        
    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        ten_mask = 1 - buf_mask  # mask
        ten_mask = ten_mask*0.99
        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:]
        return buf_r_sum, buf_adv_v

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

    def myGAE(self, buf_len, ten_reward, ten_mask, ten_value, next_v):
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for i in range(buf_len - 1, -1, -1):
            if ten_mask[i] == 1:
                delta = ten_reward[i] - ten_value[i]
                last_gae = delta
            else:
                delta = ten_reward[i] + 0.99 * next_v[i] - ten_value[i]
                last_gae = delta + 0.99 * self.lambda_gae_adv * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + ten_value[i])
        adv_v = torch.tensor(list(reversed(result_adv))).to(self.device)
        ref_v = torch.tensor(list(reversed(result_ref))).to(self.device)
        # ref_v = (ref_v - ref_v.mean()) / (ref_v.std() + 1e-10)
        return ref_v, adv_v


    @staticmethod
    def optim_update(optimizer, objective, model):
        optimizer.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=0.5
                    )
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1.0 - tau))
