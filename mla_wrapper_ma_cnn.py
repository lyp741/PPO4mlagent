import mlagents
from mlagents_envs.base_env import DecisionStep, TerminalStep

from mlagents_envs.environment import ActionTuple, BaseEnv
from typing import Dict
import random
import numpy as np
import time
from gym import spaces
from mlagents_envs.environment import UnityEnvironment

class MLA_Wrapper():
    def __init__(self):
        print('Please run Unity...')
        self.env = UnityEnvironment()
        self.reset()
        self.observation_space = []
        self.action_space = []
        for i in range(self.num_agents):
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=self.spec.observation_specs[0].shape, dtype=np.float32))  # [-inf,inf]
            self.action_space.append(spaces.Discrete(self.spec.action_spec.discrete_branches[0]))
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(20,20,6*self.num_agents), dtype=np.float32) for _ in range(self.num_agents)]

    def reset(self):
        print('reseting...')
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0] 
        print(f"Name of the behavior : {self.behavior_name}")
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        self.num_agents = None
        self.num_rolls = None
        decisionStep, terminalStep = self.env.get_steps(self.behavior_name)
        self.decisionStep = decisionStep
        self.terminalStep = terminalStep
        self.vis_idx = []
        self.vec_idx = []
        for i in range(len(decisionStep.obs)):
            obs_i = decisionStep.obs[i]
            obs_i_shape = obs_i.shape
            if len(obs_i_shape) == 4:
                self.vis_idx.append(i)
            else:
                self.vec_idx.append(i)
        vec_obs_raw = None
        vis_obs_raw = None
        if self.vis_idx:
            vis_obs_raw = decisionStep.obs[self.vis_idx[0]]
        if self.vec_idx:
            for i in self.vec_idx:
                tmp = decisionStep.obs[i]
                if vec_obs_raw is not None:
                    vec_obs_raw = np.concatenate((vec_obs_raw, tmp), axis=1)
                else:
                    vec_obs_raw = tmp
        # vis_obs_raw = decisionStep.obs[0] # 2 obs, 0 is grid sensor. (agents*platform, 20, 20, 7)
        # vec_obs_raw = decisionStep.obs[1] # 2 obs, 1 is vector. (agents*platform, 10)
        groupId = decisionStep.group_id
        self.groupId = groupId
        self.num_rolls = np.unique(groupId).shape[0]
        # self.num_rolls = len(groupId)
        print("num rolls: ", self.num_rolls)
        self.num_agents = int(len(groupId) / self.num_rolls)
        # self.num_agents = 1
        print("num agents: ", self.num_agents)
        agentsidx_in_roll = {}
        self.agentid2idx = {}
        for i in decisionStep:
            gid = decisionStep[i].group_id
            if gid-1 not in agentsidx_in_roll:
                agentsidx_in_roll[gid-1] = 0
            else:
                agentsidx_in_roll[gid-1] += 1
            self.agentid2idx[i] = agentsidx_in_roll[gid-1]
        print('ds.group_id: ', decisionStep.group_id)
        print('agentid2idx: ', self.agentid2idx)
        if self.vis_idx:
            vis_obs = np.zeros((self.num_rolls, self.num_agents)+vis_obs_raw.shape[1:])
            self.vis_obs_shape = vis_obs_raw.shape[1:]
        else:
            vis_obs = None
            self.vis_obs_shape = 0
        if self.vec_idx:
            vec_obs = np.zeros((self.num_rolls, self.num_agents)+vec_obs_raw.shape[1:])
            self.vec_obs_shape = vec_obs_raw.shape[1:][0]
        else:
            vec_obs = None
            self.vec_obs_shape = 0
        
        reward = decisionStep.reward #(agents*platform,)

        rewards = np.zeros((self.num_rolls, self.num_agents, 1))
        for i in decisionStep:
            roll = decisionStep[i].group_id-1
            agent = self.agentid2idx[i]
            if vis_obs is not None:
                vis_obs[roll, agent] = vis_obs_raw[i]
            if vec_obs is not None:
                vec_obs[roll, agent] = vec_obs_raw[i]
            rewards[roll, agent] = reward[i]
        self.infos = []
        for i in range(self.num_rolls):
            info = []
            for j in range(self.num_agents):
                info.append({'individual_reward':0})
            self.infos.append(info)
        return (vis_obs, vec_obs)

    
    def step(self, actions):
        '''
            actions: np.array, [32,3,5] one-hot encoding
            Returns:
                obs: np.array, [32,3,18]
                rewards: np.array, [32,3,1]
                dones: np.array, np.bool, [32,3]
                infos: list, len is num_rolls, [[3 individual] 32 rolls]
        '''
        actions = np.array(actions)
        numNeedDecision = len(self.decisionStep)
        # assert numNeedDecision == self.num_agents * self.num_rolls
        
        action = np.zeros((min(self.num_agents * self.num_rolls,numNeedDecision),1))
        for i in self.decisionStep:
            a = np.zeros((1,1))
            # a[0] = np.argmax(actions[self.groupId[i]-1,i%self.num_agents])
            a[0] = actions[self.decisionStep[i].group_id-1, self.agentid2idx[i]]
            # a[0] = actions[i][0]
            a = ActionTuple(discrete=a)
            # self.env.set_actions(self.behavior_name, action)
            self.env.set_action_for_agent(self.behavior_name, i, a)
        # Perform a step in the simulation
        self.env.step()
        decisionStep, terminalStep = self.env.get_steps(self.behavior_name)
        self.decisionStep = decisionStep
        self.terminalStep = terminalStep
        obs_raw = decisionStep.obs[0] # 2 obs, 0 is grid sensor. (agents*platform, 20, 20, 7)
        groupReward = decisionStep.group_reward  #(agents*platform,)
        reward = decisionStep.reward #(agents*platform,)
        groupId = decisionStep.group_id #[agents*platform]

        if self.vis_obs_shape:
            vis_obs = np.zeros((self.num_rolls, self.num_agents)+self.vis_obs_shape)
        else:
            vis_obs = None
        if self.vec_obs_shape:
            vec_obs = np.zeros((self.num_rolls, self.num_agents)+(self.vec_obs_shape,))
        else:
            vec_obs = None
        rewards = np.zeros((self.num_rolls, self.num_agents, 1))
        dones = np.zeros((self.num_rolls, self.num_agents),dtype=np.bool)
        infos = []
        masks = []

        for agent_id in decisionStep:
            ds = decisionStep[agent_id]
            roll = ds.group_id - 1
            agent = self.agentid2idx[agent_id]
            # roll = agent_id
            # agent = 0
            if vis_obs is not None:
                for i in self.vis_idx:
                    vis_obs[roll, agent] = ds.obs[i]
            if vec_obs is not None:
                tmp = None
                for i in self.vec_idx:
                    if tmp is not None:
                        tmp = np.concatenate((tmp, ds.obs[i]), axis=0)
                    else:
                        tmp = ds.obs[i]
                vec_obs[roll, agent] = tmp
            rewards[roll, agent] = ds.reward + ds.group_reward
            dones[roll, agent] = False
            self.infos[roll][agent]['individual_reward'] = ds.reward
            masks.append((roll, agent))

        for agent_id in terminalStep:
            ts = terminalStep[agent_id]
            roll = ts.group_id - 1
            agent = self.agentid2idx[agent_id]
            # roll = agent_id
            # agent = 0
            # vis_obs[roll, agent] = ts.obs[0]
            # vec_obs[roll, agent] = ts.obs[1]
            rewards[roll, agent] = ts.reward + ts.group_reward
            dones[roll, agent] = True
            masks.append((roll, agent))


        # if len(terminalStep) != 0:
        #     return self.step(actions)
        # for i in range(obs_raw.shape[0]):
        #     roll = groupId[i]-1
        #     agent = i%self.num_agents
        #     obs[roll, agent] = obs_raw[i]
        #     rewards[roll, agent] = reward[i]
        #     dones[roll, agent] = True if i in terminalStep else False
        
        # dones = np.array([True if idx in terminalStep else False for idx in range(self.num_agents)])
        return (vis_obs, vec_obs), rewards, dones, self.infos, masks, decisionStep, terminalStep

    def close(self):
        self.env.close()