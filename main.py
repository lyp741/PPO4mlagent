import os
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy
from args import Arguments
from ppo import AgentPPO
from mla_wrapper_ma_cnn import MLA_Wrapper

def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=True)

    '''init: Agent'''
    env = args.env
    agent = args.agent
    agent.init(args.net_dim, env.vis_obs_shape, env.vec_obs_shape, env.action_space[0].n, env,
               args.learning_rate, args.if_per_or_gae)

    '''init ReplayBuffer'''
    # def update_buffer(_trajectory):
    #     _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
    #     ten_state = torch.as_tensor(_trajectory[0])
    #     ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
    #     ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
    #     ten_action = torch.as_tensor(_trajectory[3])
    #     ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

    #     buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

    #     _steps = ten_reward.shape[0]
    #     _r_exp = ten_reward.mean()
    #     return _steps, _r_exp

    '''start training'''
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    agent.state = env.reset()
    if_train = False

    try:
        #agent.load_model('model.pkl')
        print('loaded model')
    except:
        print('no model')
    if_train = True
    while True:
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step)
            # steps, r_exp = update_buffer(trajectory_list)

        if if_train:
            logging_tuple = agent.update_net(batch_size, repeat_times, soft_update_tau)
            agent.save_model('model.pkl')

def main():
    args = Arguments()  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.visible_gpu = '0'

    "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
    args.env = MLA_Wrapper()
    args.reward_scale = 2 ** -1

    args.target_step = 128

    train_and_evaluate(args)

if __name__ == '__main__':
    main()