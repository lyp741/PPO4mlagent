import torch
import os
import numpy as np


class Arguments:
    def __init__(self, agent=None, env=None):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 10  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 0.00003  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.net_dim = 2 ** 9  # the network width
        self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 3  # collect target_step, then update network
        self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
        self.max_memo = self.target_step  # capacity of replay buffer
        self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times = 2  # number of times that get episode return in first
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        # np.random.seed(self.random_seed)
        # torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)
