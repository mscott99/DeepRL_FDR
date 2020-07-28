#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *
import argparse
import torch


class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.log_level = 0
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10
        self.async_actor = True
        self.tasks = False
        self.alternate = False
        self.check_for_alternation_callback = lambda:None
        self.log_keywords = []
        self.actor_hidden_units = (16,16)
        self.critic_hidden_units = (16,16)
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.critic_mom = 0
        self.actor_mom = 0
        self.actor_damp = 0
        self.critic_damp = 0
        self.X = 0.01
        self.Y = 0.9
        self.t_adaptive = 100
        self.baseline_avg_length = 1000.0 #unit: number of states. This is the number of states until the weight is 1/e
        self.dFDR_avg_length = 1000.0
        self.critic_loss_tolerance = 3.0
        self.track_critic_vals=False
        self.skeptic_period = 0 #minimal period for average to run before taking averages into account
        self.n_actor = 0
        self.stop_at_victory=False
        self.group_tag = "default_group"
        self.actor_body = None
        self.critic_body = None
        self.R = 0.97
        self.R_critic = 0.97
        self.R_actor = 0.97
        self.X_low = 0.1
        self.X_high = 1.0
        self.min_baseline_length = 100.0
        self.max_baseline_length = 1e6
        self.min_FDR_length = 100.0
        self.max_FDR_length = 1e6
        self.run_avg_base = 1.2
        self.low_count_threshold = 1.0
        self.high_count_threshold = 1.0
        self.gate = torch.tanh
        self.logger =None
        self.min_lr_actor = 1e-6
        self.max_lr_actor = 1.0
        self.min_lr_critic = 1e-6
        self.max_lr_critic = 1.0
        self.lin_decay_a = 1 #inversely proportionnal learning rate decay
        self.high_ratio = 0.5 #portion of running averages that must exceed the ratio for a change in lr
        #dont believe average before  skeptic_period has expired
    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name
        self.action_space = env.action_space

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
