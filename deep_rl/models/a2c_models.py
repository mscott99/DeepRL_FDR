from deep_rl import *
import torch


def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.max_steps = 1e4
    config.save_interval = 1e3
    run_steps(A2CAgent(config))


class SmallA2CFeature:
    def __init__(self, **kwargs):
        generate_tag(kwargs)
        kwargs.setdefault('log_level', 0)
        config = Config()
        config.merge(kwargs)

        config.num_workers = 3
        config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        config.eval_env = Task(config.game)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
        config.network_fn = lambda: CategoricalDissociatedActorCriticNet(
            config.state_dim,
            config.action_dim,
            actor_opt_fn=lambda params: torch.optim.RMSprop(params, 0.001),
            critic_opt_fn=lambda params: torch.optim.RMSprop(params, 0.0005),
            phi_body=None,
            actor_body=FCBody(4, hidden_units=(32,32), gate=torch.tanh),
            critic_body=FCBody(4, hidden_units=(32,32), gate=torch.tanh)
        )
        config.discount = 0.99
        config.use_gae = False
        # config.entropy_weight = 0.01
        config.rollout_length = 5
        config.gradient_clip = 0.5
        config.save_interval = 1e5
        config.log_interval = 1e5
        config.max_steps = 10
        self.agent = FDRA2CAgent(config)

    def train(self):
        run_steps(self.agent)




class SmallA2CFeature001:
    def __init__(self, **kwargs):
        generate_tag(kwargs)
        kwargs.setdefault('log_level', 0)
        config = Config()
        config.merge(kwargs)

        config.num_workers = 3
        config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        config.eval_env = Task(config.game)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
        config.network_fn = lambda: CategoricalDissociatedActorCriticNet(
            config.state_dim,
            config.action_dim,
            actor_opt_fn=lambda params: torch.optim.RMSprop(params, 0.0001),
            critic_opt_fn=lambda params: torch.optim.RMSprop(params, 0.0001),
            phi_body=None,
            actor_body=FCBody(4, hidden_units=(8,8), gate=F.relu),
            critic_body=FCBody(4, hidden_units=(8,8), gate=F.relu)
        )
        config.discount = 0.99
        config.use_gae = False
        # config.entropy_weight = 0.01
        config.rollout_length = 5
        config.gradient_clip = 0.5
        config.save_interval = 1e5
        config.log_interval = 1e4
        config.max_steps = 1e6
        self.agent = FDRA2CAgent(config)

    def train(self):
        run_steps(self.agent)