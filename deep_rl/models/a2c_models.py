from deep_rl import *
from deep_rl import run_steps
from deep_rl.optimizer import FDR_quencher


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


class Model:
    def record(self, game):
        env = gym.make(game)

        try:
            env = gym.wrappers.Monitor(env, "recording", video_callable=lambda ep_id: True, force=True)
        except:
            raise Exception("Could not wrap env with monitor, make sure Xvfb is set up properly if in docker")

        states = env.reset()
        agent = self.agent
        for _ in range(1000):
            env.render()

            predictions = agent.network(agent.config.state_normalizer(states))
            states, reward, done, info = env.step(predictions['a'].item())
            if done:
                env.reset()

        env.env.close()

    def train(self):
        run_steps(self.agent)


class SmallA2CFeature(Model):
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
            actor_body=FCBody(4, hidden_units=(32, 32), gate=torch.tanh),
            critic_body=FCBody(4, hidden_units=(32, 32), gate=torch.tanh)
        )
        config.discount = 0.99
        config.use_gae = False
        #config.entropy_weight = 0.01
        config.rollout_length = 5
        config.gradient_clip = 0.5
        config.save_interval = 1e5
        config.log_interval = 1e5
        # config.max_steps = 10

        self.agent = FDRA2CAgent(config)


def check_alternate_interval_by_num_steps(interval_length, critic_updating: bool, info):
    """update actor once and then critique 'interval_length' times"""
    if not critic_updating:
        return True  # change back to critique always
    elif (info['total_steps'] - info['last_change'] > interval_length * info['num_workers']):
        return True
    else:
        return False


def check_alternate_by_cFDR(X_threshold_actor, X_threshold_critic, reset_actor, reset_critic, critic_updating, info):
    """update actor and critic only upon convergence of the other"""
    if critic_updating:
        if (info['dFDR_critic'].get() < X_threshold_critic):
            reset_critic()
            return True
    else:
        if (info['dFDR_actor'].get() < X_threshold_actor):
            reset_actor()
            return True
    return False

def check_alternate_stuck(reset_actor, reset_critic, critic_updating, info):
    return False

class Small_A2C_FDR(Model):
    def __init__(self, **kwargs):
        generate_tag(kwargs)
        kwargs.setdefault('log_level', 0)
        config = Config()
        config.merge(kwargs)

        config.num_workers = 8
        config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        config.eval_env = Task(config.game)
        #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
        config.network_fn = lambda logger=None: CategoricalDissociatedActorCriticNet(
            config.state_dim,
            config.action_dim,
            #actor_opt_fn=lambda params, logger=None: torch.optim.RMSprop(params, 0.005),
            actor_opt_fn=lambda params, logger=None: FDR_quencher(params, lr_init=0.0005, momentum=0, dampening=0,
                                                                  weight_decay=0.001, t_adaptive=30, X=0.01, Y=0.9,
                                                                  logger=logger, tag="actor"),

            #critic_opt_fn=lambda params, logger=None: torch.optim.RMSprop(params, 0.005),
            critic_opt_fn=lambda params, logger=None: FDR_quencher(params, lr_init=0.0005, momentum=0, dampening=0.0,
                                                                   weight_decay=0.003, t_adaptive=100, X=0.01, Y=0.9,
                                                                   logger=logger, tag="critic"),
            phi_body=None,
            actor_body=FCBody(config.state_dim, hidden_units=(32, 32), gate=torch.tanh),
            critic_body=FCBody(config.state_dim, hidden_units=(32, 32), gate=torch.tanh),
            logger=logger
        )
        config.discount = 0.98
        config.use_gae = False
        # config.entropy_weight = 0.01
        config.tag = "lr_001"
        config.alternate = True
        config.check_for_alternation = lambda *args: check_alternate_by_cFDR(1,1, *args)
        #config.check_for_alternation = lambda *params: check_alternate_by_cFDR(0.2, 0.2, *params)
        config.rollout_length = 5
        #config.gradient_clip = 0.5
        #config.save_interval = 1e6
        #config.log_interval = 1e4
        config.max_steps = 2e5
        config.log_keywords = [("critic_loss",0),("actor_loss",0) ,("dFDR_critic",100),("cFDR_critic",100), ("dFDR_actor", 100), ("cFDR_actor", 100), ("OL_critic",100),("OR_critic",1),("Base_Theta_critic",0),("OL_actor",100),("OR_actor",1),("Base_Theta_actor",0)]
        config.tag = "mountain_car_debug"
        self.agent = FDRA2CAgent(config)


class Small_A2C_FDR_param_tuning(Model):

    def __init__(self, lr_actor, lr_critic, hidden_actor_layers,
                 hidden_critic_layers, change_interval, **kwargs):
        generate_tag(kwargs)
        kwargs.setdefault('log_level', 0)
        config = Config()
        config.merge(kwargs)

        # config.num_workers = 3
        config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        config.eval_env = Task(config.game)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
        config.network_fn = lambda logger=None: CategoricalDissociatedActorCriticNet(
            config.state_dim,
            config.action_dim,
            # actor_opt_fn=lambda params,
            #    logger=None: torch.optim.RMSprop(params, lr_actor),
            actor_opt_fn=lambda params, logger=None: FDR_quencher(params, lr_init=lr_critic, momentum=0.9, dampening=0,
                                                                  weight_decay=0.001, t_adaptive=700, X=0.01, Y=0.9,
                                                                  logger=logger, tag="actor"),
            critic_opt_fn=lambda params,
                                 logger=None: FDR_quencher(params, lr_init=lr_critic,
                                                           momentum=0.9, dampening=0,
                                                           weight_decay=0.001, t_adaptive=700, X=0.01, Y=0.9,
                                                           logger=logger, tag="critic"),
            phi_body=None,
            actor_body=FCBody(4, hidden_units=hidden_actor_layers, gate=torch.tanh),
            critic_body=FCBody(4, hidden_units=hidden_critic_layers, gate=torch.tanh),
            logger=logger
        )
        # config.discount = 0.99
        config.use_gae = False
        # config.entropy_weight = 0.01
        config.rollout_length = 5
        # config.gradient_clip = 0.5
        config.log_interval = 1e4
        # config.max_steps = 5e5
        config.alternate = True
        config.check_for_alternation = lambda *params: check_alternate_by_cFDR(0.2,0.2, *params)
        self.agent = FDRA2CAgent(config)
