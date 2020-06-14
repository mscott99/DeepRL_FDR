from gym.spaces import Discrete, Box

from deep_rl import *
from deep_rl import run_steps
from deep_rl.optimizer import FDR_quencher
from math import sqrt


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
        return run_steps(self.agent)


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

def check_alternate_cFDR_critic_loss(critic_tolerance, critic_updating, info):
    """update actor and critic only upon convergence of the other"""

    if(info['total_steps'] - info['last_change'] > info['sceptic_period']):
        if critic_updating:
            critic_optimizer = info['critic_optimizer']
            if critic_optimizer.change_learner:
                critic_optimizer.change_learner = False
                critic_optimizer.reset_stats()
                return True
        else:
            critic_distance = (info['current_critic_val']-info['last_stable_critic'])/sqrt(info['critic_variance'])
            if critic_distance > critic_tolerance:
                info['actor_optimizer'].reset_stats()
                return True
    return False


def check_alternate_by_cFDR(critic_updating, info):
    """update actor and critic only upon convergence of the other"""
    if critic_updating:
        critic_optimizer = info['critic_optimizer']
        if critic_optimizer.change_learner:
            critic_optimizer.change_learner = False
            critic_optimizer.reset_stats()
            return True
    else:
        actor_optimizer = info['actor_optimizer']
        if (actor_optimizer.change_learner):
            actor_optimizer.change_learner = False
            actor_optimizer.reset_stats()
            return True
    return False


def check_alternate_partial_cFDR(critic_updating, info):
    """update actor and critic only upon convergence of the other"""
    critic_optimizer = info['critic_optimizer']
    if critic_updating and critic_optimizer.change_learner:
            critic_optimizer.change_learner = False
            critic_optimizer.reset_stats()
            return True
    elif info['total_steps'] - info['last_change'] > info['n_actor']:
        info['actor_optimizer'].reset_stats()
        return True
    else:
        return False


def check_alternate_stuck(reset_actor, reset_critic, critic_updating, info):
    return False


class Small_A2C_FDR(Model):
    """Parent model, make children for more specific implementation"""
    def __init__(self, **kwargs):
        generate_tag(kwargs)
        kwargs.setdefault('log_level', 0)
        config = Config()
        config.merge(kwargs)

        config.num_workers = 8
        config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        config.eval_env = Task(config.game) #This also sets characterisitics of the environment in the config dict
        net_class = None
        if type(config.action_space) == Discrete:
            net_class = CategoricalActorCriticNet
        elif type(config.action_space) == Box:
            net_class = GaussianActorCriticNet
        #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
        config.network_fn = lambda: net_class(
            config.state_dim,
            config.action_dim,
            phi_body=None,
            actor_body=config.actor_body,
            critic_body=config.critic_body
        )
        config.actor_body = FCBody(config.state_dim, hidden_units=config.actor_hidden_units, gate=torch.relu)
        config.critic_body = FCBody(config.state_dim, hidden_units=config.critic_hidden_units, gate=torch.relu)


        config.actor_optimizer_fn = lambda params, logger=None: FDR_quencher(params, lr_init=config.actor_lr, momentum=config.actor_mom, dampening=config.actor_damp,
                                                                  weight_decay=0.001, t_adaptive=config.t_adaptive, X=config.X, Y=config.Y,
                                                                  logger=logger, tag="actor",time_factor=config.rollout_length*config.num_workers,
                                                                  baseline_avg_length = config.baseline_avg_length, dFDR_avg_length= config.dFDR_avg_length)
        config.critic_optimizer_fn = lambda params, logger=None: FDR_quencher(params, lr_init=config.critic_lr, momentum=config.critic_mom, dampening=config.critic_damp,
                                                                   weight_decay=0.003, t_adaptive=config.t_adaptive, X=config.X, Y=config.Y,
                                                                   logger=logger, tag="critic", time_factor=config.rollout_length*config.num_workers,
                                                                   baseline_avg_length = config.baseline_avg_length, dFDR_avg_length= config.dFDR_avg_length)
        config.discount = 0.98
        config.actor_hidden_units = (32,32)
        config.critic_hidden_units = (32,32)
        config.use_gae = False
        # config.entropy_weight = 0.01
        config.alternate = True
        config.check_for_alternation_callback = check_alternate_by_cFDR
        #config.check_for_alternation = lambda *params: check_alternate_by_cFDR(0.2, 0.2, *params)
        config.rollout_length = 5
        #config.gradient_clip = 0.5
        #config.save_interval = 1e6
        #config.log_interval = 1e4
        config.max_steps = 2e5
        #config.log_keywords = [("action",0),("critic_loss",0),("actor_loss",0) ,("dFDR_critic",100),("cFDR_critic",100), ("dFDR_actor", 100), ("cFDR_actor", 100), ("OL_critic",100),("OR_critic",1),("Base_Theta_critic",0),("OL_actor",100),("OR_actor",1),("Base_Theta_actor",0)]
        config.log_keywords = [("action",0),("critic_loss",0),("actor_loss",0) ,("dFDR_critic",100), ("dFDR_actor", 100), ("OL_critic",100),("OR_critic",1),("OL_actor",100),("OR_actor",1)]
        config.tag = "mountain_car_debug"
        self.agent = FDRA2CAgent
        config.merge(kwargs)
        self.config = config

    def initialize(self):
        self.agent = self.agent(self.config)

class FDR_A2C_Entropy(Small_A2C_FDR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config= self.config
        config.entropy_weight = 0.1
        #Override when arguments are given from the running configuration
        config.merge(kwargs)

class FDR_A2C_RMS(Small_A2C_FDR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config= self.config
        config.alternate=False
        config.actor_optimizer_fn = lambda params: torch.optim.RMSprop(params, config.actor_lr),
        config.critic_optimizer_fn= lambda params: torch.optim.RMSprop(params, config.critic_lr),
        config.merge(kwargs)


class FDR_A2C_partial(Small_A2C_FDR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config= self.config
        config.alternate = True
        config.check_for_alternation_callback = check_alternate_partial_cFDR

class FDR_A2C_partial_LSTM(FDR_A2C_partial):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        config.actor_body = LinearLSTMBody(config.state_dim)
        config.critic_body = LinearLSTMBody(config.state_dim)
class FDR_A2C_critic_loss_change(Small_A2C_FDR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        config.check_for_alternation_callback = lambda *args: check_alternate_cFDR_critic_loss(config.X, config.critic_loss_tolerance, *args)
