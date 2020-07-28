from deep_rl import *
from deep_rl.optimizer import SimpleSGDLinearDecay, FDRCallback
from deep_rl.optimizer.FDR_ctrl_freak import FDRCtrlFreak

class ActorCtrlFDROptimizer:
    def __init__(self, actor_meta_params=None, critic_meta_params=None, actor_params=None,  critic_params=None):
        self.actor_optimizer = SimpleSGDLinearDecay(actor_params, **actor_meta_params)
        self.critic_optimizer = FDRCallback(critic_params, low_callback=self.actor_optimizer.augment_lr, high_callback=self.actor_optimizer.reduce_lr, **critic_meta_params)

    def step(self):
        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def zero_grad(self):
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()


class BothCtrlFDROptimizer(ActorCtrlFDROptimizer):
    def __init__(self, actor_meta_params=None, critic_meta_params=None, actor_params=None,  critic_params=None):
        self.actor_optimizer = SimpleSGDLinearDecay(actor_params, **actor_meta_params)
        self.critic_optimizer = FDRCtrlFreak(critic_params, low_callback=self.actor_optimizer.augment_lr, high_callback=self.actor_optimizer.reduce_lr, **critic_meta_params)
