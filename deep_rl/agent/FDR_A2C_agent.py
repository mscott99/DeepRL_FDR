#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .BaseAgent import *
from ..component import *
from ..network import *


class FDRA2CAgent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.eval_task = config.eval_env
        self.network = config.network_fn(logger=self.logger)
        self.actor_optimizer = self.network.actor_opt
        self.critic_optimizer = self.network.critic_opt
        self.total_steps = 0
        self.states = self.task.reset()
        self.init_logger(config.log_keywords)
        if self.config.alternate:
            #Alternation mechanism: boolean determines self.updating_critic says which one is updated,
            #and check_for_alternation checks if requisite conditions are in place to change training.
            self.last_change = 0
            self.updating_critic=True
            default_fn = lambda : None
            actor_reset_fn = lambda:None
            critic_reset_fn = lambda:None
            if(hasattr(self.actor_optimizer, "reset_stats")):
                actor_reset_fn = self.actor_optimizer.reset_stats
            if(hasattr(self.critic_optimizer, "reset_stats")):
                critic_reset_fn = self.critic_optimizer.reset_stats
            self.check_for_alternation = lambda critic_updating, info: self.config.check_for_alternation(
                actor_reset_fn, critic_reset_fn, critic_updating, info)

    def init_logger(self, keywords):
        logger = self.logger
        for word, val in keywords:
            logger.track_scalar(word,False,val)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        action = to_np(prediction['a'])
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(config.state_normalizer(states))
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        #entropy_loss = entropy.mean()
        logger = self.logger
        #logger.update_log_value('actor_loss', policy_loss.item())
        logger.update_log_value('critic_loss', value_loss.item())
        logger.update_log_value('actor_loss', policy_loss.item())

        #(policy_loss - config.entropy_weight * entropy_loss +
        #config.value_loss_weight * value_loss).backward()
        #(policy_loss +
        # config.value_loss_weight * value_loss).backward()


        if self.config.alternate:
            updating_critique = self.updating_critic
            alternate_info = {"total_steps": self.total_steps, "last_change": self.last_change, "num_workers": self.config.num_workers}
            alternate_info.update(logger.tracked_scalars)
            if(self.check_for_alternation(updating_critique, alternate_info)):
                self.updating_critic = not updating_critique
                self.last_change = self.total_steps

            if self.updating_critic:
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                if config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.critic_optimizer.step()
            else:
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                if config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.actor_optimizer.step()

        else:
            self.actor_optimizer.step()
            self.critic_optimizer.step()


