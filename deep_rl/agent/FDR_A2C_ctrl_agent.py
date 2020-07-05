#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .BaseAgent import *
from ..component import *
from ..network import *
from collections import deque
from math import sqrt

class FDRA2CCtrlAgent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        logger = self.logger
        self.task = config.task_fn()
        self.eval_task = config.eval_env
        self.network = config.network_fn()
        self.actor_optimizer = config.actor_optimizer_fn(self.network.actor_params, logger = logger)
        self.critic_optimizer = config.critic_optimizer_fn(self.network.critic_params, logger=logger)
        self.total_steps = 0
        self.time_factor = (config.num_workers*config.rollout_length)
        self.states = self.task.reset()
        self.init_logger(config.log_keywords)
        self.episode_count=0

        #specific to Cartpole, check for completion
        if config.game == "CartPole-v0" and config.stop_at_victory:
            self.last_100_returns = deque([0 for i in range(100)],maxlen=100)

        if config.track_critic_vals:
            self.critic_loss_baseline = RunningAvg(config.baseline_avg_length/self.time_factor)
            self.critic_loss_variance = RunningAvg(config.baseline_avg_length/self.time_factor)
            self.last_stable_critic_val = 0

        if self.config.alternate:
            #Alternation mechanism: boolean determines self.updating_critic says which one is updated,
            #and check_for_alternation checks if requisite conditions are in place to change training.
            self.last_change = 0
            self.updating_critic=True
            self.check_for_alternation = config.check_for_alternation_callback

    def init_logger(self, keywords):
        logger = self.logger
        for word, val in keywords:
            logger.track_scalar(word,False,val)

    def assign_stable_critic_loss(self):
        self.last_stable_critic_val = self.critic_loss_baseline.get()

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
        logger = self.logger
        episode_count = self.episode_count
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            episode_count += np.count_nonzero(terminals)
            #logger.update_log_value("action", float(to_np(prediction['a'])[0,0]))
            if config.game == "CartPole-v0" and config.stop_at_victory:
                self.record_online_return(info, returns_vessel=self.last_100_returns)
            else:
                self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers


        self.episode_count = episode_count
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

        if config.stop_at_victory and config.game == 'CartPole-v0' and np.mean(self.last_100_returns) >= 195:
            #solved
            return self.total_steps

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        if config.track_critic_vals:
            self.critic_loss_variance.add((policy_loss-self.critic_loss_baseline.get())**2)
            self.critic_loss_baseline.add(policy_loss)
        entropy_loss = entropy.mean()

        #log values that are logged on a per-step basis
        logger.update_log_value('critic_loss', value_loss.item())
        logger.update_log_value('actor_loss', policy_loss.item())
        logger.update_log_value("episode_count", episode_count)
        logger.write_all_tracked_scalars(step=self.total_steps)

        #(policy_loss - config.entropy_weight * entropy_loss +
        #config.value_loss_weight * value_loss).backward()
        #(policy_loss +
        # config.value_loss_weight * value_loss).backward()

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
        config.value_loss_weight * value_loss).backward()
        if config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return False #By default we are not done


