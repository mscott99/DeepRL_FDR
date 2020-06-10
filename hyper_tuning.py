from deep_rl import *
import numpy as np
from itertools import product
import json
import os, re
import bisect
from sklearn.model_selection import RandomizedSearchCV

class finite_dist:
    def __init__(self, a):
        self.a = a
    def sample(self):
        result = np.random.choice(self.a)
        if isinstance(result, np.integer):
            return result.item()
        else: return result

class deterministic_dist:
    def __init__(self, a):
        self.a = a
    def sample(self):
        return self.a

class uniform_dist:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)

class log_uniform_dist:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return (self.high - self.low)*((np.exp(np.random.sample())-1)/(np.exp(1)-1)) + self.low

search_config = {
    'lr_actor':uniform_dist(1e-4, 1e-3),
     'lr_critic':uniform_dist(1e-4, 1e-3),
    #'hidden_actor_layers': finite_dist([tuple(width for i in range(depth)) for width in [24] for depth in [2]]),
    'hidden_actor_layers': deterministic_dist((24,24)),
    'hidden_critic_layers':deterministic_dist((24,24)),
    #'hidden_critic_layers':finite_dist([tuple(width for i in range(depth)) for width in [24] for depth in [2]]),
    'entropy_weight': uniform_dist(0,0.3),
    'gradient_clip':uniform_dist(0.3,0.7),
    'discount':uniform_dist(0.95,0.99),
    'change_interval':finite_dist([50,100,200,400,800,1600])
}

params_default = {'lr_actor':0.001,
                  'lr_critic':0.001,
                  'hidden_actor_layers':(32,32),
                  'hidden_critic_layers':(32,32),
                  'entropy_weight':0,
                  'gradient_clip':0.5,
                  'discount':0.99}

def estimator_fn(model_class, params, num_evals):
    model = model_class(**params)
    model.initialize()
    # model.agent.load('data/final_model_300000_4')
    model.train()
    ret=0
    for i in range(num_evals):
        ret += model.agent.eval_episode()
    ret = ret / num_evals
    return ret, model

def sample_params(params_dist):
    result={}
    for key, val in params_dist.items():
        result[key] = val.sample()
    return result

#def try_all_params_iter(params_dist):

def purge_model_logging(model_name):
    purge('data', "^" + model_name)
    purge('tf_log', "^" + "logger-" + model_name)
    purge('log', "^" + model_name)


def grid_tune_params(model_class, params_dist, const_params,num_evals, leaderboard_size=5, tune_tag="default_hyp_tune_tag", data_folder="hyp_results/", values_in_tag=[]):
    """params_dist values should be iterables"""
    leaderboard = [("", -10000, None) for i in range(leaderboard_size)] #first is best
    keys, values = zip(*params_dist.items())
    if not os.path.exists('data/'+data_folder):
        os.makedirs('data/'+data_folder)
    for i,bundle in enumerate(product(*values)):
        params = dict(zip(keys, bundle))
        params.update(const_params)
        params_string = ('_').join([str(key) + '_' + str(params[key]) for key in values_in_tag])
        params['tag'] = tune_tag + "_"+ params_string + "_model_" + str(i)
        score, model = estimator_fn(model_class, params,num_evals)
        if score > leaderboard[leaderboard_size - 1][1]:
            model_to_delete = leaderboard[leaderboard_size - 1]
            if model_to_delete[2] is not None:
                purge_model_logging(model_to_delete[0])
            leaderboard[leaderboard_size - 1] = (model.agent.config.tag, score, params)
            leaderboard = sorted(leaderboard, key=lambda obj: obj[0], reverse=True)
            model.agent.save("data/" + data_folder + model.agent.config.tag)
            with open('data/'+ data_folder + tune_tag + str(".json"), 'w+') as file:
                json.dump(leaderboard, file)
        else:
            purge_model_logging(model.agent.config.tag)


def randomised_tune_params(model_class, params_dist, const_params,num_tests , num_evals, leaderboard_size=5,
                     tune_tag="default_hyp_tune_tag", data_folder="hyp_results/", values_in_tag=[]):
    """params_dist values should be iterables"""
    leaderboard = [("", 0, None) for i in range(leaderboard_size)]  # first is best
    if not os.path.exists('data/' + data_folder):
        os.makedirs('data/' + data_folder)
    for i in range(num_tests):
        params = sample_params(params_dist)
        params.update(const_params)
        params_string = ('_').join([str(key) + '_' + str(params[key]) for key in values_in_tag])
        if tune_tag is not None:
            params['tag'] = tune_tag + "_" + params_string + "_model_" + str(i)
        score, model = estimator_fn(model_class, params,num_evals)
        if score > leaderboard[leaderboard_size - 1][1]:
            model_to_delete = leaderboard[leaderboard_size - 1]
            if model_to_delete[2] is not None:
                purge_model_logging(model_to_delete[0])
            leaderboard[leaderboard_size - 1] = (model.agent.config.tag, score, params)
            leaderboard = sorted(leaderboard, key=lambda obj: obj[0], reverse=True)
            model.agent.save("data/" + data_folder + model.agent.config.tag)
            with open('data/' + data_folder + tune_tag + str(".json"), 'w+') as file:
                json.dump(leaderboard, file)
        else:
            purge_model_logging(model.agent.config.tag)


def start_generic_run():
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    set_one_thread()
    random_seed()
    select_device(-1)
    set_one_thread()
    random_seed()
    select_device(-1)
    # select_device(0)

if __name__ == "__main__":
    start_generic_run()
    grid_search_entropy = {'entropy_weight':[0,0.1]}
    const_params={'game':'Acrobot-v1', 'max_steps': 5e5, 'tag':'tune_debug_pendulum',
                  'log_keywords':[('critic_loss',0),("episodic_return_train",0),('action',0),('dFDR_critic',0),('dFDR_actor',0),('Base_Theta_critic',0),('OL_critic',0),('OR_critic',0)],
                  'X':0.5, 'baseline_avg_length':1e4, 'dFDR_avg_length':1e4}

    grid_tune_params(Small_A2C_FDR, grid_search_entropy, const_params, num_evals=10, leaderboard_size=10, tune_tag="try_acrobot", data_folder="acrobot_first/", values_in_tag=['max_steps'])
    #randomised_tune_params(estimator_fn, search_config, num_tests=20, train_length=int(100), test_length=int(5), leaderboard_size=3, tune_version=0)

    #game = 'CartPole-v0'
    # dqn_feature(game=game)
    # quantile_regression_dqn_feain between posistions pyhtonture(game=game)
    # categorical_dqn_feature(game=game)
    #model = SmallA2CFeature(game=game)
    #model = Small_A2C_FDR(game=game)
    #model.agent.load('data/final_model_300000_4')
    #model.train()
    #record(model, game)
    #model.agent.save('data/final_model_%s_%s'%(model.agent.total_steps,4))
