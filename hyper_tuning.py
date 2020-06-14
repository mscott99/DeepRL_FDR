from deep_rl import *
import numpy as np
from itertools import product
import json
import os, re
import bisect
#from sklearn.model_selection import RandomizedSearchCV

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

def estimator_fn(params, num_evals):
    model = params['model_class'](**params)
    model.initialize()
    # model.agent.load('data/final_model_300000_4')
    model.train()
    ret=0
    for i in range(num_evals):
        ret += model.agent.eval_episode()
    ret = ret / num_evals
    return ret, model

def estimate_by_completion(params):
    model = params['model_class'](**params)
    model.initialize()
    # model.agent.load('data/final_model_300000_4')
    score = - model.train()
    return score, model

def sample_params(params_dist):
    result={}
    for key, val in params_dist.items():
        result[key] = val.sample()
    return result


def purge_model_logging(dir_name, file_name):
    purge('data' + '/' + dir_name, "^" + file_name)
    purge('tf_log' + '/' + dir_name, "^" + "logger-" + file_name)
    purge('log', "^" + file_name)


def run_single(params, values_in_tag=[], model_index=None, save_all_params=False):

    if len(values_in_tag) > 0:
        params['tag']= params['tag'] + ('_').join([str(key) + '_' + str(params[key]) for key in values_in_tag])

    if model_index is not None:
        params['tag'] = params['tag'] + "_model_" + str(model_index)

    save_folder = 'data/' + params['group_tag']+'/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    score, model = estimate_by_completion(params)
    model.agent.save(save_folder + params['tag'])

    if save_all_params:
        params['model_class'] = str(params['model_class'])
        with open(save_folder + params['tag']+ str(".json"), 'w+') as file:
            json.dump([params['tag'], score, params], file)

    return score, model





class Leaderboard:
    def __init__(self, size, save_folder='data/', params_file_name='parameters.json'):
        self.board = [("", -1e9, None) for i in range(size)]  # first is best
        self.size = size
        self.save_folder = save_folder
        self.params_file_name = params_file_name

    def add(self, score, params, model):
        leaderboard = self.board
        size = self.size
        if score >= leaderboard[size-1][1]:
            model_to_delete = leaderboard[size-1]
            if model_to_delete[2] is not None:
                purge_model_logging(model_to_delete[2]['group_tag'], model_to_delete[0])
            leaderboard[size-1] = (model.agent.config.tag, score, params)
            leaderboard = sorted(leaderboard, key=lambda obj: obj[1], reverse=True)
            self.board = leaderboard
            model.agent.save(self.save_folder + model.agent.config.tag)
            with open(self.save_folder + self.params_file_name, 'w+') as file:
                json.dump(leaderboard, file, default=lambda o:o.__name__)
        else:
            purge_model_logging(model.config.group_tag, model.config.tag)


def grid_tune_params(params_dist, const_params,num_evals, leaderboard_size=5, values_in_tag=[], follow_all_tuned=True):
    """params_dist values should be iterables"""
    if follow_all_tuned:
        values_in_tag = list(params_dist.keys())
    leaderboard = Leaderboard(leaderboard_size, save_folder='data/'+const_params['group_tag']+'/', params_file_name=const_params['group_tag']+ '.json')
    keys, values = zip(*params_dist.items())
    for i,bundle in enumerate(product(*values)):
        params = dict(zip(keys, bundle))
        params.update(const_params)
        score, model = run_single(params, num_evals, values_in_tag, model_index=i, save_all_params=False)
        leaderboard.add(score, params,model)


def randomised_tune_params(params_dist, const_params, num_tests , leaderboard_size=5, values_in_tag=[], follow_all_tuned=True):
    """params_dist elements should implement the sample() function"""
    if follow_all_tuned:
        values_in_tag = list(params_dist.keys())
    leaderboard = Leaderboard(leaderboard_size, save_folder='data/'+const_params['group_tag']+'/', params_file_name=const_params['group_tag']+ '.json')
    for i in range(num_tests):
        params = sample_params(params_dist)
        params.update(const_params)
        score, model = run_single(params,values_in_tag=values_in_tag, model_index=i, save_all_params=False)
        leaderboard.add(score, params,model)


def listed_tune_params(params_list, const_params,num_evals, leaderboard_size=5, values_in_tag=[], follow_all_tuned=True):
    if follow_all_tuned:
        values_in_tag = list(params_list[0].keys())
    leaderboard = Leaderboard(leaderboard_size, save_folder='data/' + const_params['group_tag'] + '/',
                              params_file_name=const_params['group_tag'] + '.json')
    for i,params in enumerate(params_list):
        params.update(const_params)
        score, model = run_single(params, num_evals, values_in_tag, model_index=i, save_all_params=False)
        leaderboard.add(score, params, model)

def start_generic_run():
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    set_one_thread()
    random_seed()
    select_device(-1)
    set_one_thread()
    random_seed(seed=3)
    select_device(-1)
    # select_device(0)


def tune_n_steps():
    params_dist = {
        'critic_lr': finite_dist([0.3, 0.1, 0.05, 0.01]),
        'actor_lr': finite_dist([0.3, 0.1, 0.05]),
        'X': finite_dist([2, 1, 0.5, 0.3]),
        'Y': finite_dist([0.2, 0.5, 0.9]),
        'n_actor': finite_dist([1e2, 1e3, 1e4, 1e5]),
        'baseline_avg_length': finite_dist([5e3, 1e4, 1e5]),
    }

    const_params = {'model_class': FDR_A2C_partial,
                    'track_critic_vals': False,
                    'game': 'CartPole-v0',
                    'max_steps': 1e5,
                    'tag': 'debug',
                    'group_tag': 'debug',
                    'log_keywords': [('critic_loss', 0), ('actor_loss', 0), ('episodic_return_train', 0),
                                     ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0), ('episode_count', 0),
                                     ('lr_critic', 0), ('lr_actor', 0)],
                    'dFDR_avg_length': 1e4,
                    'stop_at_victory': True
                    }

    randomised_tune_params(params_dist, const_params, leaderboard_size=7, num_tests=1000000,
                           values_in_tag=['n_actor', 'X', 'critic_lr', 'actor_lr'], follow_all_tuned=True)

def run_single_cartpole():
    const_params = {'model_class': FDR_A2C_partial,
                    'track_critic_vals': False,
                    'game': 'HalfCheetah-v2',
                    'max_steps': 1e5,
                    'tag': 'debug',
                    'group_tag': 'debug',
                    'log_keywords': [('critic_loss', 0), ('actor_loss', 0), ('episodic_return_train', 0),
                                     ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0), ('episode_count', 0),
                                     ('lr_critic', 0), ('lr_actor', 0)],
                    'dFDR_avg_length': 1e4,
                    'actor_hidden_units':(100,100),
                    'critic_hidden_units':(100,100),
                    'stop_at_victory': True,
                    'critic_lr': 0.01,
                    'actor_lr': 0.05,
                    'X': 0.3,
                    'Y': 0.9,
                    'n_actor': 1e4,
                    'baseline_avg_length': 1e5,
    }
    run_single(const_params)


def run_half_cheetah():
    params = {      'model_class': FDR_A2C_partial,
                    'track_critic_vals': False,
                    'game': 'HalfCheetah-v2',
                    'max_steps': 1e3,
                    'tag': 'debug',
                    'group_tag': 'debug',
                    'log_keywords': [('critic_loss', 0), ('actor_loss', 0), ('episodic_return_train', 0),
                                     ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0), ('episode_count', 0),
                                     ('lr_critic', 0), ('lr_actor', 0)],
                    'baseline_avg_length': 1e4,
                    'dFDR_avg_length': 1e4,
                    'stop_at_victory': True,
                    'critic_lr': 0.05,
                    'actor_lr':  0.05,
                    'X': 0.3,
                    'Y': 0.5,
                    'n_actor': 1e3,
                    'entropy_weight':0.01
                    }
    run_single(params)

if __name__ == "__main__":
    start_generic_run()
    #run_half_cheetah()
    #tune_n_steps()
    run_single_cartpole()

    #grid_tune_params(Small_A2C_FDR, grid_search_entropy, const_params, num_evals=10, leaderboard_size=10, tune_tag="acrobat_long_run", data_folder="acrobot_long/", values_in_tag=['max_steps'])
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
