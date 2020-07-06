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
    def __init__(self, low, high, base):
        self.low_indep = np.log(low)/np.log(base)
        self.high_indep = np.log(high)/np.log(base)
        self.base = base

    def sample(self):
        elt = np.random.uniform(self.low_indep, self.high_indep)
        result = np.power(self.base, elt)
        return result

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
        if type(key) == tuple:
            value = val.sample()
            for elt in key:
                result[elt] =value
        else:
            result[key] = val.sample()
    return result


def purge_model_logging(dir_name, file_name):
    purge('data' + '/' + dir_name, file_name)
    purge('tf_log' + '/' + dir_name, file_name)
    purge('log', file_name)


def run_single(params, values_in_tag=[], model_index=None, save_all_params=False, num_evals=1000):

    if len(values_in_tag) > 0:
        params['tag']= params['tag'] + ('_').join([str(key) + '_' + str(params[key]) for key in values_in_tag])

    if model_index is not None:
        params['tag'] = params['tag'] + "_model_" + str(model_index)
        params['model_index'] = model_index

    save_folder = 'data/' + params['group_tag']+'/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    score, model = estimate_by_completion(params)
    #score, model = estimator_fn(params, num_evals)
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
                purge_model_logging(params['group_tag'], "model_" + str(model_to_delete[2]['model_index']))
            leaderboard[size-1] = (model.config.tag, score, params)
            leaderboard = sorted(leaderboard, key=lambda obj: obj[1], reverse=True)
            self.board = leaderboard
            model.agent.save(self.save_folder + model.config.tag)
            with open(self.save_folder + self.params_file_name, 'w+') as file:
                json.dump(leaderboard, file, default=lambda o:o.__name__)
        else:
            purge_model_logging(params['group_tag'], "model_" + str(params['model_index']))


def grid_tune_params(params_dist, const_params,num_evals, leaderboard_size=5, values_in_tag=[], follow_all_tuned=True):
    """params_dist values should be iterables"""
    if follow_all_tuned:
        values_in_tag = list(params_dist.keys())
    leaderboard = Leaderboard(leaderboard_size, save_folder='data/'+const_params['group_tag']+'/', params_file_name=const_params['group_tag']+ '.json')
    keys, values = zip(*params_dist.items())
    for i,bundle in enumerate(product(*values)):
        params = dict(zip(keys, bundle))
        params.update(const_params)
        try:
            score, model = run_single(params, num_evals, values_in_tag, model_index=i, save_all_params=False)
        except Exception:
            print("Failed trial by exception")
            print(Exception)
            continue
        leaderboard.add(score, params,model)


def randomised_tune_params(params_dist, const_params, num_tests , leaderboard_size=5, values_in_tag=[], follow_all_tuned=True, num_evals=1000):
    """params_dist elements should implement the sample() function"""
    if follow_all_tuned:
        values_in_tag = list(params_dist.keys())
    leaderboard = Leaderboard(leaderboard_size, save_folder='data/'+const_params['group_tag']+'/', params_file_name=const_params['group_tag']+ '.json')
    for i in range(num_tests):
        params = sample_params(params_dist)
        tuned_params = params.copy()
        params.update(const_params)
        if follow_all_tuned:
            values_in_tag = list(tuned_params.keys())
        #try:
        score, model = run_single(params,values_in_tag= values_in_tag, model_index=i, save_all_params=False, num_evals=num_evals)
        #except:
        #    print("Failed trial by exception")
        #    print(Exception)
        #    continue
        leaderboard.add(score, params,model)


def listed_tune_params(params_list, const_params,num_evals, leaderboard_size=5, values_in_tag=[], follow_all_tuned=True):
    if follow_all_tuned:
        values_in_tag = list(params_list[0].keys())
    leaderboard = Leaderboard(leaderboard_size, save_folder='data/' + const_params['group_tag'] + '/',
                              params_file_name=const_params['group_tag'] + '.json')
    for i,params in enumerate(params_list):
        params.update(const_params)
        #try:
        score, model = run_single(params, values_in_tag=values_in_tag, model_index=i, save_all_params=False, num_evals=num_evals)
        #except:
        #    print("Failed trial by exception")
        #    print(Exception)
        #    continue
        leaderboard.add(score, params, model)

def start_generic_run():
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    random_seed()
    set_one_thread()
    random_seed(seed=2)
    select_device(-1)
    #select_device(0)


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
                    'log_keywords': [('critic_loss', 0), ('episodic_return_train', 0),('updating_critic', 1),
                                     ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0), ('episode_count', 0),
                                     ('lr_critic', 0), ('lr_actor', 0)],
                    'dFDR_avg_length': 1e4,
                    'stop_at_victory': True
                    }

    randomised_tune_params(params_dist, const_params, leaderboard_size=7, num_tests=1000000,
                           values_in_tag=['n_actor', 'X', 'critic_lr', 'actor_lr'], follow_all_tuned=True, num_evals=40)

def run_single_cheetah():
    const_params = {'model_class': FDR_A2C_ctrl_actor,
                    'track_critic_vals': False,
                    'game': 'HalfCheetah-v2',
                    'max_steps': 5e6,
                    'num_workers':16,
                    'tag': 'debug',
                    'group_tag': 'debug',
                    'log_keywords': [('F_norm_critic',0),('critic_loss', 0), ('actor_loss', 0), ('episodic_return_train', 0),('updating_critic',1),
                                     ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0), ('episode_count', 0),
                                     ('lr_critic', 0), ('lr_actor', 0)],
                    'dFDR_avg_length': 3e4,
                    'actor_hidden_units':(200,200),
                    'critic_hidden_units':(200,200),
                    'stop_at_victory': True,
                    'critic_lr': 0.005,
                    'actor_lr': 0.005,
                    'X': 0.1,
                    'Y': 0.9,
                    'n_actor': 2e5,
                    'baseline_avg_length': 3e4,
                    'sceptic_period':1e3,
                    'entropy_weight':0.01,
                    'gradient_clip':50
    }
    run_single(const_params)



def tune_half_cheetah():
    dFDRs = [('dFDR_' + str(i) + '_critic',0) for i in range(10)]
    log_vals = [('critic_loss', 0), ('actor_loss', 0), ('episodic_return_train', 0),
                                     ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0), ('episode_count', 0),
                                     ('lr_critic', 0), ('lr_actor', 0),('F_norm_critic',0),('updating_critic',1)]
    log_vals = log_vals + dFDRs
    params_dist = {
        'critic_lr': finite_dist([0.01, 0.005, 0.001, 0.0005, 0.0001]),
        'actor_lr': finite_dist([0.01, 0.005, 0.001, 0.0005, 0.0001]),
        'X': finite_dist([0.5, 0.3, 0.1, 0.05]),
        'n_actor': finite_dist([1e3, 1e4, 1e5]),
        'baseline_avg_length': finite_dist([5e3, 1e4, 1e5]),
        'dFDR_avg_length': finite_dist([1e4, 5e4, 1e5]),
        'entropy_weight': finite_dist([0.01,0.03])
    }
    params = {      'model_class': FDR_A2C_partial,
                    'track_critic_vals': False,
                    'game': 'HalfCheetah-v2',
                    'max_steps': 1e6,
                    'tag': 'debug',
                    'group_tag': 'debug',
                    'log_keywords': log_vals,
                    'stop_at_victory': True,
                    'X': 0.3,
                    'Y': 0.5,
                    'n_actor': 1e4,
                    'entropy_weight':0.01,
                    'gradient_clip': 50.0
                    }
    randomised_tune_params(params_dist, params, 1000, leaderboard_size=5, follow_all_tuned=True, num_evals=40)

def run_ctrl_cheetah():
    dFDRs = [('dFDR_' + str(i) + '_critic', 0) for i in range(10)]
    log_vals = [('episodic_return_train', 0), ('dFDR_critic', 0), ('OL_critic', 0), ('OR_critic', 0),
                ('episode_count', 0), ('lr_critic', 0), ('lr_actor', 0),('critic_loss',0)]
    log_vals = log_vals + dFDRs
    const_params = {'model_class': FDR_A2C_ctrl,
                    'track_critic_vals': False,
                    'game': 'HalfCheetah-v2',
                    'max_steps': 5e6,
                    'num_workers': 16,
                    'tag': 'first_ctrl',
                    'group_tag': 'first_ctrl',
                    'log_keywords': log_vals,
                    'actor_hidden_units': (200, 200, 200),
                    'critic_hidden_units': (200, 200, 200),
                    'stop_at_victory': True,
                    'critic_lr': 0.5,
                    'actor_lr': 0.001,
                    'actor_mom':0.9,
                    'actor_damp':0.9,
                    'X_low': 0.1,
                    'X_high': 1.0,
                    'R': 0.9,
                    'sceptic_period': 1000,
                    'entropy_weight': 0.01,
                    'gradient_clip': 50,
                    'min_FDR_length':100,
                    'run_avg_base':1.5,
                    'critic_mom':0.8,
                    'low_count_threshold': 10,
                    'high_count_threshold':10
                    }
    run_single(const_params)

def tune_ctrl_cheetah():
    #dFDRs = [('dFDR_' + str(i) + '_critic', 0) for i in range(10)]
    log_vals = [('critic_loss', 0), ('actor_loss', 0), ('episodic_return_train', 0),
                ('dFDR_critic', 0),('episode_count', 0),
                ('lr_critic', 0)]
    #log_vals = log_vals
    params_dist = {
        #'actor_lr': finite_dist([0.01, 0.008,0.008, 0.0075, 0.0075, 0.0075,0.007, 0.007, 0.006, 0.005]),
        'actor_lr': finite_dist([0.0004, 0.0004, 0.00035, 0.0003, 0.0003, 0.00025, 0.0002, 0.0002, 0.00015, 0.0001]),
        'X_low': finite_dist([0.1, 0.1, 0.05,0.05, 0.05,0.05, 0.03, 0.03, 0.02]),
        'hidden_units' : finite_dist([tuple(width for i in range(depth)) for width in (200, 200, 150, 400, 400) for depth in (3,3,3,3,3,4)]),
        'critic_lr' : finite_dist([0.01,0.005,0.001, 0.0005, 0.0005, 0.0005, 0.0001]),
        'low_count_threshold': finite_dist([10, 10, 10, 10, 10, 10,10,10,10, 15, 15, 20, 20, 30]),
        'high_count_threshold': finite_dist([10, 10, 10, 10, 10, 10,10,10,10, 15, 15, 20, 20, 30]),
        'high_ratio': finite_dist([0.1,0.2, 0.5, 0.5,0.5,0.8,0.8,0.8, 0.9,0.9,0.9,0.9, 0.95]),
        'gradient_clip': finite_dist([5, 7,7,10,10,10,10,10,10,10,10,10,10,10,10,15,15,15,20,20,20, 30, 30]),
        'R':finite_dist([0.85, 0.9, 0.94, 0.94, 0.95, 0.95, 0.95, 0.97, 0.97,0.975, 0.98]),
        'min_baseline_length': finite_dist([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,100,150, 200]),
        'max_baseline_length': finite_dist([5e4,1e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 1e6]),
        'max_FDR_length':  finite_dist([5e4,1e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 1e6])
        #'actor_mom': finite_dist([0.8, 0.85, 0.9, 0.9, 0.95]),
        #'actor_damp': finite_dist([0.8, 0.85, 0.9, 0.9, 0.95]),
    }
    params = {'model_class': FDR_A2C_RMS_ctrl,
                    'track_critic_vals': False,
                    'game': 'HalfCheetah-v2',
                    'max_steps': 1e6,
                    'num_workers': 16,
                    'tag': 'tune_ctrl_rms_v2',
                    'group_tag': 'tune_ctrl_rms_v2',
                    'log_keywords': log_vals,
                    'min_baseline_length':100,
                    'min_FDR_length':100,
                    'max_FDR_length':5e5,
                    'sceptic_period': 1000,
                    'entropy_weight': 0.01,
                    'run_avg_base': 1.3,
                    'gate': torch.relu,
                    'X_high': 1.5,
              }
    randomised_tune_params(params_dist, params, 10000, leaderboard_size=7,follow_all_tuned=False, values_in_tag=['actor_lr', 'critic_lr', 'X_low','high_ratio', 'R', 'gradient_clip', 'min_baseline_length','low_count_threshold', 'high_count_threshold'], num_evals=100)

def eval_champ():
    const_params = {
         'critic_lr': 0.0005,
         'gradient_clip': 10,
         'entropy_weight': 0.01,
         'max_baseline_length': 500000.0,
         'critic_hidden_units': (400, 400, 400),
         'sceptic_period': 1000,
         'min_baseline_length': 100,
         'track_critic_vals': False,
         'max_steps': 1e6,
         'X_low': 0.05,
         'tag': 'champ_ctrl_',
         'R': 0.97,
         'low_count_threshold': 10,
         'run_avg_base': 1.3,
         'min_FDR_length': 100,
         'model_class': FDR_A2C_ctrl,
         'high_count_threshold': 10,
         'gate': torch.relu,
         'max_FDR_length': 500000.0,
         'actor_hidden_units': (400, 400, 400),
         'num_workers': 16,
         'actor_mom': 0.9,
         'high_ratio': 0.2,
         'actor_damp': 0.9,
         'X_high': 1.5,
         'log_keywords': [['critic_loss', 0], ['actor_loss',0], ['episodic_return_train', 0],
                                         ['dFDR_critic', 0], ['episode_count', 0], ['lr_critic', 0]],
         'group_tag': 'test_ctrl_champ_v1',
         'actor_lr': 0.006,
         'game': 'HalfCheetah-v2',
    }
    num_tests = 50
    params_list = [{} for i in range(num_tests)]
    listed_tune_params(params_list, const_params,1, leaderboard_size=num_tests, values_in_tag=[], follow_all_tuned=False)


def run_variable_actor():
    const_params = {
         'critic_lr': 0.001,
         'critic_mom':0.9,
         'critic_damp':0.9,
         'gradient_clip': 10,
         'entropy_weight': 0.01,
         'max_baseline_length': 500000.0,
         'critic_hidden_units': (64,64,64),
         'sceptic_period': 1e3,
         'min_baseline_length': 100,
         'track_critic_vals': False,
         'max_steps': 1e6,
         'X_low': 0.1,
         'tag': 'var_actor_',
         'R': 0.99,
         'low_count_threshold': 10,
         'run_avg_base': 1.3,
         'min_FDR_length': 100,
         'model_class': FDR_A2C_ctrl_actor,
         'high_count_threshold': 10,
         'gate': torch.relu,
         'max_FDR_length': 500000.0,
         'actor_hidden_units': (64,64,64),
         'num_workers': 16,
         'actor_mom': 0.9,
         'high_ratio': 0.2,
         'actor_damp': 0.9,
         'X_high': 2.5,
         'log_keywords': [['critic_loss', 0], ['actor_loss',0], ['episodic_return_train', 0],
                                         ['dFDR_critic', 0], ['episode_count', 0], ['lr_actor', 0], ['max_low_count',0],['max_high_count',0]],
         'group_tag': 'var_actor_debug',
         'actor_lr': 1e-4,
         'game': 'HalfCheetah-v2',
         'min_lr_actor': 1e-4,
         'max_lr_actor': 0.005
    }
    num_tests = 50
    params_list = [{} for i in range(num_tests)]
    run_single(const_params)


def run_variable_both():
    const_params = {
         'critic_lr': 25e-3,
         'critic_mom':0.5,
         'critic_damp':0.5,
         'gradient_clip': 10,
         'entropy_weight': 0.01,
         'max_baseline_length': 500000.0,
         'critic_hidden_units': (64,64,64),
         'sceptic_period': 1e3,
         'min_baseline_length': 100,
         'track_critic_vals': False,
         'max_steps': 1e6,
         'X_low': 0.1,
         'tag': 'var_both_',
         'R_actor': 0.98,
         'R_critic': 0.973,
         'low_count_threshold': 5,
         'run_avg_base': 1.3,
         'min_FDR_length': 100,
         'model_class': FDR_A2C_ctrl_both,
         'high_count_threshold': 10,
         'gate': torch.relu,
         'max_FDR_length': 500000.0,
         'actor_hidden_units': (64,64,64),
         'num_workers': 16,
         'actor_mom': 0.9,
         'high_ratio': 0.7,
         'actor_damp': 0.9,
         'X_high': 2.5,
         'log_keywords': [['critic_loss', 0], ['actor_loss',0], ['episodic_return_train', 0],
                                         ['dFDR_critic', 0], ['episode_count', 0], ['lr_critic',0], ['lr_actor', 0], ['max_low_count',0],['max_high_count',0]],
         'group_tag': 'var_ctrl_both_debug',
         'actor_lr': 1e-4,
         'game': 'HalfCheetah-v2',
         'min_lr_actor': 1e-4,
         'max_lr_actor': 1e-2,
         'min_lr_critic':1e-3,
         'max_lr_critic':5e-3
    }
    num_tests = 50
    run_single(const_params)


def tune_variable_both():
    const_params = {
        'critic_mom': 0.3,
        'critic_damp': 0.3,
        'gradient_clip': 10,
        'entropy_weight': 0.01,
        'max_baseline_length': 500000.0,
        'critic_hidden_units': (200, 200, 200),
        'sceptic_period': 1e3,
        'min_baseline_length': 100,
        'track_critic_vals': False,
        'max_steps': 5e5,
        'X_low': 0.1,
        'tag': 'ctrl_both_',
        'low_count_threshold': 5,
        'run_avg_base': 1.3,
        'min_FDR_length': 100,
        'model_class': FDR_A2C_ctrl_both,
        'high_count_threshold': 10,
        'gate': torch.relu,
        'max_FDR_length': 500000.0,
        'actor_hidden_units': (200, 200, 200),
        'num_workers': 16,
        'actor_mom': 0.9,
        'high_ratio': 0.5,
        'actor_damp': 0.9,
        'X_high': 2.5,
        'log_keywords': [['critic_loss', 0], ['actor_loss', 0], ['episodic_return_train', 0],
                         ['dFDR_critic', 0], ['episode_count', 0], ['lr_critic', 0], ['lr_actor', 0],
                         ['max_low_count', 0], ['max_high_count', 0]],
        'group_tag': 'tune_ctrl_both_v2',
        'game': 'HalfCheetah-v2',
    }
    params_dist= {
        ('min_lr_actor', 'actor_lr') : finite_dist([1e-4,2e-4, 3e-4, 4e-4,8e-4,1e-3]),
        ('max_lr_critic', 'critic_lr'): finite_dist([9e-3, 1e-2,2e-2, 2e-2]),
        'max_lr_actor': finite_dist([1e-3,2e-3,4e-3, 8e-3]),
        'min_lr_critic': finite_dist([5e-4,1e-3,1e-3,1.5e-3,1.5e-3,2e-3,2e-3,4e-3, 4e-3,8e-3]),
        'R_actor': finite_dist([0.9,0.95,0.97,0.975,0.98,0.983,0.987,0.99]),
        'R_critic': finite_dist([0.9,0.95,0.97,0.975,0.98,0.983,0.987,0.99]),
    }
    randomised_tune_params(params_dist,const_params=const_params,num_tests=1000, leaderboard_size=7,num_evals=32)


def tune_variable_RMS():
    const_params = {
        'model_class':FDR_A2C_RMS,
        'track_critic_vals':True,
        'stop_at_victory':True,
        'actor_hidden_units':(24,24),
        'critic_hidden_units':(24,24),
        'gate':torch.tanh,
        'group_tag':'normalised_tune_rms',
        'tag':'tune_rms_',
        'max_steps': 3e5,
        'game': 'CartPole-v0',
    }
    params_dist={
        'actor_lr': log_uniform_dist(1e-4, 1e-2, 10),
        'critic_lr': log_uniform_dist(1e-4, 1e-2, 10)
    }
    randomised_tune_params(const_params=const_params, params_dist=params_dist, num_tests=40, leaderboard_size=5)

def tune_variable_FDR():
    const_params = {
        'game':'CartPole-v0',
        'track_critic_vals':True,
        'stop_at_victory':True,
        'model_class': New_FDR_A2C_ctrl,
        'actor_hidden_units': (24, 24),
        'critic_hidden_units': (24, 24),
        'gate': torch.tanh,
        'group_tag': 'normalised_tune_FDR',
        'tag': 'tune_FDR_',
        'max_steps': 3e5,
        'actor_mom': 0.9,
        'actor_damp':0.9,
        'X_low':0.1,
        'X_high':1.0,
        'sceptic_period':1000,
        'min_baseline_length':100,
        'max_baseline_length':1e5,
        'min_FDR_length':100,
        'max_FDR_length':1e5,
        'low_count_threshold':10,
        'high_count_threshold':10,
        'high_ratio':0.5,
    }
    params_dist = {
        'actor_lr': log_uniform_dist(1e-4, 1e-2, 10),
        'critic_lr': log_uniform_dist(1e-4, 1e-2, 10),
        'R': finite_dist([0.95,0.975,0.98,0.985,0.99]),
    }

    randomised_tune_params(const_params=const_params, params_dist=params_dist, num_tests=40, leaderboard_size=5)

if __name__ == "__main__":
    start_generic_run()
    tune_variable_RMS()
    tune_variable_FDR()
    #run_half_cheetah()
    #tune_n_steps()
    #run_single_cheetah()
    #tune_half_cheetah()
    #run_ctrl_cheetah()
    #tune_ctrl_cheetah()
    #eval_champ()
    #tune_variable_both()

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
