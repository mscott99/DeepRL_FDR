from deep_rl import *
import numpy as np
import json
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
    'hidden_actor_layers': finite_dist([tuple(width for i in range(depth)) for width in [16,32,64] for depth in [2,3]]),
    'hidden_critic_layers':finite_dist([tuple(width for i in range(depth)) for width in [16,32,64] for depth in [2,3]]),
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
def estimator_fn(params, train_length, test_length):
    game = 'CartPole-v0'
    params['game'] = game
    params['max_steps'] = train_length
    model = Small_A2C_FDR_param_tuning(**params)
    # model.agent.load('data/final_model_300000_4')
    model.train()
    ret=0
    for i in range(test_length):
        ret += model.agent.eval_episode()
    ret = ret/test_length
    return ret, model

def sample_params(params_dist):
    result={}
    for key, val in params_dist.items():
        result[key] = val.sample()
    return result

def tune_params(estimator_fn, params_dist, num_tests, train_length, test_length, leaderboard_size=5, tune_version=0):
    leaderboard = [(0, None) for i in range(leaderboard_size)] #first is best
    model_version = 0
    for i in range(num_tests):
        params = sample_params(params_dist)
        score, model = estimator_fn(params, train_length, test_length)
        if score > leaderboard[leaderboard_size-1][0]:
            leaderboard[leaderboard_size-1]=(score, params, model_version)
            leaderboard = sorted(leaderboard, key=lambda obj:obj[0], reverse=True)
            model.agent.save('data/hyper_model_' + str(model_version) + "_tune_" + str(tune_version))
            model_version = model_version+1
            with open('data/hypResults_' + str(tune_version)+str(".json"), 'w+') as file:
                json.dump(leaderboard, file)



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
    tune_params(estimator_fn, search_config, num_tests=100, train_length=int(3e5), test_length=int(1e4), leaderboard_size=10, tune_version=7)

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
