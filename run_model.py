from deep_rl import *

if __name__ == "__main__":

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

    game = 'CartPole-v0'
    # dqn_feature(game=game)
    # quantile_regression_dqn_feature(game=game)
    # categorical_dqn_feature(game=game)
    #model = SmallA2CFeature(game=game)
    model = Small_A2C_FDR(game=game)
    #model.agent.load('data/final_model_300000_4')
    model.train()
    #record(model, game)
    model.agent.save('data/final_model_%s_%s'%(model.agent.total_steps,4))


