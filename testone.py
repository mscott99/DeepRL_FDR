import gym
env = gym.make('CartPole-v0')
#env = gym.wrappers.Monitor(env, "recording", video_callable=lambda ep_id:True, force=True)
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()

