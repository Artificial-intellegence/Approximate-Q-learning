"""
Use this program to try out different environments in Open AI Gym.
"""

from time import sleep
import gym

# These are other environments you can try
# env = gym.make("LunarLander-v2")
# env = gym.make("BipedalWalker-v3")

env = gym.make("CartPole-v1")
print("observation space:", env.observation_space.shape)
try:
    print("action space:", env.action_space.n)
except:
    pass

for i in range(2):
    print("-"*70)
    print("Episode", i)
    env.reset()
    total_reward = 0
    for j in range(200):
        env.render()
        #sleep(0.05) #uncomment to slow down animation
        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        #print(observation) #uncomment to see state information
        print("action", action, "reward:", reward)
        if done:
            break
    print("Episode ended after", j, "steps, total reward:", total_reward )

env.close()

