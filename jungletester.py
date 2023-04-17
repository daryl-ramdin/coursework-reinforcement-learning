import random
from jungle import Jungle
import gymnasium as gym
from gymnasium import spaces
from jungleenv import JungleEnv

import numpy
import numpy as np

'''
env = JungleEnv({"size":7})
env.reset([3,5])
env.render()
print("Move north")
print(env.step("North"))
env.render()

print("Move east")
print(env.step("East"))
env.render()
print("Move east")
print(env.step("East"))
env.render()
print("Move east")
print(env.step("East"))
env.render()
'''

import ray
from ray.rllib.algorithms import ppo, dqn
from ray.tune.logger import pretty_print

ray.shutdown()
ray.init()

#ref: INM707 Lab 6
algo = ppo.PPO(env=JungleEnv, config={"env_config": {"size":15},})

while True:
    metrics = algo.train()
    print("metrics",metrics)
    mean_reward = metrics['episode_reward_mean']
    print("mean reward",mean_reward)