import ray
import gymnasium
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray import air, tune



# # #ray.shutdown()
# # #ray.init()

# #Let's configure our algorithm

environ = "Riverraid-ramDeterministic-v4"
episodes = 500

config = DQNConfig()
config = config.environment(environ)
config.replay_buffer_config["capacity"] = 5000
config.double_q = True
if torch.cuda.is_available(): config = config.resources(num_gpus = 2)
config.replay_buffer_config["prioritized_replay"] = False
algo = config.build()

result = {}
logger1 = []
for i in range(episodes):
    start_time = datetime.now()
    results = algo.train()
    end_time = datetime.now()
    time_diff = end_time-start_time
    logger1.append([i,results["episode_reward_mean"],0])
    print("Epoch",i,"Episode Mean Reward",results["episode_reward_mean"], "Duration(s)",time_diff.total_seconds())

print("done")

config = DQNConfig()
config = config.environment(environ)
if torch.cuda.is_available(): config = config.resources(num_gpus = 2)
config.replay_buffer_config["prioritized_replay"] = True
config.replay_buffer_config["capacity"] = 5000
config.double_q = True
print(config.to_dict())
algo = config.build()

logger2 = []
for i in range(episodes):
    start_time = datetime.now()
    results = algo.train()
    end_time = datetime.now()
    time_diff = end_time - start_time
    logger2.append([i,results["episode_reward_mean"],0])
    print("Epoch",i,"Episode Mean Reward",results["episode_reward_mean"], "Duration(s)",time_diff.total_seconds())


#Calculate the cumulative average over the episodes
logger1 = np.array(logger1)
for i in range(len(logger1)):
    logger1[i][2] = sum(logger1[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward

logger2 = np.array(logger2)
for i in range(len(logger1)):
    logger2[i][2] = sum(logger2[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward

plt.plot(logger1[:, [0]], logger1[:, [1]], label="False")
plt.plot(logger2[:, [0]], logger2[:, [1]], label="True")
plt.legend()
plt.show()