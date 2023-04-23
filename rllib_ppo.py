# Reference for this code came from https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn

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

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda')

print("Selected device:",device)

#Reference for this code came from https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
environ = "Riverraid-ramDeterministic-v4"
episodes = 100

parameters = [
      {"experiment_id":1, "lr":5e-05, "episodes":episodes, "display":["lr"]}
]

final_results = []
for param in parameters:
  config = PPOConfig()
  config = config.environment(environ)
  if torch.cuda.is_available(): config = config.resources(num_gpus = 1)
  algo = config.build()


  episode_results = []
  start_time = datetime.now()
  for episode in range(param["episodes"]):
      results = algo.train()
      episode_results.append({"experiment_id": param["experiment_id"], "parameters": param, "episode": episode,"episode_mean_reward": results["episode_reward_mean"]})
      if episode%10==0:
        end_time = datetime.now()
        time_diff = end_time-start_time
        print("Epoch",episode,"Episode Mean Reward",results["episode_reward_mean"], "Duration(s)",time_diff.total_seconds())
        start_time = datetime.now()
  final_results += episode_results

for params in parameters:
  # Get the experiment id
  id = params["experiment_id"]
  display_params = params["display"]
  # Get all the results for this experiment
  exp_res = []
  for res in final_results:
      if res["experiment_id"] == id:
          exp_res.append(res)

  # Build the label
  display_label = ""
  for item in display_params:
      display_label += item + ": " + str(params[item]) + " "

  # For the experiment, get the metrics to display
  rewards = np.array([[res["episode"], res["episode_mean_reward"], 0] for res in exp_res])

  for i in range(len(rewards)):
      rewards[i][2] = sum(rewards[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward

  plt.figure(1)
  plt.plot(rewards[:, [0]], rewards[:, [1]], label=display_label)
  plt.title("Episode Mean Reward")

plt.figure(1)
plt.legend()

plt.show()
