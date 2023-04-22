import sys
sys.path.insert(0,"/content/drive/MyDrive/Colab Notebooks")
#!pip install gymnasium

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #INM707 Lab 8
import random
import jungle
from jungleenv import JungleEnv
import dqn
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch.optim as optim

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("cuda")
else:
  device = torch.device("cpu")
  print("cpu")

# Set Q Learning parameters
experiment_name = "exp1"
episodes = 200



parameters = [
    {"experiment_id":1, "seed":45, "algorithm":"Double", "episodes":episodes, "epsilon": 0.9, "gamma":0.7, "epsilon_decay":0.999, "learning_rate":1e-03, "buffer_size":1000, "batch_size":100, "target_update_interval":100, "loss_type":dqn.LossFunction.MSE,"device":device,"duelling":True, "sizeof_hidden":256,"number_of_hidden": 4,"number_of_common": 2,"number_of_value": 2,"number_of_advantage": 2,"display":["number_of_common","number_of_value","number_of_advantage"]}
    ,{"experiment_id":2, "seed":45, "algorithm":"Double", "episodes":episodes, "epsilon": 0.9, "gamma":0.7, "epsilon_decay":0.999, "learning_rate":1e-03, "buffer_size":1000, "batch_size":100, "target_update_interval":100, "loss_type":dqn.LossFunction.MSE,"device":device,"duelling":True, "sizeof_hidden":256,"number_of_hidden": 4,"number_of_common": 4,"number_of_value": 2,"number_of_advantage": 2,"display":["number_of_common","number_of_value","number_of_advantage"]}
    ,{"experiment_id":3, "seed":45, "algorithm":"Double", "episodes":episodes, "epsilon": 0.9, "gamma":0.7, "epsilon_decay":0.999, "learning_rate":1e-03, "buffer_size":1000, "batch_size":100, "target_update_interval":100, "loss_type":dqn.LossFunction.MSE,"device":device,"duelling":True, "sizeof_hidden":256,"number_of_hidden": 4,"number_of_common": 2,"number_of_value": 4,"number_of_advantage": 2,"display":["number_of_common","number_of_value","number_of_advantage"]}
    ,{"experiment_id":4, "seed":45, "algorithm":"Double", "episodes":episodes, "epsilon": 0.9, "gamma":0.7, "epsilon_decay":0.999, "learning_rate":1e-03, "buffer_size":1000, "batch_size":100, "target_update_interval":100, "loss_type":dqn.LossFunction.MSE,"device":device,"duelling":True, "sizeof_hidden":256,"number_of_hidden": 4,"number_of_common": 2,"number_of_value": 2,"number_of_advantage": 4,"display":["number_of_common","number_of_value","number_of_advantage"]}
    ,{"experiment_id":5, "seed":45, "algorithm":"Double", "episodes":episodes, "epsilon": 0.9, "gamma":0.7, "epsilon_decay":0.999, "learning_rate":1e-03, "buffer_size":1000, "batch_size":100, "target_update_interval":100, "loss_type":dqn.LossFunction.MSE,"device":device,"duelling":True, "sizeof_hidden":256,"number_of_hidden": 4,"number_of_common": 4,"number_of_value": 4,"number_of_advantage": 4,"display":["number_of_common","number_of_value","number_of_advantage"]}
]
env = JungleEnv(7)
kwargs = {}

results = []
for kwargs in parameters:
    if kwargs["algorithm"] == "Classic":
      dqnagent = dqn.ClassicDQNAgent(env, **kwargs)
    else:
        dqnagent = dqn.DoubleDQNAgent(env,**kwargs)

    results += dqnagent.train(experiment_id = kwargs["experiment_id"],episodes=kwargs["episodes"], seed=kwargs["seed"])

for params in parameters:
    # Get the experiment id
    id = params["experiment_id"]

    # Get all the results for this experiment
    exp_res = [res for res in results if res["experiment_id"] == id]
    settings = params
    display_params = params["display"]

    # Build the label
    display_label = ""
    for item in display_params:
        display_label += item + ": " + str(settings[item]) + " "

    # For the experiment, get the metrics to display
    rewards = np.array([[res["episode"], res["episode_reward"], 0] for res in exp_res])
    steps_in_episode = np.array([[res["episode"], res["steps_taken"], 0] for res in exp_res])
    ending_topography = np.array([[res["episode"], res["topography"], 0] for res in exp_res])
    treasure_counter = np.array([[res["episode"], res["treasure_counter"], 0] for res in exp_res])

    for i in range(len(rewards)):
        rewards[i][2] = sum(rewards[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward
        steps_in_episode[i][2] = sum(steps_in_episode[0:i + 1, [1]]) / (i + 1)  # calculate the average number of steps
        # treasure_counter[i][2] = sum(treasure_counter[0:i + 1, [1]]) / (i + 1)  # calculate the average number of steps

    plt.figure(1)
    plt.plot(rewards[:, [0]], rewards[:, [2]], label=display_label)
    plt.title("Average Cummulative Reward")
    plt.figure(2)
    plt.plot(steps_in_episode[:, [0]], steps_in_episode[:, [2]], label=display_label)
    plt.title("Running Average Number of Steps")
    plt.figure(3)
    plt.plot(treasure_counter[:, [0]], treasure_counter[:, [1]], label=display_label)
    plt.title("Treasure Counter")

plt.figure(1)
plt.legend()

plt.figure(2)
plt.legend()

plt.figure(3)
plt.legend()

plt.show()

print(results)