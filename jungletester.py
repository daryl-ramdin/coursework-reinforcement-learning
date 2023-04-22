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

  env = JungleEnv(7)
  kwargs = {}

  dqnagent = dqn.ClassicDQNAgent(env=env, buffer_size=5000, batch_size=500,
                                 learning_rate=1e-02,
                                 gamma=1,
                                 target_update_interval=10,
                                 epsilon=0.9, epsilon_decay=0.99,
                                 loss_type=dqn.LossFunction.MSE,
                                 duelling=False,
                                 device=device
                                 )

  results = dqnagent.train(episode_count=50)

  rewards = np.array([[res["episode"], res["episode_reward"], 0] for res in results])
  steps_in_episode = np.array([[res["episode"], res["steps_taken"], 0] for res in results])
  ending_topography = np.array([[res["episode"], res["topography"], 0] for res in results])

  for i in range(len(results)):
      rewards[i][2] = sum(rewards[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward
      steps_in_episode[i][2] = sum(steps_in_episode[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward

  plt.figure(1)
  plt.plot(rewards[:, [0]], rewards[:, [2]], label="Average cumulative Reward")
  plt.legend()
  plt.show()

  plt.figure(2)
  plt.plot(steps_in_episode[:, [0]], steps_in_episode[:, [2]], label="Average Steps Taken")
  plt.legend()
  plt.show()