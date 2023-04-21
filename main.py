from matplotlib import pyplot as plt
from jungle import  Jungle
from hiker import HikerAgent
import numpy as np
import pandas as pd

#Set Q Learning parameters
experiment_name = "exp1"
episodes = 1000

parameters = [{"experiment_id":1,"sarsa":False, "epsilon": 1, "alpha":1, "gamma": 1, "epsilon_decay":0.999, "timesteps":10, "episodes":episodes, "display":["sarsa"]},
    {"experiment_id":2,"sarsa":True,"epsilon": 1, "alpha":1, "gamma": 1, "epsilon_decay":0.999, "timesteps":10, "episodes":episodes,"display":["sarsa"]},
    ]

jungle = Jungle(rows=5,cols=4,vanishing_treasure=False)
jungle.add_topography([(1, 3, "M")])
jungle.add_topography([(2, 2, "S")])
jungle.add_topography([(4, 3, "B")])
jungle.add_topography([(5, 2, "R")])
jungle.add_topography([(4, 1, "L")])
jungle.add_topography([(4, 4, "E"), (5, 4, "E")])
jungle.add_topography([(1, 4, "T")])
jungle.fill_r_matrix()

print(jungle.jungle_floor)

results = []
for kwargs in parameters:
    hiker = HikerAgent(jungle=jungle,sarsa = kwargs["sarsa"])
    results += hiker.train(**kwargs)

for params in parameters:
    # Get the experiment id
    id = params["experiment_id"]

    # Get all the results for this experiment
    exp_res = [res for res in results if res["experiment_id"] == id]
    settings = exp_res[0]["parameters"]
    display_params = exp_res[0]["parameters"]["display"]
    # Build the label
    display_label = ""
    for item in display_params:
        display_label += item + ": " + str(settings[item]) + " "

    # For the experiment, get the metrics to display
    rewards = np.array([[res["episode"], res["episode_reward"], 0] for res in exp_res])
    steps_in_episode = np.array([[res["episode"], res["steps_taken"], 0] for res in exp_res])
    ending_topography = np.array([[res["episode"], res["topography"], 0] for res in exp_res])

    for i in range(len(rewards)):
        rewards[i][2] = sum(rewards[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward
        steps_in_episode[i][2] = sum(steps_in_episode[0:i + 1, [1]]) / (i + 1)  # calculate the average number of steps
    plt.figure(1)
    plt.plot(rewards[:, [0]], rewards[:, [2]], label=display_label)
    plt.title("Average Cummulative Reward")
    plt.figure(2)
    plt.plot(steps_in_episode[:, [0]], steps_in_episode[:, [2]], label=display_label)
    plt.title("Average Number of Steps")

plt.figure(1)
plt.legend()

plt.figure(2)
plt.legend()

plt.show()