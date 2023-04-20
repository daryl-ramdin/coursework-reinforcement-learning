from matplotlib import pyplot as plt
from jungle import  Jungle
from hiker import HikerAgent
import numpy as np
import pandas as pd




#Let's setup the environment


jungle = Jungle(rows=5,cols=4,vanishing_treasure=False)
jungle.add_topography([(1, 3, "M")])
jungle.add_topography([(2, 2, "S")])
jungle.add_topography([(4, 3, "B")])
jungle.add_topography([(5, 2, "R")])
jungle.add_topography([(4, 1, "L")])
jungle.add_topography([(4, 4, "E"), (5, 4, "E")])
jungle.add_topography([(1, 4, "T")])
jungle.fill_r_matrix()

#Set Q Learning parameters
epsilon = 1 #The higher the value, the more exploration
alpha = 0.5 #learning rate
gamma = 0.2 #discount rate
epsilon_decay = 0.999

#Create our agent
hiker = HikerAgent((1,1),epsilon=epsilon,alpha=alpha,gamma=gamma,policy_type='Epsilon-Greedy',jungle=jungle,learning_algorithm= "Q-Learning")
print(jungle.jungle_floor)

#Start training our Agent ref: INM707 La4
tracker = {"last_episode":0,"trend":[]}
metrics = {"E":1,"S":2,"CR":3,"Occurrence":4}
logger = {"E":{"last_episode":0,"trend":[]},
          "S":{"last_episode":0,"trend":[]},
          "_":{"last_episode":0,"trend":[]},
          "R":{"last_episode":0,"trend":[]},
          "T":{"last_episode":0,"trend":[]},
          "M":{"last_episode":0,"trend":[]},
          "L":{"last_episode":0,"trend":[]},
          "B":{"last_episode":0,"trend":[]},
          "CR":{"trend":[]},
          "Occurrence":{"trend":[]}
          }

episodes = 1000
timesteps = 100000

for episode in range(episodes):
    hiker.random_start()
    number_of_steps = 0

    #Run through the timesteps until we reach the end our our timestep limit
    for timestep in range(timesteps):
        if not hiker.move(): break
        number_of_steps+=1
        hiker.epsilon = hiker.epsilon * epsilon_decay
        #hiker.alpha = hiker.alpha - (hiker.alpha * 1e-5)
        #hiker.gamma = min(hiker.gamma + (hiker.gamma * 1e-5),1)


    #Let's see what state the hiker ended up in
    topography = hiker.get_current_topography()
    if topography!="E" and topography!="S": print("Hiker ended at:", hiker.get_current_topography())

    #Add the metrics
    last_episode = logger[topography]["last_episode"]
    epochs_from_last = episode - last_episode
    logger[topography]["trend"].append([episode,number_of_steps, epochs_from_last])
    logger[topography]["last_episode"] = episode
    logger["CR"]["trend"].append([episode,hiker.cumulative_reward])
    logger["Occurrence"]["trend"].append([episode, topography])
print('Final Q matrix: \n{}'.format(hiker.q_matrix.round(0)))

x = np.arange(0,episodes)
x = np.c_[x,np.zeros((episodes,2),dtype=int)]
plt.figure(1)
fig, ax = plt.subplots(3)
#Calculate additional metrics
for metric in [m for m in metrics.keys() if m not in ["CR","Occurrence"]]:
    if metric!="CR" and metric!="Occurrence":
        trend = np.array(logger[metric]["trend"])
        trend = np.c_[trend,np.zeros((len(trend),3),dtype=float)]
        #Our trend has format [epoch, number_of_timesteps, episode_distance]
        for i in range(len(trend)):
            x[int(trend[i][0])][metrics[metric]] = 1
            trend[i][3] = sum(trend[0:i + 1, [1]]) / (i + 1) #calculate the number of steps
            trend[i][4] = sum(trend[0:i + 1, [2]]) / (i + 1) #calculate the average episode distance
            trend[i][5] = (i + 1)  # calculate the total number of occurences
        ax[0].plot(trend[:, [0]], trend[:, [3]],label=metric)
        ax[1].plot(trend[:, [0]], trend[:, [4]],label=metric)
plt.legend()
plt.show()

plt.figure(2)
for metric in [m for m in metrics.keys() if m in ["CR"]]:
    if metric=="CR":
        trend = np.array(logger[metric]["trend"])
        trend = np.c_[trend,np.zeros((len(trend),1),dtype=float)]
        #Our trend has format [epoch, cumulative_reward]
        for i in range(len(trend)):
            trend[i][2] = sum(trend[0:i + 1, [1]]) / (i + 1) #calculate the average cumulative reward
        plt.plot(trend[:, [0]], trend[:, [2]],label=metric)
plt.show()

print(x)

for i in range(1,len(x)):
    x[i][1] = x[i-1][1]+x[i][1]
    x[i][2] = x[i - 1][2] + x[i][2]
print(x)

plt.figure(3)
plt.bar(x[:,[0]].reshape((episodes,)),x[:,[1]].reshape((episodes,)), label="E")
plt.bar(x[:,[0]].reshape((episodes,)),x[:,[2]].reshape((episodes,)), label="S")
plt.legend()
plt.show()


import pandas as pd
my_df = pd.DataFrame(logger)
my_df.to_csv('my_array.csv',header = False, index= False)

'''
#Our trend is of the format [epoch,number_of_timesteps,epochs_from_last, average_epochs, average_timesteps]
#Get the average episode distance at each epsiode

trend = np.array(trend)
for i in range(len(trend)):
    trend[i][3] = sum(trend[0:i+1,[2]])/(i+1)
    trend[i][4] = sum(trend[0:i + 1, [1]]) / (i + 1)
    plt.plot(trend)
#print(trend)

fig, ax = plt.subplots(2)
i = 0
#for key in metrics.keys():
key = "E"
vals = trend
if len(vals) > 0:
    #ax[i].plot(vals[:,[0]],vals[:,[1]], '.', alpha=0.4)
    ax[i].plot(vals[:, [0]], vals[:, [4]])
    ax[i].set_title(key)
#i+=1

trend = logger["CR"]
vals = np.array(trend)
for i in range(len(vals)):
    vals[i][2] = sum(vals[0:i+1,[1]])/(i+1)
ax[1].plot(vals[:, [0]], vals[:, [2]])
ax[1].set_title("CR")

plt.show()

print(jungle.jungle_floor)
#print(hiker.show_path([1,1]))
print("Done")
#hiker.show_path([1,1])
'''

