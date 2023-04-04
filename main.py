from matplotlib import pyplot as plt
from environment import  JungleEnv
from agent import HikerAgent
import numpy as np




#Let's setup the environment
jungle = JungleEnv(rows=7,cols=4,vanishing_treasure=True)
jungle.add_mountains([(1, 3), (2, 3)])
jungle.add_sinkholes([(7, 3)])
jungle.add_bears([(4, 3)])
jungle.add_rivers([(4, 2)])
jungle.add_lakes([(4, 1)])
jungle.add_exits([(4, 4), (7, 4)])
jungle.add_treasure([(7,1)])
jungle.fill_r_matrix()

#Set Q Learning parameters
epsilon = 0.1 #The higher the value, the more exploration
alpha = 0.2 #learning rate
gamma = 0.8 #discount rate

#Create our agent
hiker = HikerAgent((1,1),epsilon=epsilon,alpha=alpha,gamma=gamma,policy_type='Epsilon-Greedy',jungle=jungle,learning_algorithm= "Q-Learning")
print(jungle.jungle_floor)

#Start training our Agent ref: INM707 Lab4
tracker = {"last_episode":0,"trend":[]}
metrics = ["E","S","F","R","T","M","L","B", "CR"]
logger = {"E":{"last_episode":0,"trend":[]},"S":{"last_episode":0,"trend":[]},"F":{"last_episode":0,"trend":[]},"R":{"last_episode":0,"trend":[]},"T":{"last_episode":0,"trend":[]},"M":{"last_episode":0,"trend":[]},"L":{"last_episode":0,"trend":[]},"B":{"last_episode":0,"trend":[]}, "CR":[]}
episodes = 100
timesteps = 50000

for episode in range(episodes):
    hiker.random_start()
    last_timestep = 0

    #Run through the timesteps until we reach the end our our timestep limit
    for timestep in range(timesteps):
        if not hiker.move(): break
        last_timestep+=1


    #Let's see what state the hiker ended up in
    topography = hiker.get_current_topography()
    if topography!="E" and topography!="S":
        print("Hiker ended at:", hiker.get_current_topography())
    last_episode = logger[topography]["last_episode"]
    epochs_from_last = episode - last_episode
    logger[topography]["trend"].append([episode,last_timestep, epochs_from_last,0,0])
    logger[topography]["last_episode"] = episode
    logger["CR"].append([episode,hiker.cumulative_reward])

print('Final Q matrix: \n{}'.format(hiker.q_matrix.round(0)))



#Let's plot a graph of the number of timesteps per epoch to reach the exit

trend = logger["E"]["trend"]
#Our trend is of the format [epoch,number_of_timesteps,epochs_from_last]
#Get the average episode distance at each epsiode

trend = np.array(trend)
for i in range(len(trend)):
    trend[i][3] = sum(trend[0:i+1,[2]])/(i+1)
    trend[i][4] = sum(trend[0:i + 1, [1]]) / (i + 1)

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
ax[1].plot(vals[:, [0]], vals[:, [1]])
ax[1].set_title("CR")

plt.show()
'''
print(jungle.jungle_floor)
#print(hiker.show_path([1,1]))
print("Done")
#hiker.show_path([1,1])
'''
