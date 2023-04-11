import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #INM707 Lab 8
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = 0.1
#ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple("Transition",("state", "action", "next_state", "reward"))

class DQN(nn.Module):
    #This is the DQN class
    def __init__(self, configuration):
        #ref: INM 707 Lab 8 Feedback
        super().__init__()
        '''
        :param configuration: [{in:int, out:int}]
        '''
        self.layers = []
        for layer in configuration:
            self.layers.append(nn.Linear(in_features=layer["in"],out_features=layer["out"]))

    def forward(self,input):
        #Run the forward pass. ref: INM707 Lab 8
        i = 0
        for i in range(len(self.layers)-1):
            input = F.relu(self.layers[i](input))
        #return the output
        output = self.layers[i+1](input)
        return output


def get_next_action(state):
    #We operate in epsilon greedy
    action = None

    if np.random.uniform() > epsilon:
        #We use the best
        action = q_net(torch.tensor(state,device=device,dtype=torch.float))
    else:
        #We explore
        return env.action_space.sample()

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.buffer = deque([],maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size


    def push(self,state,action,next_state,reward):
        #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.buffer.append(Transition(state,action,next_state,reward))

    def __len__(self):
        return len(self.buffer_size)

    def __getitem__(self, item=None):
        return random.sample(self.buffer,self.batch_size)

def train():
    return

def update_target():
    return



#Create our environment
env = gym.make('MountainCar-v0')

#Reset the environment and get the observation. This is an array [position,velocity]

#Let's create the configuration for our network
sizeof_obs = 2
sizeof_actn = 3
config = [{"in":sizeof_obs,"out":256},
          {"in":256,"out":256},
          {"in":256,"out":256},
          {"in":256,"out":sizeof_actn}]

q_net = DQN(config).to(device)
tgt_net = DQN(config).to(device)
tgt_net.load_state_dict(q_net.state_dict())



#Let's get the observation. ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
replay_buffer = ReplayBuffer(10,5)



episode_count = 1

#Training loop
for i in range(episode_count):
    #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # reset the environment
    state, info = env.reset()

    while 1:

        #Get the action
        action = get_next_action(state)

        #Get the transition for the action, ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        observation, reward, terminated, truncated, _ = env.step(action)

        next_state = observation

        #Store the transition
        replay_buffer.push(state, action, next_state, truncated)

        #Let our model train on a batch of transitions
        train()

        state = next_state
        break

    #Update the weights for the target network
    update_target()










