import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F #INM707 Lab 8
import gymnasium as gym
import random
from jungleenv import JungleEnv
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.5
GAMMA = 0.99
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
        self.fc1 = nn.Linear(2,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)
        '''
        self.layers = []
        for layer in configuration:
            self.layers.append(nn.Linear(in_features=layer["in"],out_features=layer["out"]))
        '''

    def forward(self,input):
        #Run the forward pass. ref: INM707 Lab 8
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = F.relu(self.fc3(input))
        output = self.fc4(input)
        '''
        i = 0
        for i in range(len(self.layers)-1):
            input = F.relu(self.layers[i](input))
        #return the output
        output = self.layers[i+1](input)
        '''
        return output


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
        return len(self.buffer)

    def __getitem__(self, item=None):
        return random.sample(self.buffer,self.batch_size)


def get_next_action(state):
    #The state is a Dict {"jungle_position":np.array[row,col]}
    state = torch.tensor(state["jungle_position"],device=device,dtype=torch.float)
    #We operate in epsilon greedy
    next_action = None

    if np.random.uniform() > EPSILON:
        #We use the best
        next_action = Q_network(state).argmax()
    else:
        #We explore
        next_action = torch.tensor(env.action_space.sample(),device=device,dtype=torch.int64)
    return next_action

def train():
    return

def update_target():
    return



#Create our environment and render it

env = JungleEnv({"size":7})
env.reset()
env.render()

#Reset the environment and get the observation. This is an array [position,velocity]

#Let's create the configuration for our network
sizeof_obs = 2
sizeof_actn = 4
config = [{"in":sizeof_obs,"out":256},
          {"in":256,"out":256},
          {"in":256,"out":256},
          {"in":256,"out":sizeof_actn}]

Q_network = DQN(config).to(device)
T_network = DQN(config).to(device)
T_network.load_state_dict(Q_network.state_dict())
optimizer = optim.AdamW(Q_network.parameters(),lr=1e-05)

#Let's get the observation. ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
replay_buffer = ReplayBuffer(50,5)

episode_count = 100

#Training loop
for i in range(episode_count):
    #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # reset the environment
    state, info = env.reset()
    go = True
    episode_reward = 0
    while go:

        #Get the action
        action = get_next_action(state)

        #Choose the best action
        sel_act = action
        #Get the transition for the action, ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        observation, reward, terminated, truncated, _ = env.step(sel_act.item())

        episode_reward+=reward
        next_state = observation

        #INM 707 Lab 8
        go = not terminated
        if terminated:
            next_state = None

        #Store the transition
        replay_buffer.push(state, sel_act, next_state, reward)

        #Let our model train on a batch of transitions
        if len(replay_buffer) > replay_buffer.batch_size:
            batch = next(iter(replay_buffer))
            #We now have a batch of transitions on which we will train our
            #Q_network. Get the states from the batch and get the best
            #predicted action
            for trn in batch:
                state_action_value = Q_network(torch.tensor(trn.state["jungle_position"],device=device,dtype=torch.float))
                #If the transitions next_state is non final, we use our target network to get
                #its value
                if trn.next_state is not None:
                    with torch.no_grad():
                        next_state_action = T_network(torch.tensor(trn.next_state["jungle_position"],device=device,dtype=torch.float))
                    #Let's calculate the expected Q value
                    expected_state_action_values = (next_state_action * GAMMA) + trn.reward
                    criterion = nn.SmoothL1Loss()
                    loss = F.mse_loss(state_action_value,expected_state_action_values)
                    #loss =  criterion(state_action_value.unsqueeze(0).unsqueeze(1),expected_state_action_values.unsqueeze(0).unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        state = next_state

    print("episode_reward",episode_reward)
    #Update the weights for the target network after every episode
    T_network.load_state_dict(Q_network.state_dict())












