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

class LossFunction():
    SmoothL1= "SmoothL1"
    MSE = "MSE"
    def __init__(self,loss_type):
        self.loss_type = loss_type
    def calculate(self,input,target):
        loss = None
        if self.loss_type==LossFunction.MSE:
            loss = F.mse_loss(input,target)
        elif self.loss_type==LossFunction.SmoothL1:
            loss = F.smooth_l1_loss(input,target)
        return loss

class DQN(nn.Module):
    #This is the DQN class
    def __init__(self, sizeof_obs_space=2, sizeof_act_space=4,sizeof_hidden=254):
        #ref: INM 707 Lab 8 Feedback
        super().__init__()
        '''
        :param configuration: [{in:int, out:int}]
        '''
        self.sizeof_obs_space = sizeof_obs_space
        self.sizeof_act_space = sizeof_act_space
        self.sizeof_hidden = sizeof_hidden

        self.seq = nn.Sequential(
            nn.Linear(self.sizeof_obs_space, self.sizeof_hidden),
            nn.ReLU(),
            nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
            nn.ReLU(),
            nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
            nn.ReLU(),
            nn.Linear(self.sizeof_hidden, self.sizeof_act_space)
        )
        '''
        self.fc1 = nn.Linear(self.sizeof_obs_space,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.sizeof_act_space)
        '''
        '''
        self.layers = []
        for layer in configuration:
            self.layers.append(nn.Linear(in_features=layer["in"],out_features=layer["out"]))
        '''

    def forward(self,input):
        #Run the forward pass. ref: INM707 Lab 8
        '''
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = F.relu(self.fc3(input))
        output = self.fc4(input)
        '''
        '''
        i = 0
        for i in range(len(self.layers)-1):
            input = F.relu(self.layers[i](input))
        #return the output
        output = self.layers[i+1](input)
        '''
        output = self.seq(input)
        return output


class DuellingDQN(nn.Module):
    #ref: https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
    def __init__(self,sizeof_obs_space=2, sizeof_act_space=4,sizeof_hidden=256):
        super().__init__()
        self.sizeof_obs_space = sizeof_obs_space
        self.sizeof_act_space = sizeof_act_space
        self.sizeof_hidden = 256
        self.mode = "mean"

        #Our DQN contains two additional layers, one for the value and the other the advantage
        #ref: https://arxiv.org/pdf/1511.06581.pdf
        self.common_stream = nn.Sequential(
            nn.Linear(self.sizeof_obs_space,self.sizeof_hidden),
            nn.ReLU(),
            nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.sizeof_hidden,self.sizeof_hidden),
            nn.ReLU(),
            nn.Linear(self.sizeof_hidden, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.sizeof_hidden,self.sizeof_hidden),
            nn.ReLU(),
            nn.Linear(self.sizeof_hidden, self.sizeof_act_space)
        )

    def forward(self, input):
        # Run the forward pass. ref: INM707 Lab 8
        common_value = self.common_stream(input)
        state_value = self.value_stream(common_value)
        advantage = self.advantage_stream(common_value)

        q_value = state_value + (advantage - advantage.mean())

        return q_value

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

class DQNAgent():
    def __init__(self,env, epsilon=0.5, gamma=0.99, epsilon_decay=1e-06,buffer_size=50, batch_size = 5, loss_type="MSE", device=None):
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.loss_function = LossFunction(loss_type)
        self.env = env

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Let's get the observation. ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

class ClassicDQNAgent(DQNAgent):

    def __init__(self, env, epsilon=0.9, gamma=0.99, epsilon_decay=0.99,buffer_size=50, batch_size = 5, loss_type ="MSE", device=None,
                 **kwargs):
        super().__init__(env, epsilon, gamma, epsilon_decay, buffer_size, batch_size, loss_type, device)


        if "duellingdqn" in kwargs:
            self.P_network = DuellingDQN()
            self.T_network = DuellingDQN()
        else:
            self.P_network = DQN()
            self.T_network = DQN()

        self.optimizer = optim.AdamW(self.P_network.parameters(), lr=1e-05)

        self.T_network.load_state_dict(self.P_network.state_dict())


    def get_next_action(self, state):
        #The state is a Dict {"jungle_position":np.array[row,col]}
        state = torch.tensor(state["jungle_position"],device=device,dtype=torch.float)
        #We operate in epsilon greedy
        next_action = None

        if np.random.uniform() > EPSILON:
            #We use the best
            next_action = self.P_network(state).argmax()
        else:
            #We explore
            next_action = torch.tensor(self.env.action_space.sample(),device=device,dtype=torch.int64)
        return next_action

    def train(self):
        return

    def update_target(self):
        return

    def train(self):
        return

    def learn_policy(self,episode_count=100):

        #Reset and render our environment
        self.env.reset()
        self.env.render()

        '''
        #Let's create the configuration for our network
        sizeof_obs = 2
        sizeof_actn = 4
        config = [{"in":sizeof_obs,"out":256},
                  {"in":256,"out":256},
                  {"in":256,"out":256},
                  {"in":256,"out":sizeof_actn}]
        
        self.P_network = DQN(config).to(device)
        self.T_network = DQN(config).to(device)
        self.T_network.load_state_dict(self.P_network.state_dict())
        optimizer = optim.AdamW(self.P_network.parameters(),lr=1e-05)
        '''
        episode_logger = []
        #Training loop
        for episode in range(episode_count):
            #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            # reset the environment
            state, info = self.env.reset()
            go = True
            episode_reward = 0
            while go:

                #Get the action
                action = self.get_next_action(state)

                #Choose the best action
                sel_act = action
                #Get the transition for the action, ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                observation, reward, terminated, truncated, _ = self.env.step(sel_act.item())

                episode_reward+=reward
                next_state = observation

                #Get the states to put in the buffer
                pstate = state["jungle_position"].tolist()
                pnstate = next_state["jungle_position"].tolist()

                #INM 707 Lab 8
                go = not terminated
                if terminated:
                    pnstate = None

                #Store the transition
                self.replay_buffer.push(pstate, sel_act, pnstate, reward)

                #Let our model train on a batch of transitions
                if len(self.replay_buffer) > self.batch_size:
                    batch = next(iter(self.replay_buffer))
                    #We now have a batch of transitions on which we will train our networks
                    i = 1
                    #Let's start extracting from the batch of transitions
                    states = torch.tensor([trn.state for trn in batch],device=device,dtype=torch.float)
                    non_termination_next_states = torch.tensor([trn.next_state for trn in batch if trn.next_state is not None],device=device,dtype=torch.float)
                    actions = torch.tensor([trn.action for trn in batch],device=device,dtype=torch.int64)
                    rewards = torch.tensor([trn.reward for trn in batch],device=device)

                    # Get the states from the batch and get the state action values
                    state_action_values = self.P_network(states).gather(1, actions.unsqueeze(1))

                    #The next step is to get the next state action values from the target network
                    #Some of our next states are final states and we do not calculate action values for them
                    #We must create a mask to block them out
                    non_terminating_mask = [trn.next_state is not None for trn in batch]

                    if (len(non_terminating_mask)) < len(batch):
                        print("Found onee")

                    next_state_action_value = torch.zeros(len(batch),device=device)

                    with torch.no_grad():
                        results = self.T_network(non_termination_next_states)
                    next_state_action_value[non_terminating_mask] = results.max(1)[0]
                    #Calculate the expected state action values
                    expected_state_action_values = (next_state_action_value * GAMMA) + rewards

                    loss = self.loss_function.calculate(state_action_values.squeeze(1), expected_state_action_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state

            print("Episode:",episode, "Reward:",episode_reward)
            episode_logger.append([episode,episode_reward,0])
            #Update the weights for the target network after every episode
            if episode%5==0:
                self.T_network.load_state_dict(self.P_network.state_dict())
            self.epsilon *= self.epsilon_decay
        print("Learning complete")
        return episode_logger

class DoubleDQNAgent(ClassicDQNAgent):
    #ref: https://www.datahubbs.com/double-deep-q-learning-to-get-the-most-out-of-your-dqn/
    def __init__(self, env, epsilon=0.5, gamma=0.99, epsilon_decay=1e-06,buffer_size=50, batch_size = 5, loss_type ="MSE", device=None,**kwargs):
        super().__init__(env, epsilon, gamma, epsilon_decay, buffer_size, batch_size, loss_type, device,**kwargs)

    def learn_policy(self, episode_count=100):

        # Reset and render our environment
        self.env.reset()
        self.env.render()

        episode_logger = []
        # Training loop
        for episode in range(episode_count):
            # ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            # reset the environment
            state, info = self.env.reset()
            go = True
            episode_reward = 0

            while go:

                # Get the action
                action = self.get_next_action(state)

                # Choose the best action
                sel_act = action
                # Get the transition for the action, ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                observation, reward, terminated, truncated, _ = self.env.step(sel_act.item())

                episode_reward += reward
                next_state = observation

                # Get the states to put in the buffer
                pstate = state["jungle_position"].tolist()
                pnstate = next_state["jungle_position"].tolist()

                # INM 707 Lab 8
                go = not terminated
                if terminated:
                    pnstate = None

                # Store the transition
                self.replay_buffer.push(pstate, sel_act, pnstate, reward)

                # Let our model train on a batch of transitions
                if len(self.replay_buffer) > self.batch_size:
                    batch = next(iter(self.replay_buffer))
                    # We now have a batch of transitions on which we will train our networks
                    i = 1
                    # Let's start extracting from the batch of transitions
                    states = torch.tensor([trn.state for trn in batch], device=device, dtype=torch.float)
                    non_termination_next_states = torch.tensor([trn.next_state for trn in batch if trn.next_state is not None],
                                                               device=device,dtype=torch.float)
                    actions = torch.tensor([trn.action for trn in batch], device=device, dtype=torch.int64)
                    rewards = torch.tensor([trn.reward for trn in batch], device=device)

                    # Get the states from the batch and get the state action values
                    state_action_values = self.P_network(states).gather(1, actions.unsqueeze(1))

                    # The next step is to get the next state action values from the target network
                    # Some of our next states are final states and we do not calculate action values for them
                    # We must create a mask to block them out
                    non_terminating_mask = [trn.next_state is not None for trn in batch]

                    if (len(non_terminating_mask)) < len(batch):
                        print("Found onee")


                    next_state_action_value = torch.zeros(len(batch), device=device)

                    #For Double DQN, we use our policy network to get the action
                    #with the best value for the next state and use this action to get
                    #the next state action value for the target network
                    best_next_state_actions = self.P_network(non_termination_next_states).argmax(1).unsqueeze(1)

                    with torch.no_grad():
                        next_state_action_value[non_terminating_mask] = self.T_network(non_termination_next_states).gather(1,best_next_state_actions).squeeze(1)

                    # Calculate the expected state action values
                    expected_state_action_values = (next_state_action_value * GAMMA) + rewards

                    loss = self.loss_function.calculate(state_action_values.squeeze(1), expected_state_action_values)
                    # loss =  criterion(state_action_value.unsqueeze(0).unsqueeze(1),expected_state_action_values.unsqueeze(0).unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state

            if episode%5==0:
                self.T_network.load_state_dict(self.P_network.state_dict())

            print("Episode:", episode, "Reward:", episode_reward, "Epsilon",self.epsilon)
            self.epsilon *= self.epsilon_decay
            episode_logger.append([episode, episode_reward, 0])
            # Update the weights for the target network after every episode
            self.T_network.load_state_dict(self.P_network.state_dict())
        print("Learning complete")
        return episode_logger


env = JungleEnv({"size":7})
kwargs = {}
dqnagent = ClassicDQNAgent(env,buffer_size=10000,batch_size=256,**kwargs,epsilon=0.9,epsilon_decay=0.99,loss_type="SmoothL1")
results = dqnagent.learn_policy(episode_count=50)
results = np.array(results)
for i in range(len(results)):
    results[i][2] = sum(results[0:i + 1, [1]]) / (i + 1)  # calculate the average cumulative reward
plt.plot(results[:, [0]], results[:, [2]], label="Average cumulative Reward")
plt.show()












