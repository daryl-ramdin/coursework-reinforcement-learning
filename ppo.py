import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #INM707 Lab 8

#Reference for this PPO implementation came from
# https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
# and https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py

class actor(nn.Module):
    #ref: https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
    #This is the DQN class
    def __init__(self, sizeof_obs_space=2, sizeof_act_space=4,sizeof_hidden=256,number_of_layers=2):
        #ref: INM 707 Lab 8 Feedback
        super().__init__()
        self.sizeof_obs_space = sizeof_obs_space
        self.sizeof_act_space = sizeof_act_space
        self.sizeof_hidden = sizeof_hidden

        #Create the first layer
        self.seq = nn.Sequential(
            nn.Linear(self.sizeof_obs_space, self.sizeof_hidden),
            nn.ReLU())

        #Add the intermediate layers
        for i in range(0, number_of_layers-2):
            self.seq.append(module=nn.Linear(self.sizeof_hidden, self.sizeof_hidden))
            self.seq.append(nn.ReLU())

        #add the final layer
        self.seq.append(nn.Linear(self.sizeof_hidden, self.sizeof_act_space))
        self.seq.append(nn.Softmax())


    def forward(self,input):
        #Run the forward pass. ref: INM707 Lab 8
        output = self.seq(input)
        return output

class critic(nn.Module):
    # ref: https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
    #This is the DQN class
    def __init__(self, sizeof_obs_space=2, sizeof_hidden=256,number_of_layers=4):
        #ref: INM 707 Lab 8 Feedback
        super().__init__()
        self.sizeof_obs_space = sizeof_obs_space
        self.sizeof_hidden = sizeof_hidden

        #Create the first layer
        self.seq = nn.Sequential(
            nn.Linear(self.sizeof_obs_space, self.sizeof_hidden),
            nn.ReLU())

        #Add the intermediate layers
        for i in range(0, number_of_layers-2):
            self.seq.append(module=nn.Linear(self.sizeof_hidden, self.sizeof_hidden))
            self.seq.append(nn.ReLU())

        #add the final layer
        self.seq.append(nn.Linear(self.sizeof_hidden, 1))


    def forward(self,input):
        #Run the forward pass. ref: INM707 Lab 8
        output = self.seq(input)
        return output


class PPOAgent():
    def __init__(self,env, sizeof_obs_space, sizeof_action_space):
        self.actor = actor(sizeof_obs_space=sizeof_obs_space,sizeof_act_space=sizeof_action_space,number_of_layers=3,sizeof_hidden=64)
        self.critic = critic(sizeof_obs_space=sizeof_obs_space,sizeof_hidden=64,number_of_layers=3)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=1e-06)
        self. critic_optimizer = optim.AdamW(self.critic.parameters(), lr=1e-06)
        self.clipping_parameter = 0.2

        self.episodes_per_batch = 1000
        self.max_timesteps_per_episode = 3000

        self.batch_timesteps = 1000 #Number of timesteps per batch
        self.episode_max_timesteps = 3000 #maximum number of timesteps in an entire episode
        self.env = env

    def select_action(self,state):
        prob = torch.tensor(self.actor(state),dtype=torch.float)

        dist = torch.distributions.categorical.Categorical(prob)

        action = dist.sample()

        return action.item()

    def train(self,timesteps):

        for i in range(timesteps):

            #Get a batch of transitions
            observations, actions, probs, rtgs, lens = self.get_batch()

            #For the batch, calculate the value of each observation
            #as well as the log probability of each action

            #Our next step is to update the network


    def get_batch(self):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtg = []
        batch_lens = []
        #Step through the environment until we get the required number of transitions

        for i in range(self.episodes_per_batch):
            obs = self.env.reset()

            batch_obs.append(obs)
            episode_reward = 0
            for t in range(self.max_timesteps_per_episode):
                #get the action for the observation
                action = self.select_action(obs)
                #Step through the environment with the action
                obs, reward, terminated,_ = self.env.step(action)

                #add the transition information
                batch_obs.append(obs)
                batch_actions.append(action)
                batch_rewards.append(reward)
                #if terminated, exit
                if terminated: break

        #Once we have collected the batch, return as tensor
        batch_obs = torch.tensor(batch_obs,dtype=torch.float)
        batch_actions = torch.tensor(batch_actions)
        batch_rewards = torch.tensor(batch_rewards)
        return

    def record_batch(self,data):


        return

agent = PPOAgent(2,4)
agent.train(2)
