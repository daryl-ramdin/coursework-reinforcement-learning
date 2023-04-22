import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #INM707 Lab 8
import random
from collections import namedtuple, deque
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, sizeof_obs_space=2, sizeof_act_space=4,sizeof_hidden=254,number_of_layers=4,device=None):
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
            nn.ReLU())

        for i in range(0, number_of_layers-2):
            self.seq.append(module=nn.Linear(self.sizeof_hidden, self.sizeof_hidden))
            self.seq.append(nn.ReLU())

        self.seq.append(nn.Linear(self.sizeof_hidden, self.sizeof_act_space))

        '''
        self.fc1 = nn.Linear(sizeof_obs_space, sizeof_hidden,device=device)
        self.fc2 = nn.Linear(sizeof_hidden, sizeof_hidden,device=device)
        self.fc3 = nn.Linear(sizeof_hidden, sizeof_hidden,device=device)
        self.fc4 = nn.Linear(sizeof_hidden, sizeof_act_space,device=device)
        '''

        self.seq.to(device=device)

        #ref https://www.geeksforgeeks.org/initialize-weights-in-pytorch/
        #ref: https://github.com/pytorch/examples/blob/main/dcgan/main.py#L95
        for mod in self.seq:
            if mod.__class__.__name__=="Linear":
                torch.nn.init.normal_(mod.weight,mean=0, std=1)

    def forward(self,input):
        #Run the forward pass. ref: INM707 Lab 8
        output = self.seq(input)
        # x = F.relu(self.fc1(input))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # output = self.fc4(x)
        return output


class DuellingDQN(nn.Module):
    #ref: https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
    def __init__(self,sizeof_obs_space=2, sizeof_act_space=4,sizeof_hidden=256,
                 number_of_common=2,number_of_value = 2,number_of_advantage=2, device=None):
        super().__init__()
        self.sizeof_obs_space = sizeof_obs_space
        self.sizeof_act_space = sizeof_act_space
        self.sizeof_hidden = 256
        self.mode = "mean"

        #Our DQN contains two additional layers, one for the value and the other the advantage
        #ref: https://arxiv.org/pdf/1511.06581.pdf
        self.common_stream = nn.Sequential(
            nn.Linear(self.sizeof_obs_space, self.sizeof_hidden),
            nn.ReLU())

        for i in range(0, number_of_common-1):
            self.common_stream.append(module=nn.Linear(self.sizeof_hidden, self.sizeof_hidden))
            self.common_stream.append(nn.ReLU())


        # self.common_stream = nn.Sequential(
        #     nn.Linear(self.sizeof_obs_space,self.sizeof_hidden),
        #     nn.ReLU(),
        #     nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
        #     nn.ReLU()
        # )

        # self.value_stream = nn.Sequential(
        #     nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
        #     nn.ReLU())

        for i in range(0, number_of_value):
            if i==0:
                #Add the first layer
                self.value_stream = nn.Sequential(
                    nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
                    nn.ReLU())
            elif i==number_of_value-1:
                #Add the last layer
                self.value_stream.append(nn.Linear(self.sizeof_hidden, 1))
            else:
                #Add all other layers
                self.value_stream.append(module=nn.Linear(self.sizeof_hidden, self.sizeof_hidden))
                self.value_stream.append(nn.ReLU())

        # self.value_stream = nn.Sequential(
        #     nn.Linear(self.sizeof_hidden,self.sizeof_hidden),
        #     nn.ReLU(),
        #     nn.Linear(self.sizeof_hidden, 1)
        # )

        # self.advantage_stream = nn.Sequential(
        #     nn.Linear(self.sizeof_hidden,self.sizeof_hidden),
        #     nn.ReLU(),
        #     nn.Linear(self.sizeof_hidden, self.sizeof_act_space)
        # )

        for i in range(0, number_of_advantage):
            if i == 0:
                # Add the first layer
                self.advantage_stream = nn.Sequential(
                    nn.Linear(self.sizeof_hidden, self.sizeof_hidden),
                    nn.ReLU())
            elif i == number_of_value - 1:
                # Add the last layer
                self.advantage_stream.append(nn.Linear(self.sizeof_hidden, 1))
            else:
                # Add all other layers
                self.advantage_stream.append(module=nn.Linear(self.sizeof_hidden, self.sizeof_hidden))
                self.advantage_stream.append(nn.ReLU())

        self.common_stream.to(device=device)
        self.value_stream.to(device=device)
        self.advantage_stream.to(device=device)
        # ref https://www.geeksforgeeks.org/initialize-weights-in-pytorch/
        # ref: https://github.com/pytorch/examples/blob/main/dcgan/main.py#L95
        for mod in self.common_stream:
            if mod.__class__.__name__ == "Linear":
                torch.nn.init.normal_(mod.weight, mean=0, std=1)
        for mod in self.value_stream:
            if mod.__class__.__name__ == "Linear":
                torch.nn.init.normal_(mod.weight, mean=0, std=1)
        for mod in self.advantage_stream:
            if mod.__class__.__name__ == "Linear":
                torch.nn.init.normal_(mod.weight, mean=0, std=1)

    def forward(self, input):
        # Run the forward pass. ref: INM707 Lab 8
        common_value = self.common_stream(input)
        state_value = self.value_stream(common_value)
        advantage = self.advantage_stream(common_value)

        q_value = state_value + (advantage - advantage.mean())

        return q_value

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size,state_transforms,action_transforms):
        #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.buffer = deque([],maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_transforms = state_transforms
        self.action_transforms = action_transforms


    def push(self,state,action,next_state,reward):
        #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        state = self.state_transforms(state).tolist()
        action = self.action_transforms(action)
        if next_state is not None:
            next_state=self.state_transforms(next_state).tolist()

        self.buffer.append(Transition(state,action,next_state,reward))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item=None):
        return random.sample(self.buffer,self.batch_size)


class ClassicDQNAgent():

    def __init__(self, env, **kwargs):
        self.epsilon = kwargs["epsilon"]
        self.gamma = kwargs["gamma"]
        self.epsilon_decay = kwargs["epsilon_decay"]
        self.buffer_size = kwargs["buffer_size"]
        self.batch_size = kwargs["batch_size"]
        self.loss_function = LossFunction(kwargs["loss_type"])
        self.env = env
        self.target_update_interval = kwargs["target_update_interval"]
        self.seed = kwargs["seed"]
        self.duelling = kwargs["duelling"]
        self.learning_rate = kwargs["learning_rate"]
        self.parameters = kwargs

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Let's get the observation. ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        #ref https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
        self.state_transforms = lambda state: state["jungle_position"]/self.env.size
        self.action_transforms = lambda action: action/self.env.action_space.n.item()
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size,self.state_transforms,self.action_transforms)

        if self.duelling==True:
            self.P_network = DuellingDQN(device=device,
                                         number_of_common=kwargs["number_of_common"],
                                         number_of_value=kwargs["number_of_value"],
                                         number_of_advantage=kwargs["number_of_advantage"])
            self.T_network = DuellingDQN(device=device,
                                         number_of_common=kwargs["number_of_common"],
                                         number_of_value=kwargs["number_of_value"],
                                         number_of_advantage=kwargs["number_of_advantage"])
        else:
            self.P_network = DQN(device=device,sizeof_hidden=kwargs["sizeof_hidden"])
            self.T_network = DQN(device=device,sizeof_hidden=kwargs["sizeof_hidden"])

        self.optimizer = optim.AdamW(self.P_network.parameters(), lr=self.learning_rate)

        self.T_network.load_state_dict(self.P_network.state_dict())

    def select_action(self, observation):
        #We operate in epsilon greedy
        next_action = None

        if np.random.uniform() > self.epsilon:
            # We use the best
            # ref: INM707 Lab 8
            state = torch.tensor(self.state_transforms(observation),device=device,dtype=torch.float)
            self.P_network.eval()
            with torch.no_grad():
                next_action = self.P_network(state).argmax()
            self.P_network.train()
        else:
            #We explore
            next_action = torch.tensor(self.env.action_space.sample(),device=device,dtype=torch.int64)

        return next_action.item()

    def train(self,experiment_id, episodes=100,seed=45):

        self.seed = seed
        #Reset and render our environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        #ref: https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(self.seed)

        episode_results = []
        #Training loop
        for episode in range(episodes):
            #ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            # reset the environment and get the observation
            # We increment the seed so that we can re-produce our results
            # but our experiments will still be different
            self.seed += 1
            random.seed(self.seed)
            np.random.seed(self.seed)
            #self.env.render()

            observation, info = self.env.reset(seed=self.seed)
            go = True
            episode_reward = 0
            steps_taken = 0
            # Our first step is to transform the state.
            state = observation
            while go:

                steps_taken += 1

                #Select the next action for the state
                action = self.select_action(state)

                #Get the transition for the action, ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                observation, reward, terminated, truncated, info = self.env.step(action)

                #Increment the reward and use the observation as the next state
                episode_reward += reward

                # INM 707 Lab 8
                if terminated:
                    next_state = None
                else:
                    next_state = observation

                #Store the transition
                #Normalise the action

                self.replay_buffer.push(state, action, next_state, reward)

                #Let our model train on a batch of transitions
                if len(self.replay_buffer) > self.batch_size:
                    self.optimize_policy()

                state = next_state

                #Determine whether to continue
                go = not terminated

            #Get the results
            episode_results.append({"experiment_id": experiment_id, "parameters": self.parameters, "episode":episode,"episode_reward":episode_reward,"steps_taken":steps_taken, "topography": info["topography"]})
            print("Episode",episode,"Reward",episode_reward, "Epsilon", self.epsilon, "Steps Taken",steps_taken,"Topography", info["topography"])

            #Update the weights for the target network after every episode
            if episode%self.target_update_interval==0:
                self.T_network.load_state_dict(self.P_network.state_dict())

            #Update epsilon at the end of every episode
            self.epsilon *= self.epsilon_decay

        print("Learning complete")
        return episode_results

    def optimize_policy(self):
        batch = next(iter(self.replay_buffer))
        # We now have a batch of transitions on which we will train our networks
        i = 1
        # Let's start extracting from the batch of transitions
        states = torch.tensor([trn.state for trn in batch], device=device, dtype=torch.float)

        non_termination_next_states = torch.tensor([trn.next_state for trn in batch if trn.next_state is not None],
                                                   device=device, dtype=torch.float)
        actions = torch.tensor([trn.action for trn in batch], device=device, dtype=torch.int64)
        rewards = torch.tensor([trn.reward for trn in batch], device=device)

        # Get the states from the batch and get the state action values
        state_action_values = self.P_network(states).gather(1, actions.unsqueeze(1))

        # The next step is to get the next state action values from the target network
        # Some of our next states are final states and we do not calculate action values for them
        # We must create a mask to block them out
        non_terminating_mask = [trn.next_state is not None for trn in batch]

        next_state_action_value = torch.zeros(len(batch), device=device)

        with torch.no_grad():
            results = self.T_network(non_termination_next_states)
        next_state_action_value[non_terminating_mask] = results.max(1)[0]

        # Calculate the expected state action values
        expected_state_action_values = (next_state_action_value * self.gamma) + rewards

        loss = self.loss_function.calculate(state_action_values.squeeze(1), expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DoubleDQNAgent(ClassicDQNAgent):
    #ref: https://www.datahubbs.com/double-deep-q-learning-to-get-the-most-out-of-your-dqn/
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def optimize_policy(self, episode_count=100):

        batch = next(iter(self.replay_buffer))
        # We now have a batch of transitions on which we will train our networks
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

        next_state_action_value = torch.zeros(len(batch), device=device)

        #For Double DQN, we use our policy network to get the action
        #with the best value for the next state and use this action to get
        #the next state action value for the target network
        best_next_state_actions = self.P_network(non_termination_next_states).argmax(1).unsqueeze(1)

        with torch.no_grad():
            next_state_action_value[non_terminating_mask] = self.T_network(non_termination_next_states).gather(1,best_next_state_actions).squeeze(1)

        # Calculate the expected state action values
        expected_state_action_values = (next_state_action_value * self.gamma) + rewards

        loss = self.loss_function.calculate(state_action_values.squeeze(1), expected_state_action_values)
        # loss =  criterion(state_action_value.unsqueeze(0).unsqueeze(1),expected_state_action_values.unsqueeze(0).unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()














