import numpy as np
import random
from matplotlib import pyplot as plt
from environment import  JungleEnv

class LearningBy:
    def __init__(self,learning_algorithm):
        self.learning_algorithm = learning_algorithm

    def learn(self, learning_parameters):
        #For now we are just using Q-Learning.
        state = learning_parameters["state"]
        action = learning_parameters["action"]
        reward = learning_parameters["reward"]
        q_matrix = learning_parameters["q_matrix"]
        return 0


class HikerAgent:
    def __init__(self,start_position, epsilon, alpha, gamma, policy_type, jungle: JungleEnv, learning_algorithm):
        self.environment = jungle
        self.initialise_q_matrix()
        self.current_position = start_position
        self.learning_by = LearningBy(learning_algorithm)
        self.available_moves = jungle.get_available_moves(self.current_position)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.policy_type = policy_type

    def move(self):
        can_move = False
        #Let's see if we can move
        if len(self.available_moves) > 0:
            best_action = self.get_best_action()
            #print("Best actions",best_action)

            #If there are no available actions, then do nothing
            if best_action is None:
                #print("No more available moves")
                can_move = False
            else:
                #print("Moving through jungle with following action",best_action)
                reward, new_position, available_moves, topography = jungle.move(self.current_position,best_action)
                #print("Reward", reward, "New Pos", new_position, "available_moves", available_moves, "topography", topography)

                #The next step is to update the q matrix with the reward
                q_value_old, q_max, q_value_new = self.update_q_matrix(reward,self.current_position, new_position, best_action)
                #print("q_value_old",q_value_old,"q_max",q_max,"q_value_new",q_value_new)
                #Update the agent's state and available moves
                self.current_position = new_position
                self.available_moves = available_moves

                if topography == self.environment.goal_state:
                    #We've reached the goal so there are no more moves:
                    #print("Reached goal state")
                    can_move = False
                else:
                    can_move = True
        else:
            #print("No more moves")
            can_move = False
        return can_move

    def initialise_q_matrix(self):
        self.q_matrix = np.full(self.environment.reward_matrix.shape, 0)

    def get_q_value(self,state,action_index):
        return self.q_matrix[self.environment.get_r_index((state[0],state[1])), action_index]

    def update_q_matrix(self, reward, state, new_state, action):
        #We will update the Q value for the given state and action
        #The state defines the position at which the action was taken
        #Get the index into the Q matrix
        shape = self.environment.reward_matrix.shape
        row, col = state[0],state[1]
        action_index = self.environment.all_actions[action]
        #Get the old q_value which is found in the q matrix for state,action
        q_value_old = self.get_q_value(state,action_index)
        #For the new_state, get the action that has the best estimated q value for the new_state
        new_state_index = self.environment.get_r_index((new_state[0],new_state[1]))
        q_max = max(self.q_matrix[new_state_index])
        #Update the old q value
        q_value_new = q_value_old + self.alpha * ( reward+(self.gamma*q_max) - q_value_old )

        self.q_matrix[self.environment.get_r_index((row,col)), action_index] = q_value_new
        return q_value_old, q_max, q_value_new

    def get_best_action(self):
        #Create a list of available moves from the curent state
        moves = np.array([x for x in self.available_moves.values()])

        #If there are no available moves then the best action is none
        if len(moves) == 0:
            best_action = None
        else:
            #Look into the q matrix and get the action from the current state with the largest q values
            r_index = self.environment.get_r_index(self.current_position)
            # Let's select the move with the highest expected q_max
            q_max = self.q_matrix[r_index, moves].max()
            # Let's get all the available actions that have this q_max. Important to note
            # that flatnonzero will return more indices into moves
            best_action_values = moves[np.flatnonzero(self.q_matrix[r_index, moves] == q_max)]
            best_actions = [i[0] for i in list(self.environment.all_actions.items()) if i[1] in best_action_values]
            #Let's look at our learning policy
            if self.policy_type == 'Greedy':
                #Randomly choose best action. If there is only one, the code below will select it
                #best_action = best_actions[random.randint(0,len(best_actions)-1)]
                best_action = np.random.choice(best_actions)
            else:
                #We are using Epsilon-greedy
                #ref: INM707 Lab 4
                if np.random.uniform() > self.epsilon:
                    #We randomly choose a move from the available moves
                    action_value = np.random.choice(moves)
                    best_action = [i[0] for i in list(self.environment.all_actions.items()) if i[1] == action_value][0]
                    # print("Selecting random action '{}' with current Q value {}".format(A[a], Q[s,a]))
                else:
                    #We choose the available move that has the highest estimated reward
                    best_action = np.random.choice(best_actions)
                    # print("Selecting greedy action '{}' with current Q value {}".format(A[a], Q[s,a]))

        return best_action

    def random_start(self):
        self.current_position = self.environment.get_start_position()
        self.available_moves = jungle.get_available_moves(self.current_position)

    def get_current_topography(self):
        return self.environment.get_topography(self.current_position)

    def show_path(self,position):
        #Convert the position to an index in the q_matrix
        index = self.environment.get_r_index(position)
        print("Start",position, self.environment.get_topography(position))
        #Given the index let's get the action with the best reward

        while 1:
            next_move = self.q_matrix[index].argmax()
            index = self.environment.get_new_index(index,next_move)
            position = self.environment.get_position(index)
            topography = self.environment.get_topography(position)
            print("Move to:",position,topography)
            if topography in ["S","E"]: break


jungle = JungleEnv(7, 7)
jungle.add_mountains([(1, 5), (2, 5), (3, 5)])
jungle.add_sinkholes([(1, 4), (3, 4), (6, 2), (7, 4)])
jungle.add_tigers([(4, 5)])
jungle.add_rivers([(4, 4)])
jungle.add_lakes([(5, 4)])
jungle.add_exits([(4, 7), [5, 7]])
jungle.fill_r_matrix()
epsilon = 0
alpha = 1.0
gamma = 1.0

hiker = HikerAgent((1,1),0.8,1.0,1.0,'Epsilon-Greedy',jungle,"Q-Learning")
print(jungle.jungle_floor)
#reward, new_position, available_moves, topography = jungle.move((4, 4), "West")

#We go through 1000 episodes
#ref: INM707 Lab4
metrics = {"E":[],"S":[],"F":[]}
episodes = 10000
timesteps = 5000

for episode in range(episodes):
    #For each episode we go through 500 timesteps
    hiker.random_start()
    last_timestep = 0
    for timestep in range(timesteps):
        if not hiker.move(): break
        last_timestep+=1

    #Let's see what state the hiker ended up in
    topography = hiker.get_current_topography()
    metrics[topography].append([episode,last_timestep])
    if episode == 5000: hiker.epsilon = 0.9

    #print('Episode {} finished. Q matrix values:\n{}'.format(episode, hiker.q_matrix.round(1)))
print('Final Q matrix: \n{}'.format(hiker.q_matrix.round(0)))

#Let's plot a graph of the number of timesteps per epoch to reach the exit
fig, ax = plt.subplots(3)
i = 0
for key in metrics.keys():
    vals = np.array(metrics[key])
    if len(vals) > 0:
        ax[i].plot(vals[:,[0]],vals[:,[1]])
        ax[i].set_title(key)
    i+=1

plt.show()
hiker.show_path([1,1])







