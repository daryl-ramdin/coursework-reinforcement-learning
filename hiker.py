import numpy as np
import random
from matplotlib import pyplot as plt
from jungle import  JungleEnv

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
    def __init__(self,start_position, epsilon, alpha, gamma, policy_type, jungle: JungleEnv, learning_algorithm, seed=45):
        self.environment = jungle
        self.initialise_q_matrix()
        self.current_position = start_position
        self.learning_by = LearningBy(learning_algorithm)
        self.available_moves = self.environment.get_available_moves(self.current_position)
        self.initial_epsilon = epsilon
        self.epsilon = self.initial_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.policy_type = policy_type
        self.cumulative_reward = 0
        self.seed = seed

    def move(self):
        can_move = False
        #Let's see if we can move
        if len(self.available_moves) > 0:
            next_action = self.get_next_action()
            #print("Next actions",next_action)

            #If there are no available actions, then do nothing
            if next_action is None:
                #print("No more available moves")
                can_move = False
            else:
                #print("Moving through jungle with following action",best_action)
                reward, new_position, available_moves, topography = self.environment.move(self.current_position,next_action)
                #print("Reward", reward, "New Pos", new_position, "available_moves", available_moves, "topography", topography)
                #Update the cumulative reward
                self.cumulative_reward+=reward
                #The next step is to update the q matrix with the reward
                q_value_old, q_max, q_value_new = self.update_q_matrix(reward,self.current_position, new_position, next_action)
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

    def get_next_action(self):
        #Create a list of available moves from the curent state
        avlbl_moves = np.array([x for x in self.available_moves.values()])

        next_action = None
        #If there are no available moves then the best action is none
        if len(avlbl_moves) == 0:
            next_action = None
        else:
            #Look into the q matrix and get the action from the current state with the largest q values
            r_index = self.environment.get_r_index(self.current_position)
            # Let's select the move with the highest expected q_max
            q_max = self.q_matrix[r_index, avlbl_moves].max()
            # Let's get all the available actions that have this q_max. Important to note
            # that flatnonzero will return more indices into moves
            best_action_values = avlbl_moves[np.flatnonzero(self.q_matrix[r_index, avlbl_moves] == q_max)]
            best_actions = [i[0] for i in list(self.environment.all_actions.items()) if i[1] in best_action_values]
            #Let's look at our learning policy
            if self.policy_type == 'Greedy':
                #Randomly choose best action. If there is only one, the code below will select it
                #best_action = best_actions[random.randint(0,len(best_actions)-1)]
                next_action = np.random.choice(best_actions)
            else:
                #We are using Epsilon-greedy
                #ref: INM707 Lab 4
                if np.random.uniform() > self.epsilon:
                    #We exploit. Choose the available move that has the highest estimated reward
                    next_action = np.random.choice(best_actions)
                    # print("Selecting greedy action")
                else:
                    # We explore. Randomly choose from the available moves
                    action_value = np.random.choice(avlbl_moves)
                    next_action = [i[0] for i in list(self.environment.all_actions.items()) if i[1] == action_value][0]
                    # print("Selecting random action")

        return next_action

    def random_start(self):
        self.cumulative_reward = 0
        #self.epsilon = self.initial_epsilon
        self.current_position = self.environment.get_start_position()
        self.available_moves = self.environment.get_available_moves(self.current_position)

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










