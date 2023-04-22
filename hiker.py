import numpy as np
import random
from matplotlib import pyplot as plt
from jungle import  Jungle

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
    def __init__(self, jungle: Jungle, sarsa=False, seed=45):
        self.environment = jungle
        self.initialise_q_matrix()
        #ref: https://www.geeksforgeeks.org/assign-function-to-a-variable-in-python/
        self.sarsa = sarsa
        if self.sarsa:
            self.policy_update = self.sarsa_learn
        else:
            self.policy_update = self.q_learn
        self.epsilon = 0.9
        self.epsilon_decay = 1e-03
        self.alpha = 0.9
        self.gamma = 0.9
        self.cumulative_reward = 0
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.treasure_counter = 0
        self.current_position, self.available_moves = self.reset_episode()

    def reset_episode(self):
        self.cumulative_reward = 0
        self.treasure_counter = 0
        self.current_position = self.environment.reset(self.seed)
        self.available_moves = self.environment.get_available_moves(self.current_position)
        return self.current_position, self.available_moves

    def train(self,**kwargs):
        self.epsilon = kwargs["epsilon"]
        self.epsilon_decay = kwargs["epsilon_decay"]
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]
        self.seed = kwargs["seed"]
        random.seed(self.seed)
        timesteps = kwargs["timesteps"]
        episodes = kwargs["episodes"]
        experiment_id = kwargs["experiment_id"]
        episode_results = []
        self.initialise_q_matrix()

        for episode in range(episodes):
            # For every episode increment the seed. This
            # allows for the results to be reproduced and still make
            # each episode different from the other

            self.seed += 10
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.reset_episode()
            steps_taken = 0
            action = None

            #Select the first action
            action = self.select_action(self.current_position,self.available_moves)

            # Run through the timesteps until we reach the end  our timestep limit
            for timestep in range(timesteps):

                #Move based on the action. This will update the current position
                can_move, next_action = self.move(action)

                #If we cannot move, then the epsiode ends
                if not can_move: break

                #If this is sarsa, then next action becomes the action
                if self.sarsa:
                    action = next_action
                else:
                    # This is Q learning so select an action from the current position.
                    # This is usually at the start of the timestep. To cater for that
                    # it is called before the first timestep. As that has been put in place
                    # calling it at the end of the timestep achieves the same purpose
                    action = self.select_action(self.current_position, self.available_moves)

                steps_taken += 1
                self.epsilon = self.epsilon * self.epsilon_decay

            # Let's see what state the hiker ended up in
            topography = self.get_current_topography()

            episode_results.append({"experiment_id": experiment_id, "parameters": kwargs, "episode": episode,
                                    "episode_reward": self.cumulative_reward, "steps_taken": steps_taken,
                                    "topography": topography, "treasure_counter":self.treasure_counter})

        return episode_results

    def move(self,action):
        can_move = False

        next_action = None
        #If there are available actions, then move
        if action is not None:
            #Move
            reward, new_position, available_moves, terminated, info = self.environment.move(action)

            #If the agent collected the treasure, increase the treasure counter
            if info["topography"]=="$":
                self.treasure_counter += 1

            #If new_position is not None then it means we can move
            if new_position is not None:
                #Update the cumulative reward
                self.cumulative_reward+=reward

                #The next step is to update the q matrix with the reward.
                # If sarsa then we get set our action to the result of the policy_update
                # which is the action for the next state
                next_action = self.policy_update(reward,self.current_position, new_position, action, available_moves)

                #Update the agent's state and available moves
                self.current_position = new_position
                self.available_moves = available_moves

                #Check to see if we can move. If the terminated state is True, then we cannot move
                can_move = not terminated

        return can_move, next_action

    def initialise_q_matrix(self):
        self.q_matrix = np.full(self.environment.reward_shape, 0)

    def get_q_value(self,state,action_index):
        return self.q_matrix[self.environment.get_r_index((state[0],state[1])), action_index]

    def q_learn(self, reward, state, next_state, action, available_moves):
        #We will update the Q value for the given state and action
        #For Q Learning we do not use the available moves

        #The state defines the position at which the action was taken
        #Get the index into the Q matrix
        shape = self.environment.reward_shape
        row, col = state[0],state[1]
        action_index = self.environment.all_actions[action]

        #Get the old q_value which is found in the q matrix for state,action
        q_value_old = self.get_q_value(state,action_index)

        #For the new_state, get the action that has the best estimated q value for the new_state
        new_state_index = self.environment.get_r_index((next_state[0],next_state[1]))
        q_max = max(self.q_matrix[new_state_index])

        #Update the old q value
        q_value_new = q_value_old + self.alpha * ( reward+(self.gamma*q_max) - q_value_old)

        self.q_matrix[self.environment.get_r_index((row,col)), action_index] = q_value_new
        return None

    def sarsa_learn(self, reward, state, next_state, action, available_moves):
        # ref: https://towardsdatascience.com/q-learning-and-sasar-with-python-3775f86bd178
        # We will update the Q value for the given state and action
        # The state defines the position at which the action was taken
        # Get the index into the Q matrix
        shape = self.environment.reward_shape
        row, col = state[0], state[1]
        action_index = self.environment.all_actions[action]

        # Get the old q_value which is found in the q matrix for state,action
        q_value_old = self.get_q_value(state, action_index)

        #For SARSA, we get the Q value for taking an action in the next state
        next_state_action = self.select_action(next_state,available_moves)
        next_state_action_index = self.environment.all_actions[next_state_action]
        q_value_next = self.get_q_value(next_state, next_state_action_index)

        # Update the old q value
        q_value_new = q_value_old + self.alpha * (reward + (self.gamma * q_value_next) - q_value_old)

        self.q_matrix[self.environment.get_r_index((row, col)), action_index] = q_value_new
        return next_state_action

    def select_action(self,position, available_moves):
        #Create a list of available moves from the curent state
        avlbl_moves = np.array([x for x in available_moves.values()])

        next_action = None
        #If there are no available moves then the best action is none
        if len(avlbl_moves) == 0:
            next_action = None
        else:
            #Look into the q matrix and get the action from the current state with the largest q values
            r_index = self.environment.get_r_index(position)

            # Let's select the move with the highest expected q_max
            q_max = self.q_matrix[r_index, avlbl_moves].max()

            # Let's get all the available actions that have this q_max. Important to note
            # that flatnonzero will return more indices into moves
            best_action_indices = avlbl_moves[np.flatnonzero(self.q_matrix[r_index, avlbl_moves] == q_max)]
            best_actions = [i[0] for i in list(self.environment.all_actions.items()) if i[1] in best_action_indices]

            #We are using Epsilon-greedy
            #If epsilon=0 then Greedy
            #If epsilon=1 then Random
            #ref: INM707 Lab 4
            if np.random.uniform() > self.epsilon:
                #We exploit. Choose the available move that has the highest estimated reward.
                #In case there are multiple moves that share the same maximum q value, we randomly
                #choose one
                next_action = np.random.choice(best_actions)
            else:
                # We explore. Randomly choose from the available moves
                action_index = np.random.choice(avlbl_moves)
                next_action = [i[0] for i in list(self.environment.all_actions.items()) if i[1] == action_index][0]

        return next_action

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
            position = self.environment.index_to_position(index)
            topography = self.environment.get_topography(position)
            print("Move to:",position,topography)
            if topography in ["S","E"]: break







