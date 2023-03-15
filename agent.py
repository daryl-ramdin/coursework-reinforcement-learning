import numpy as np

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
        best_action = self.get_best_action()
        #If more than 1 action is returned, we randomly choose one
        if len(best_action) > 1:
            best_action = "West"
        else:
            best_action = "West"

        reward, new_position, available_moves, topography = jungle.move(self.current_position,best_action)

        #The next step is to update the q matrix with the reward
        self.update_q_matrix(reward,self.current_position, new_position, best_action)

        #Update the agent's state and available moves
        self.current_position = new_position
        self.available_moves = available_moves
        return

    def initialise_q_matrix(self):
        self.q_matrix = np.full(self.environment.reward_matrix.shape, 0)

    def update_q_matrix(self, reward, state, new_state, action):
        #We will update the Q value for ther given state and action
        #The state defines the position at which the action was taken
        #Get the index into the Q matrix
        shape = self.environment.reward_matrix.shape
        row, col = state[0],state[1]
        action = self.environment.all_actions[action]
        #Get the old q_value
        q_value_old = 0
        #For the new_state, get the action that has the best estimated q value
        q_max = 1
        #Update the old q value
        q_value_old = q_value_old + self.alpha * ( reward+(self.gamma*q_max) - q_value_old )

        self.q_matrix[self.environment.get_r_index((row,col)), action] = q_value_old
        return

    def get_best_action(self):
        #Create a list of available moves column indices
        moves = [x for x in self.available_moves.values()]
        #Look into the q matrix and get the action with the largest q values
        best_action = self.q_matrix[:,moves].max(1)
        return best_action

jungle = JungleEnv(7, 7)
jungle.add_mountains([(1, 5), (2, 5), (3, 5)])
jungle.add_sinkholes([(1, 4), (3, 4), (6, 2), (7, 4)])
jungle.add_tigers([(4, 5)])
jungle.add_rivers([(4, 4)])
jungle.add_lakes([(5, 4)])
jungle.add_exits([(4, 7), [5, 7]])
jungle.fill_r_matrix()
hiker = HikerAgent((1,1),0.9,1,1,'Greedy',jungle)
print(jungle.jungle_floor)
print(hiker.q_matrix)
#print(jungle.jungle_floor)
#print(jungle.reward_matrix)
#print(jungle.move((4, 4), "West"))
#print(jungle.move((4, 4), "East"))
# print(jungle.get_available_moves((3,4)))


