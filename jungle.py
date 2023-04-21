import random

import numpy
import numpy as np


class Jungle:

    def __init__(self, rows, cols, revisits=True, vanishing_treasure=False, seed=45):
        self.rows = rows
        self.cols = cols
        self.revisits = revisits
        self.vanishing_treasure = vanishing_treasure
        self.all_actions = {"North":0,"South":1,"East":2,"West":3}
        self.penalty = -5   #The penalty for making an illegal move

        #Ref INM707 Lab 6
        #self.blocked_locations = []
        #self.agent_position = None

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        #Let's add the rewards. They're as follows:
        #F: Jungle floor that takes 1 day to cross and you lose 1 point
        #R: River that will cut the journey by 5 days so you gain 5 points
        #B: Bear will attack and slow you down by 70 days as you recover so you lose 70 points
        #L: Lake that takes 2 days to cross so you lose 2 points
        #M: Mountains take 5 days to cross
        #S: Sinkhole that ends the game so you lose 1000 points
        #T: Tigers can attack and slow the hiker by 10 days
        #E: Exit that gives you 1000 points
        #self.topographies = {"R":self.rivers,"B":self.bears,"L":self.lakes,"M":self.mountains,"S":self.sinkholes,"T":self.treasure,"E":self.exits}
        #self.topographies = ("R", "B", "L", "M", "S", "T", "E")

        self.rewards = {"_":-1,"R":-5,"B":-10,"L":-2,"M":-5,"S":-100,"T":-10,"$":500,"E":5000}
        self.termination_topography = ["S","E"]

        self.jungle_floor = np.full([self.rows,self.cols,],'_')
        #Our reward matrix consists of s rows and a columns where
        #s: rows*cols which is the number of states. A state corresponds to the position on the jungle floow
        #a: number of actions which in this case are 4
        self.reward_matrix = np.full([self.rows*self.cols, len(self.all_actions)],np.nan)

        #These variables change state as the agent moves

        #Agent position: randomly assign a start position
        self.agent_position = [random.randint(1, self.rows), random.randint(1, self.cols)]
        #Blocked Locations: There are no blocked locatoin as yet
        self.blocked_locations = []

    def reset(self,seed):
        #Initialise the variables that change state
        random.seed(seed)
        np.random.seed(seed)
        # Randomly choose a start position that is not in a termination state
        self.agent_position = [random.randint(1, self.rows), random.randint(1, self.cols)]

        # Clear the blocked locations
        self.blocked_locations = []

        # Return the agent's position
        return self.agent_position

    def build_jungle(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        #Let's create a random jungle.
        position_indices = list(range(0,(self.rows*self.cols)))
        available_positions = [self.index_to_position(i) for i in position_indices]

        #Initialise the jungle floor with '_'
        self.jungle_floor = np.full([self.rows, self.cols, ], '_')

        #Let's randomly add topographies
        for key in self.rewards:
            if key != '_':
                position = random.sample(available_positions,1)[0]
                self.jungle_floor[position[0] - 1, position[1] - 1] = key
                available_positions.remove(position)

    def add_topography(self,topographies: []):
        for topography in topographies:
            self.jungle_floor[topography[0] - 1, topography[1] - 1] = topography[2]

    def block_revisits(self,blocked_locations: []):
        for location in blocked_locations:
            self.blocked_locations.append(location)
            self.jungle_floor[location[0]-1, location[1]-1] = "X"

    def fill_r_matrix(self):
        #We build the r matrix based on the jungle floor
        #Go through the rows
        #Go through each column in the row and set the available moves and reward

        for row in range(1,self.rows+1):
            for col in range(1,self.cols+1):
                #Get the list of available moves and get the reward based on that move
                #print((row,col))
                moves = self.get_available_moves((row,col))
                for move in moves:
                    if move=="North":
                        self.reward_matrix[self.get_r_index((row,col)),self.all_actions["North"]] = self.get_reward((row-1,col))
                    elif move=="South":
                        self.reward_matrix[self.get_r_index((row,col)),self.all_actions["South"]] = self.get_reward((row+1,col))
                    elif move == "East":
                        self.reward_matrix[self.get_r_index((row,col)),self.all_actions["East"]] = self.get_reward((row,col+1))
                    else:
                        self.reward_matrix[self.get_r_index((row,col)), self.all_actions["West"]] = self.get_reward((row,col-1))
        return

    def get_reward(self,position):
        topography = self.jungle_floor[position[0]-1,position[1]-1]
        return self.rewards[topography]

    def get_topography(self,position):
        topography = self.jungle_floor[position[0]-1,position[1]-1]
        return topography

    def get_r_index(self,position):
        index = ((position[0]-1)*self.cols) + position[1]-1
        return index

    def move(self,in_direction: str):
        #Return the reward if you move in the given direction from the position along
        #with the list of available moves

        from_position = self.agent_position
        reward = 0
        new_position = from_position
        terminated = False
        topography = self.get_topography(from_position)

        #Get the list of available moves from this location
        available_moves = self.get_available_moves(from_position)

        #If there are none it means the hiker has reached a terminating state
        if len(available_moves)> 0:

            #If the move is in the list of available moves, then continue
            #otherwise terminate and return nothing
            if in_direction in available_moves:
                if in_direction == "North":
                    new_position = (from_position[0]-1,from_position[1])
                elif in_direction == "South":
                    new_position = (from_position[0]+1, from_position[1])
                elif in_direction == "East":
                    new_position = (from_position[0], from_position[1]+1)
                else:
                    new_position = (from_position[0],from_position[1]-1)

                reward = self.get_reward(new_position)

                topography = self.get_topography(new_position)

                #If the agent is at the exit or sink hole then set the terminated flag to True
                #as the agent can no longer move. ref: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
                if topography in self.termination_topography: terminated = True

                #If vanishing_treasure is true and we are on a position with treasure,
                #then this position cannot be revisited
                if self.vanishing_treasure and topography=="$":
                    self.block_revisits([new_position])

                #If revisits is false, then the current position cannot be revisited
                if self.revisits==False:
                    self.block_revisits([new_position])

                #Set the new position of the agent
                self.agent_position = new_position
            else:
                # You cannot make this move so you are penalised
                # ref: INM707 Lab 8
                reward = self.penalty
        else:
            terminated = True

        info = {"topography": topography}
        return reward, new_position, available_moves, terminated, info

    def get_available_moves(self,from_position):
        #Get the list of available moves from the position
        moves = self.all_actions.copy()

        #If we're in a sinkhole then we cannot move
        if self.jungle_floor[from_position[0]-1,from_position[1]-1]=="S":
            moves.clear()
        else:
            #If on a boundary, the agent cannot move in certain directions
            if(from_position[1] == 1):
                #Agent at the west border so cannot move west
                moves.pop("West")
            if(from_position[1] == self.cols):
                #Agent at the eastern border so cannot move east
                moves.pop("East")
            if(from_position[0] == 1):
                #Agent at the northern border so cannot move north
                moves.pop("North")
            if(from_position[0] == self.rows):
                #Agent at the southern border so cannot move south
                moves.pop("South")

            #Finally, checking the remainding moves and ensure that
            #they are not in one of the locations that has been blocked for revisits.
            remainding_moves = moves.copy()
            for move in remainding_moves.keys():
                if move == "North" and [from_position[0] - 1, from_position[1]] in self.blocked_locations:
                    moves.pop(move)
                elif move == "South" and [from_position[0] + 1, from_position[1]] in self.blocked_locations:
                    moves.pop(move)
                elif move == "East" and [from_position[0], from_position[1] + 1] in self.blocked_locations:
                    moves.pop(move)
                elif move=="West" and [from_position[0], from_position[1] - 1] in self.blocked_locations:
                    moves.pop(move)

        return moves

    def get_next_position(self, direction, from_position):
        next_position = None

        if direction == "North":
            next_position = (from_position[0] - 1, from_position[1])
        elif direction == "South":
            next_position = (from_position[0] + 1, from_position[1])
        elif direction == "East":
            next_position = (from_position[0], from_position[1] + 1)
        else:
            next_position = (from_position[0], from_position[1] - 1)
        return next_position

    def get_new_index(self, index, move):
        new_index = 0
        if move == 0: #North
            new_index = index-self.cols
        elif move ==1: #South
            new_index = index+self.cols
        elif move == 2: #East
            new_index = index+1
        else: #West
            new_index = index-1
        return new_index

    def index_to_position(self,index):
        #The position is given by index/rows,remainder
        return [(index//self.cols)+1,(index%self.cols)+1]

    def show(self):
        #Create a copy of the jungle and show where the agent is currently located
        current_view = self.jungle_floor.copy()
        current_view[self.agent_position[0]-1,self.agent_position[1]-1] = "*"
        print(np.array2string(current_view,formatter={'str_kind': lambda x: x}).replace("[","").replace("]",""),"\n")

