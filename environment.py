import random

import numpy
import numpy as np


class JungleEnv:

    def __init__(self, rows, cols, rewards = None,goal_state=None):
        self.rows = rows
        self.cols = cols
        self.exit_reward = 500
        self.mountains = []
        self.tigers = []
        self.sinkholes = []
        self.rivers = []
        self.lakes = []
        self.exits = []
        self.all_actions = {"North":0,"South":1,"East":2,"West":3}
        if goal_state is None:
            self.goal_state = "E"
        else:
            self.goal_state = goal_state

        #Let's add the rewards. They're as follows:
        #F: Jungle floor that takes 1 day to cross and you lose 1 point
        #R: River that will cut the journey by 5 days so you gain 5 points
        #T: Tiger will attack and slow you down by 70 days as you recover so you lose 70 points
        #L: Lake that takes 2 days to cross so you lose 2 points
        #S: Sinkhole that ends the game so you lose 1000 points
        #E: Exit that gives you 500 points so you gain 500 points
        if rewards is None:
            self.rewards = {"F":-1,"R":5,"T":-70,"L":-2,"M":-5,"S":-1000,"E":500}
        else:
            self.rewards = rewards
        self.jungle_floor = np.full([rows,cols,],'F')
        #Our reward matrix consists of s rows and a columns where
        #s: rows*cols which is the number of states. A state corresponds to the position on the jungle floow
        #a: number of actions which in this case are 4
        self.reward_matrix = np.full([rows*cols, len(self.all_actions)],np.nan)

    def add_mountains(self,mountains: []):
        for mountain in mountains:
            self.mountains.append(mountain)
            self.jungle_floor[mountain[0]-1,mountain[1]-1] = "M"

    def add_tigers(self,tigers: []):
        for tiger in tigers:
            self.tigers.append(tiger)
            self.jungle_floor[tiger[0]-1,tiger[1]-1] = "T"

    def add_sinkholes(self,sinkholes: []):
        for sinkhole in sinkholes:
            self.sinkholes.append(sinkhole)
            self.jungle_floor[sinkhole[0]-1,sinkhole[1]-1] = "S"

    def add_rivers(self,rivers: []):
        for river in rivers:
            self.rivers.append(river)
            self.jungle_floor[river[0]-1, river[1]-1] = "R"

    def add_lakes(self,lakes: []):
        for lake in lakes:
            self.lakes.append(lake)
            self.jungle_floor[lake[0]-1, lake[1]-1] = "L"

    def add_exits(self,exits: []):
        for exit in exits:
            self.lakes.append(exit)
            self.jungle_floor[exit[0]-1, exit[1]-1] = "E"

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

    def move(self,from_position: (),in_direction: str):
        #Return the reward if you move in the given direction from the position along
        #with the list of available moves
        new_position = ()

        if in_direction == "North":
            new_position = (from_position[0]-1,from_position[1])
        elif in_direction == "South":
            new_position = (from_position[0]+1, from_position[1])
        elif in_direction == "East":
            new_position = (from_position[0], from_position[1]+1)
        else:
            new_position = (from_position[0],from_position[1]-1)

        reward = self.get_reward(new_position)

        available_moves = self.get_available_moves(new_position)

        topography = self.get_topography(new_position)

        return reward, new_position, available_moves, topography

    def get_available_moves(self,position):
        #Get the list of available moves from the position
        moves = self.all_actions.copy()
        forbidden_moves = []

        #If we're in a sinkhole then we cannot move
        if self.jungle_floor[position[0]-1,position[1]-1]=="S":
            moves.clear()
        else:
            #Check if on a boundary
            if(position[1] == 1):
                #We're at the west border so cannot move west
                moves.pop("West")
            if(position[1] == self.cols):
                #We're at the eastern border so cannot move east
                moves.pop("East")
            if(position[0] == 1):
                #We're at the northern border so cannot move north
                moves.pop("North")
            if(position[0] == self.rows):
                #We're at the southern border so cannot move south
                moves.pop("South")

            #If we're on a mountain we cannot move to any adjacent cell that contains another mountain
            if self.jungle_floor[position[0] - 1, position[1] - 1] == "M":
                #We're on a mountain. If any of the available moves leads to a mountain, remove it
                for move in moves.keys():
                    if move =="North":
                        #Check the north adjacent
                        if self.jungle_floor[position[0] - 1 - 1, position[1] - 1] == "M":
                            forbidden_moves.append("North")
                    elif move =="South":
                        # Check the south adjacent
                        if self.jungle_floor[position[0] - 1+1, position[1] - 1] == "M":
                            forbidden_moves.append("South")
                    elif move =="East":
                        # Check the east adjacent
                        if self.jungle_floor[position[0] - 1, position[1] - 1+1] == "M":
                            forbidden_moves.append("East")
                    else:
                        # Check the west adjacent
                        if self.jungle_floor[position[0] - 1, position[1] - 1-1] == "M":
                            forbidden_moves.append("West")

            for move in forbidden_moves:
                moves.pop(move)

        return moves

    def get_start_position(self):
        #Randomly choose a start position
        row = random.randint(1,self.rows)
        col = random.randint(1,self.cols)
        return [row,col]

    def get_new_index(self, index, move):
        new_index = 0
        if move == 0:
            new_index = index-self.rows
        elif move ==1:
            new_index = index+self.rows
        elif move == 2:
            new_index = index+1
        else:
            new_index = index-1
        return new_index

    def get_position(self,index):
        #The position is given by index/rows,remainder
        return [(index//self.rows)+1,(index%self.rows)+1]

