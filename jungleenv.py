import random
from jungle import Jungle
import gymnasium as gym
from gymnasium import spaces

import numpy
import numpy as np


class JungleEnv(gym.Env):
    #Code inspired by ref: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
    def __init__(self, size, seed=45):
        self.size = size

        self.observation_space = spaces.Dict(
            {
                "jungle_position":spaces.Box(low=1,high=self.size,shape=(2,), dtype=np.int)
            }
        )
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.action_space = spaces.Discrete(4)
        self._jungle = Jungle(self.size,self.size)
        self._jungle.build_jungle(seed)
        self._jungle.reset(seed)

    def reset(self,start_position = None, seed = 45,options = None):
        #ref INM707 Lab 6
        super().reset(seed=seed)

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        #Create a new JungleEnv
        self._jungle = Jungle(rows=self.size, cols=self.size,vanishing_treasure=False,seed=self.seed)
        self._jungle.build_jungle(seed=self.seed)
        obs = self._jungle.reset(seed=self.seed)

        obs = {"jungle_position":np.array(obs)}

        return obs, {}

    def step(self, action):
        jaction = list(self._jungle.all_actions.keys())[action]
        reward, observation, available_moves, terminated, topography = self._jungle.move(jaction)
        #ref INM707 Lab 6
        return {"jungle_position":np.array(observation)}, reward, terminated, False, topography

    def render(self):
        self._jungle.show()


