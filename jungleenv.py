import random
from jungle import Jungle
import gymnasium as gym
from gymnasium import spaces

import numpy
import numpy as np


class JungleEnv(gym.Env):

    def __init__(self, config):
        self.size = config["size"]

        self.observation_space = spaces.Dict(
            {
                "jungle_position":spaces.Box(low=1,high=self.size,shape=(2,), dtype=np.int)
            }
        )
        self.action_space = spaces.Discrete(4)
        self._jungle = Jungle(self.size,self.size)
        self._jungle.build_jungle()
        self._jungle.reset()

    def reset(self,start_position = None,seed = None,options = None):
        #ref INM707 Lab 6
        super().reset(seed=seed)

        #Create a new JungleEnv
        self._jungle = Jungle(rows=self.size, cols=self.size,vanishing_treasure=False)
        self._jungle.build_jungle()
        obs = self._jungle.reset()

        obs = {"jungle_position":np.array(obs)}
        #self.render()
        return obs, {}

    def step(self, action):
        jaction = list(self._jungle.all_actions.keys())[action]
        reward, observation, available_moves, terminated, topography = self._jungle.move(jaction)
        #ref INM707 Lab 6
        #print("action", jaction, "obs",observation,"reward",reward,"termindated",terminated)
        return {"jungle_position":np.array(observation)}, reward, terminated, False, topography

    def render(self):
        self._jungle.show()
