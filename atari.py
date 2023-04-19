from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray import air, tune
import ray
import shimmy

from ray.tune.logger import pretty_print



# # #ray.shutdown()
# # #ray.init()

# #Let's configure our algorithm


config = DQNConfig()
config.double_q = True
config = config.environment("Riverraid-ram-v4")
config.num_gpus = 1
config.replay_buffer_config["prioritized_replay"] = True #tune.grid_search([False,True])


#algo = config.build()
tuner = tune.Tuner("DQN",param_space=config.to_dict())
results = tuner.fit()



# result = {}
# logger = []

# for i in range(10):
#   results = algo.train()
#   logger.append([i,results["episode_reward_mean"],0])
#   print("Epoch",i,"Episode Mean Reward",results["episode_reward_mean"])

print("done")

print("Done")
