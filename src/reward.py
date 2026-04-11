import numpy as np

class Reward:
    '''Implements a reward function.'''
    def __init__(self, reward_params):
        self.reward_params = reward_params
        self.reward = self._create_reward_function()

    def _create_reward_function(self):
        '''From the params, return the reward function.'''
        if self.reward_params["name"] == "ucb":
            return lambda mean, variance: mean + np.sqrt(self.reward_params["alpha"]) * variance
        elif self.reward_params["name"] == "mvi":  # optional extension exercise
            return lambda mean, variance: 1.0
        else:
            return lambda mean, variance: None

    def info(self):
        """Returns information to save about the reward function."""
        return {
            "Reward Params": self.reward_params,
        }
