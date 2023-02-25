import gym
import numpy as np
import time
import random
import math
from IPython.display import clear_output

#%%
env = gym.make('MountainCarContinuous-v0')
#%%
A = env.action_space
S = env.observation_space



