import numpy as np
import gym
import try03 as base
from matplotlib import pyplot as plt
#%%

env = gym.make("Pendulum-v0")
seed = 99
C = [[5,0,0],[0,5,0],[0,0,5],[5,5,0],[5,0,5],[0,5,5],[1,1,4]]
C = np.array(C)
V = base.ValueFunctionWithTile(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 8,
                          tile_width = np.array([0.2,0.2,0.5]),
                          λ=0, α=1e-2                          
                          )

π = base.PiApproximationWithGaussian(C,
                                     state_lo=env.observation_space.low,
                                     state_hi=env.observation_space.high,
                                     λ=0.3, α=1e-2
                                     )

rewards = base.actor_critic(π,V,env,α=1e-2,T=10000,seed=seed)


#%%
fig,ax = plt.subplots(1,1,figsize=(12,5))
ax.plot(rewards)
