import numpy as np
import gym
import try10 as base
from matplotlib import pyplot as plt
#%%

env = gym.make("Pendulum-v0")
seed = 99

V = base.ValueFunctionWithTile(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 16,
                          tile_width = np.array([0.25,0.25,0.5]),
                          λ=0.1, α=2e-1                         
                          )

pi = base.PiApproximationWithGaussian(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 16,
                          tile_width = np.array([0.25,0.25,0.5]),
                          λ=0.1, α=np.array((2e-2,2e-2))
                          )


T=int(3000)
lr = 1e-5
rbar = 0
if seed is not None:
    env.seed(seed)
s = env.reset()
V.ix = V.feat(s)
env.spec.max_episode_steps = T+1
Rewards = np.empty(T)
for t in range(T):
    a = pi(s)
#    env.render()
    s1,r,done,_ = env.step(np.array([a]))  
    δ = V.update(a,r-rbar,s1)
    pi.update(s,a,δ)
    rbar += lr*δ
    s = s1
    Rewards[t] = r
#%%
fig,ax = plt.subplots(1,1,figsize=(12,5))
ax.plot(Rewards)
