import numpy as np
import gym
import try06 as base
from matplotlib import pyplot as plt      
import math
#%%
env = gym.make("Pendulum-v0")
seed = 99
C = [[5,0,0],[0,5,0],[0,0,5],[5,5,0],[5,0,5],[0,5,5],[1,1,4]]
C = np.array(C)

V = base.ValueFunctionWithTile(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 16,
                          tile_width = np.array([0.2,0.2,0.5]),
                          λ=0.2, α=1e-2                       
                          )

pi = base.PiApproximationWithGaussian(C,
                                     state_lo=env.observation_space.low,
                                     state_hi=env.observation_space.high,
                                     λ=0.2, α=1e-2
                                     )

#rewards = base.actor_critic(π,V,env,α=1e-3,T=3000,seed=None)


#%%
ε = 1e-10
T = int(1e4)
lr = 1e-5
rbar = 0
if seed is not None:
    env.seed(seed)
s = env.reset()
env.spec.max_episode_steps = T+1
Rewards = np.empty(T)
for t in range(T):
    a = pi(s)
#    env.render()
    if a == 1.:
        a-=ε
    elif a == 0.:
        a+=ε
#    env.render()
    s1,r,done,_ = env.step(np.array([4*a-2]))
    δ = r-rbar+V(s1)-V(s)
    rbar += lr*δ
    V.update(s,δ)
    pi.update(s,a,δ)
    s = s1
    Rewards[t] = r

#%%
fig,ax = plt.subplots(1,1,figsize=(12,5))
ax.plot(Rewards)
