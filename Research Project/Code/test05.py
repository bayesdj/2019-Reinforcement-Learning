import numpy as np
import gym
import try05 as base
from matplotlib import pyplot as plt      

#%%
env = gym.make("Pendulum-v0")
seed = None
poly = 3
V = base.ValueFunctionWithTile(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 32,
                          tile_width = np.array([0.25,0.25,1]),
                          λ=0, α=2e-3                        
                          )

pi = base.PiApproximationWithGaussian(state_lo=env.observation_space.low,
                                     state_hi=env.observation_space.high,
                                     λ=0, α=2e-3, poly=3
                                     )

#rewards = base.actor_critic(π,V,env,α=1e-3,T=3000,seed=None)


#%%
ε = 1e-16
T = 3000
α = 0.0156
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
    s1,r,done,_ = env.step(np.array([4*a-2]))
    δ = r-rbar+V(s1)-V(s)
    rbar += α*δ
    V.update(s,δ)
    pi.update(s,a,δ)
    s = s1
    Rewards[t] = r

#%%
fig,ax = plt.subplots(1,1,figsize=(12,5))
ax.plot(Rewards)
