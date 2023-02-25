import numpy as np
import gym
import try04 as base
from matplotlib import pyplot as plt
#%%
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
        

#%%
env = gym.make("Pendulum-v0")
seed = None
poly = 3
V = base.ValueFunctionWithTile(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 8,
                          tile_width = np.array([0.25,0.25,1]),
                          λ=0, α=2e-2                         
                          )

pi = base.PiApproximationWithGaussian(state_lo=env.observation_space.low,
                                     state_hi=env.observation_space.high,
                                     λ=0, α=2e-2, poly=2
                                     )

#%%
ε = 1e-16
T = 3000
α = 2e-2
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
