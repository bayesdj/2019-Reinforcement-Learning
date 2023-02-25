import numpy as np
import gym
import base
from matplotlib import pyplot as plt
#%%

env = gym.make("Pendulum-v0")
seed = 99
C = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1],[0,0,0],
     [1,2,3],[2,1,3],[3,3,1],[1,1,3]]
C = np.array(C)*3
V = base.ValueFunctionWithTile(env.observation_space.low,
                          env.observation_space.high,
                          num_tilings = 16,
                          tile_width = np.array([0.2,0.2,0.5]),
                          λ=0.7, α=2e-2                          
                          )

pi = base.PiApproximationWithGaussian(C,
                                     state_lo=env.observation_space.low,
                                     state_hi=env.observation_space.high,
                                     λ=0.2, α=(2e-2,2e-2)
                                     )



T=int(3000)
lr = 1e-5
Rewards = np.empty(shape=(2,T))


for i in (0,1):
    rbar = 0
    if seed is not None:
        env.seed(seed)
    s = env.reset()
    V.ix = V.feat(s)
    env.spec.max_episode_steps = T+1
    for t in range(T):
        a = pi(s)
    #    env.render()
        s1,r,done,_ = env.step(np.array([a]))  
        δ = V.update(a,r-rbar,s1)
        pi.update(s,a,δ)
        rbar += lr*δ
        s = s1
        Rewards[i,t] = r
#%%
fig,ax = plt.subplots(2,1,figsize=(12,6),sharex=True,sharey=True)
colors = ['r','b']
for i in (0,1):
    ax[i].plot(Rewards[i],color=colors[i])
ax[0].set_title('Successful Case')
ax[1].set_title('Unsuccessful Case')
fig.text(0.06, 0.5, 'Rewards', ha='center', va='center', rotation='vertical')
plt.xlabel('Step')
