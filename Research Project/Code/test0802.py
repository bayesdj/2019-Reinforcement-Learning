import numpy as np
import gym
import try08 as base
import pandas as pd
from matplotlib import pyplot as plt
#%%
env = gym.make("Pendulum-v0")

C = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1],[0,0,0],
     [1,2,3],[2,1,3],[3,3,1],[1,1,3]]
C = np.array(C)*3

Lam = [np.round(i*0.1,1) for i in range(7,-1,-1)]
Results = []

N = 100
Seeds = np.random.randint(low=1,high=int(1e5),size=N)
lr_rbar = 2e-5
lr_critic = 2e-2
lr_actor = (2e-2,2e-2)
T = 10000

def getMeanReward(actor,critic,rbar,λπ,λv,seed):
    V = base.ValueFunctionWithTile(
        env.observation_space.low,
        env.observation_space.high,
        num_tilings = 16,
        tile_width = np.array([0.2,0.2,0.5]),
        λ=λv, α=critic)
    pi = base.PiApproximationWithGaussian(C,
        state_lo=env.observation_space.low,
        state_hi=env.observation_space.high,
        λ=λπ, α=actor)
    rewards = base.actor_critic(pi,V,env,α=rbar,T=T,seed=seed)
    return rewards

#%%
r = []
for i in range(N):
    r.append(getMeanReward(lr_actor,lr_critic,lr_rbar,λπ=0.2,λv=0.7,seed=None))

r = np.array(r)
            
#%%
win = 50
df = pd.DataFrame(r).T
df1 = df.rolling(win,min_periods=win).mean().dropna().values
mu = df1.mean(1)
#%%
fig,ax = plt.subplots(1,1,figsize=(12,5))
ax.plot(mu)
ax.set_xlabel('steps')
ax.set_ylabel('moving average rewards')
ax.axhline(y=0,color='r')
#%%

#see = r.sum(1)
#
#nbins=20
#
#fig,ax = plt.subplots(1,1,figsize=(12,6),sharex=True,sharey=True)
#colors = ['r','b']
#h = ax.hist(ddpg,bins=nbins,density=True,color='r')
#h = ax.hist(see,bins=nbins,density=True,color='b',histtype='step')
#ax.set_xlabel('Total Rewards')
#ax.set_ylabel('Density')
#ax.legend(["Author's Design",'DDPG'])