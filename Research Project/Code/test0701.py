import numpy as np
import gym
import try07 as base
from matplotlib import pyplot as plt
import pandas as pd
#%%
env = gym.make("Pendulum-v0")

C = [[5,0,0],[0,5,0],[0,0,5],[5,5,0],[5,0,5],[0,5,5],[1,1,4]]
C = np.array(C)

Lam = [np.round(i*0.1,1) for i in range(8,-1,-1)]
Results = []

N = 50
Seeds = np.random.randint(low=1,high=1000,size=N)
lr_rbar = 2e-5
lr_critic = 2e-2
lr_actor = (2e-2,2e-2)
T = 3000

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
    rewards = base.actor_critic(pi,V,env,
                                α=rbar,T=T,seed=seed)[-int(T/2):].mean()
    return rewards.mean()

Results = []
for x1 in Lam:
    for x2 in Lam:
        r = [getMeanReward(lr_actor,lr_critic,lr_rbar,x1,x2,int(Seeds[i])) for i in range(N)]
        row = [x1,x2,np.array(r).mean()]
        print(row)
        Results.append(row)
            
Results = pd.DataFrame(np.array(Results),columns=['λπ','λv','mean'])
Results.sort_values(by='mean',ascending=False,inplace=True)



