import numpy as np
import gym
import try07 as base
from matplotlib import pyplot as plt
import pandas as pd
#%%
env = gym.make("Pendulum-v0")

C = [[5,0,0],[0,5,0],[0,0,5],[5,5,0],[5,0,5],[0,5,5],[1,1,4]]
C = np.array(C)

LR = [1*10**i for i in range(-1,-7,-1)]
Results = []

N = 50
Seeds = np.random.randint(low=1,high=1000,size=N)
lr_rbar = 2e-5
lr_critic = 2e-2
T = 3000

def getMeanReward(actor,critic,rbar,seed):
    V = base.ValueFunctionWithTile(
        env.observation_space.low,
        env.observation_space.high,
        num_tilings = 16,
        tile_width = np.array([0.2,0.2,0.5]),
        λ=0.2, α=critic)
    pi = base.PiApproximationWithGaussian(C,
        state_lo=env.observation_space.low,
        state_hi=env.observation_space.high,
        λ=0.2, α=actor)
    rewards = base.actor_critic(pi,V,env,
                                α=rbar,T=T,seed=seed)[-int(T/2):].mean()
    return rewards.mean()

Results = []
for a1 in LR:
    for a2 in LR:
        α = (a1,a2)
        r = [getMeanReward(α,lr_critic,lr_rbar,int(Seeds[i])) for i in range(N)]
        row = [a1,a2,np.array(r).mean()]
        print(row)
        Results.append(row)
            
Results = pd.DataFrame(np.array(Results),columns=['α_μ','α_σ','mean'])
Results.sort_values(by='mean',ascending=False,inplace=True)



