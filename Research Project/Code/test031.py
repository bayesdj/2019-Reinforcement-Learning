import numpy as np
import gym
import try03 as base
import pandas as pd
#%%

env = gym.make("Pendulum-v0")

C = [[5,0,0],[0,5,0],[0,0,5],[5,5,0],[5,0,5],[0,5,5],[1,1,4]]
C = np.array(C)

LR = [2*10**i for i in range(-2,-7,-1)]
Results = []

N = 50
Seeds = np.random.randint(low=1,high=1000,size=N)

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
for lr_actor in LR:
    for lr_critic in LR:
        for lr_rbar in LR:          
            r = [getMeanReward(lr_actor,lr_critic,lr_rbar,int(Seeds[i])) for i in range(N)]
            Results.append([lr_actor,lr_critic,lr_rbar,np.array(r).mean()])
            
Results = pd.DataFrame(np.array(Results),columns=['actor','critic','rbar','mean'])
Results.sort_values(by='mean',ascending=False,inplace=True)



