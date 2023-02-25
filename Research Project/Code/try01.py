import gym
import numpy as np
import time
import random
import math
from IPython.display import clear_output

#%%
env = gym.make('FrozenLake-v0')

nA = env.action_space.n
nS = env.observation_space.n

qTable = np.zeros((nS,nA))

nEpisode = int(1e4)
maxSteps = 100

a = 0.1
g = 0.99

rExplore = 1
maxExplore = 1
minExplore = 0.01
rangeExplore = maxExplore - minExplore
decayExplore = 0.001

Rewards = []

for episode in range(nEpisode):
    state = env.reset()
    done = False    
    rewards = 0
    
    for i in range(maxSteps):        
        threshold = random.uniform(0,1)
        if threshold > rExplore:
            action = np.argmax(qTable[state,:])
        else:
            action = env.action_space.sample()
    
        state1, reward, done, info = env.step(action)
        
        qTable[state,action] = qTable[state,action] * (1-a) + a * \
            (reward + g*np.max(qTable[state1,:]))
            
        state = state1
        rewards += reward
        
        if done == True: break

    rExplore = minExplore + rangeExplore*math.exp(-decayExplore*episode)
    Rewards.append(rewards)

    #env.render()
Rewards = np.array(Rewards)
#%%
rewards1000 = np.split(Rewards,nEpisode/1000)
count = 1000
for r in rewards1000:
    print(f'count: {sum(r/1000)}')
    count += 1000
    
print(qTable)
#%%
for episode in range(3):
    state = env.reset()
    done = False    
    print(f'************EPISODE {episode+1}******************\n\n\n')
    time.sleep(1)
    
    for i in range(maxSteps):        
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        action = np.argmax(qTable[state,:])
    
        state1, reward, done, info = env.step(action)        
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print('****Reached the goal****')
                time.sleep(3)
            else:
                print('****Fell through the hole****')
                time.sleep(3)            
            clear_output(wait=True)
            break
        state = state1

env.close()