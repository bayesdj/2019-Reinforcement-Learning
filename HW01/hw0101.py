import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import timeit
from numba import njit
#%%
def plot(sample_average,constant):
    fig,axes = plt.subplots(2,1)
    axes[1].set_ylim([0.,1.])
    axes[0].plot(sample_average['average_rs'])
    axes[1].plot(sample_average['average_best_action_taken'])
    axes[0].plot(constant['average_rs'])
    axes[1].plot(constant['average_best_action_taken'])

@njit()
def getRew(Qstar,ε):
    R = np.empty((T,S))
    percent = np.full((T,S),False,dtype=np.bool_)
    for s in range(S):        
        Q = np.zeros(nA)
        Qcount = Q.copy()   
        Explore = np.random.rand(T) < ε
        best = Qstar[s,:].argmax()
        norm = np.random.randn(T)
        for t in range(T):
            At = np.random.choice(actions) if Explore[t] else Q.argmax()
            r = norm[t] + Qstar[s,At]
            n = Qcount[At]+1
            Q[At] = (Q[At]*(n-1) + r)/n
            Qcount[At] = n
            R[t,s] = r   
            percent[t,s] = (At==best)
    return R.sum(1)/S, percent.sum(1)/S
    
    
#%%   
rv = norm(loc=0,scale=0.01)
nA = 10
actions = np.arange(nA)
T = 1000
S = 2000
Qstar = norm.rvs(size=(S,nA))

ε = [0,0.01,0.1]
Ans = [getRew(Qstar,e) for e in ε]

#%%
R = np.array([Ans[i][0] for i in range(3)]).T
Opt = np.array([Ans[i][1] for i in range(3)]).T

plt.figure(figsize=(10,5))
h = plt.plot(R)
plt.legend(ε)
plt.figure(figsize=(10,5))
h = plt.plot(Opt)
plt.legend(ε)