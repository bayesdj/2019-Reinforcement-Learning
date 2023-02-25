import numpy as np
from matplotlib import pyplot as plt
import sys
#from numba import njit
#%%
filename = sys.argv[1]

#@njit()
def getRew(α,ε):
    R = np.empty((T,S))
    R_a = R.copy()
    percent = np.full((T,S),False,dtype=np.bool_)
    percent_a = percent.copy()
    for s in range(S):        
        Q,Qcount,Qa,Qacount,qstar = [np.zeros(nA) for i in range(5)]
        Explore = np.random.rand(T) < ε
        norm = np.random.randn(T)
        norm2 = np.random.randn(T,nA)*0.01
        norm2[0] = 0
        for t in range(T):
            qstar += norm2[t]
            if Explore[t]:
                At = np.random.choice(actions)
                At_a = At
            else:
                At = Q.argmax()
                At_a = Qa.argmax()
            noise = norm[t]
            r = noise + qstar[At]
            if At_a == At:
                r_a = r
            else:
                r_a = np.random.randn() + qstar[At_a]
            n = Qcount[At]+1
            Q[At] = (Q[At]*(n-1) + r)/n
            Qa[At_a] = Qa[At_a] + α*(r_a-Qa[At_a])
            Qcount[At] = n
            R[t,s] = r  
            R_a[t,s] = r_a
            A_best = qstar.argmax()
            percent[t,s] = (At==A_best)
            percent_a[t,s] = (At_a==A_best)
    muR,muPercent = (R.sum(1)/S, percent.sum(1)/S)
    muRa,muPercenta = (R_a.sum(1)/S, percent_a.sum(1)/S)
    return np.stack((muR,muRa)),np.stack((muPercent,muPercenta)) 
#    return 3,2
    
#%%   
nA = 10
actions = np.arange(nA)
T = 10000
S = 300
Qstar = np.zeros((S,nA))

α,ε = (0.1,0.1)
R,pi = getRew(α,ε)

#%%
ans = np.vstack((R[0],pi[0],R[1],pi[1]))
np.savetxt(filename,ans)