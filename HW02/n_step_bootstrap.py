from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    nS,g = (env_spec.nS,env_spec.gamma)
    n1 = n+1 
    S = np.zeros(n1,dtype=np.uint32)
    R = np.zeros(n1)
    V = initV.copy()
    Gamma = g**np.arange(n1)
    gamman = g**n
    tau = 0
    
    for episode in trajs:
        T,t = (np.inf,0)
        S_terminal = episode[-1][-1]
        R[0] = 0
        S[0] = episode[0][0]
        # R.fill(0)
        #S[-1] = S_0
        while tau != T-1:            
            if t < T:
                j1 = (t+1)%n1
                s,a,r,s1 = episode[t]
                # print(t)
                S[j1] = s1
                R[j1] = r
                if s1 == S_terminal: # terminal state is not just the last state. 
                    T = t+1
            tau = t-n+1
            if tau >= 0:
                j = tau%n1
                steps = min(T-tau,n)
                ix = np.arange(j+1,j+1+steps)%n1
                G = R[ix]@Gamma[:steps]
                tau_n = tau+n
                if tau_n < T:
                    G += gamman*V[S[tau_n%n1]]
                S_tau = S[j]    
                V[S_tau] += alpha*(G-V[S_tau])                
            t += 1    
    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:

    nS,nA = initQ.shape
    g = env_spec.gamma
    n1 = n+1 
    S = np.zeros(n1,dtype=np.uint32)
    A = S.copy()
    R = np.zeros(n1)
    Q = initQ.copy()
    Pi = PolicyQ(Q)
    Gamma = g**np.arange(n1)
    gamman = g**n
    tau = 0
    
    for episode in trajs:
        T,t = (np.inf,0)
        S_terminal = episode[-1][-1]
        S[0] = episode[0][0]
        A[0] = episode[0][1]
        R[0] = 0
        # R.fill(0)
        while tau != T-1:            
            if t < T:
                s,a,r,s1 = episode[t]
                j1 = (t+1)%n1
                S[j1] = s1
                R[j1] = r
                if s1 == S_terminal: # terminal state is not just the last state. 
                    T = t+1
                else:
                    s,a,r,s1 = episode[t+1]
                    A[j1] = a
            tau = t-n+1
            if tau >= 0:
                ix = np.arange(tau+1,min(tau+n,T-1)+1)%n1
                Rho = [Pi.action_prob(S[i],A[i])/bpi.action_prob(S[i],A[i]) for i in ix]
                if 0 in Rho:
                    t += 1
                    continue
                else:                
                    j = tau%n1
                    steps = min(T-tau,n)
                    ix = np.arange(j+1,j+1+steps)%n1
                    G = R[ix]@Gamma[:steps]
                    tau_n = tau+n
                    if tau_n < T:
                        j1 = tau_n%n1
                        G += gamman*Q[S[j1],A[j1]]
                    S_tau,A_tau = (S[j],A[j])
                    rho = np.array(Rho).prod()
                    Q[S_tau,A_tau] += alpha*rho*(G-Q[S_tau,A_tau])
                    Pi = PolicyQ(Q)
            t += 1
    return Q,Pi
    
class PolicyQ(Policy):
    def __init__(self,Q):
        self.a = Q.argmax(1)

    def action_prob(self,state,action):
        return 1 if action == self.a[state] else 0

    def action(self,state):
        return self.a[state]
