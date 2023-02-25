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
    ix = np.arange(n1)
    IX = np.array([np.roll(ix,i) for i in ix])
    Gamma = g**ix
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
            j1 = (t+1)%n1
            if t < T:
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
                if steps > n:
                    G += gamman*V[S[j1]]
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
    ix = np.arange(n1)
    IX = np.array([np.roll(ix,i) for i in ix])
    Gamma = g**ix
    gamman = g**n
    tau = 0
    
    for episode in trajs:
        T,t = (np.inf,0)
        S_terminal = episode[-1][-1]
        S[-1] = episode[0][0]
        A[-1] = episode[0][1]
        # R.fill(0)
        while tau != T-1:
            j = t%n1
            if t < T:
                s,a,r,s1 = episode[t]
                # print(t)
                S[j] = s1
                R[j] = r
                if s1 == S_terminal: # terminal state is not just the last state. 
                    T = t+1
                else:
                    s,a,r,s1 = episode[t+1]
                    A[j] = a
            tau = t-n+1
            if tau >= 0:
                ix = IX[tau%n1]
                Stemp,Atemp = (S[ix],A[ix])
                Rho = [Pi.action_prob(Stemp[i],Atemp[i])/bpi.action_prob(Stemp[i],Atemp[i]) 
                       for i in range(min(n,T-tau-1))]
                if 0 in Rho:
                    t += 1
                    continue
                else:                
                    steps = min(T-tau,n)
                    G = (R[ix][:steps])@Gamma[:steps]
                    if steps > n:
                        G += gamman*Q[Stemp[-2],Atemp[-2]]
                    S_tau,A_tau = (Stemp[-1],Atemp[-1])
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
