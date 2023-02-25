    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################  

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
        #n = len(episode)
        T,t = (np.inf,0)
        S_terminal = episode[-1][-1]
        S_0 = episode[0][0]
        while tau != T-1:
            j = t%n1
            if t < T:
                s,a,r,s1 = episode[t]
                # print(t)
                S[j] = s1
                R[j] = r
                if s1 == S_terminal: # terminal state is not just the last state. 
                    T = t+1
            tau = t-n+1
            if tau >= 0:
                j = tau%n1
                ix = IX[j]
                Stemp = S[ix]
                steps = min(T-tau,n)
                S_tau = Stemp[-1]
                G = (R[ix][:steps])@Gamma[:steps]
                if steps > n:
                    G += gamman*V[Stemp[-2]]
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
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    return Q, pi
