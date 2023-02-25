from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    nS,nA = initQ.shape
    Q = initQ.copy()
    C = np.zeros((nS,nA))
    g = env_spec.gamma 
    for episode in trajs:
        T = len(episode)
        G,W = (0,1)
        for t in range(T-1,-1,-1):
            s,a,r,s1 = episode[t]
            G = g*G + r
            C[s,a] += 1
            Q[s,a] = Q[s,a] + W/C[s,a]*(G-Q[s,a])
            W = W*pi.action_prob(s,a)/bpi.action_prob(s,a)
            if W == 0:
                break            
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    nS,nA = initQ.shape
    Q = initQ.copy()
    C = np.zeros((nS,nA))
    g = env_spec.gamma 
    for episode in trajs:
        T = len(episode)
        G,W = (0,1)
        for t in range(T-1,-1,-1):
            s,a,r,s1 = episode[t]
            G = g*G + r
            C[s,a] = C[s,a] + W
            Q[s,a] = Q[s,a] + W/C[s,a]*(G-Q[s,a])
            W = W*pi.action_prob(s,a)/bpi.action_prob(s,a)
            if W == 0:
                break    
    return Q
