from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    spec = env.spec
    nS,nA,g = (spec.nS, spec.nA, spec.gamma)
    e = np.inf		
    states = np.arange(nS)
    actions = np.arange(nA)
    V = initV.copy()
    Q = np.tile(V,(nA,1)).T
    #Q = V2.T
    # Pi = {s:np.array([pi.action_prob(s,a) for a in actions]) for s in states}
    Pi = np.array([[pi.action_prob(s,a) for a in actions] for s in states])
    P,R = (env.TD, env.R)
    while e > theta:
        e = 0
        V0 = V.copy()
        for s in states:
            p,r = (P[s],R[s])
            x = (p*(r + g*V)).sum(1)
            Q[s] = x
            V[s] = Pi[s]@x
        #V = (Pi*Q).sum(1)
        e = np.abs(V-V0).max()        
    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """
    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    spec = env.spec
    nS,nA,g = (spec.nS, spec.nA, spec.gamma)
    e = np.inf		
    states = np.arange(nS)
    V = initV.copy()
    Pi0 = np.zeros((nS,nA))
    P,R = (env.TD, env.R)
    while e > theta:
        e = 0
        V0 = V.copy()
        Pi = Pi0.copy()
        for s in states:
            p,r = (P[s],R[s])
            x = (p*(r + g*V)).sum(1)
            a = x.argmax()
            V[s] = x[a]
            Pi[s,a] = 1
        e = np.abs(V-V0).max()        
    return V, Policy1(Pi)

class Policy1(Policy):
    def __init__(self,P):
        self.p = P

    def action_prob(self,state,action):
        return self.p[state,action]

    def action(self,state):
        p = self.p[state]        
        return p.argmax()