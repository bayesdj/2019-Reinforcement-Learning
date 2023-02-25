import numpy as np
import torch.nn as nn
import torch
import gym
import math
import random
#%%
class ValueFunctionWithTile():
    def __init__(self,
                 C:np.ndarray,
                 state_lo:np.array,
                 state_hi:np.array,
                 λ:float,
                 α:float):
  
        self.n, self.d = C.shape
        self.α = α
        self.λ = λ    
        self.C = C*np.pi
        self.state_lo = state_lo
        self.state_range = state_hi-state_lo
        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.λ = λ
        self.v0 = 0

    def __call__(self,s):
        xs = self.feat(s)
        return self.w@xs
    
    def feat(self,s):
        s = (s-self.state_lo)/self.state_range
        xs = np.cos(self.C@s)
        return xs        
        
    def update(self,a,dr,s1):
        # TODO: implement this method
        x1 = self.feat(s1)
        x = self.x
        v = self.w@x
        v1 = self.w@x1
        δ = dr+v1-v
        z = self.λ*self.z
        z += (1-self.α*z@x)*x
        dv = v-self.v0
        self.z = z
        self.w += self.α*(δ+dv)*z-self.α*dv*x
        self.v0 = v1
        self.x = x1
        return δ

class PiApproximationWithGaussian():
    def __init__(self,
                 C:np.ndarray,
                 state_lo:np.array,
                 state_hi:np.array,
                 λ:float,
                 α:np.array):   
        #self.d = state_limit.shape[1]
        self.n, self.d = C.shape
        self.α = α
        self.λ = λ    
        self.C = C*np.pi
        self.state_lo = state_lo
        self.state_range = state_hi-state_lo
        self.θμ = np.zeros(self.n)
        self.θσ = np.zeros(self.n)
        self.zμ = np.zeros(self.n)
        self.zσ = np.zeros(self.n)
        
    def feat(self,s) -> np.array:
        s = (s-self.state_lo)/self.state_range
        xs = np.cos(self.C@s)
        return xs
       
    def __call__(self,s,seed=None) -> float:
        # TODO: implement this method
        # state to action 
        # if seed is not None:
            # random.seed(seed)
        xs = self.feat(s)
        μ = self.θμ@xs
        σ = math.exp(self.θσ@xs)
        return random.gauss(μ,σ)

    def update(self,s,a,δ):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        xs = self.feat(s)
        μ = self.θμ@xs
        σ2 = math.exp(self.θσ@xs)**2
        z = (a-μ)/σ2
        gradμ = z*xs
        gradσ = (z*(a-μ)-1)*xs
        self.zμ = self.λ*self.zμ + gradμ
        self.zσ = self.λ*self.zσ + gradσ
        self.θμ += self.α[0]*δ*self.zμ
        self.θσ += self.α[1]*δ*self.zσ
        
def actor_critic(pi:PiApproximationWithGaussian,
                 V:ValueFunctionWithTile,
                 env,
                 α:float,
                 T=200,
                 seed=None) -> float:
    rbar = 0
    if seed is not None:
        env.seed(seed)
    s = env.reset()
    V.x = V.feat(s)
    env.spec.max_episode_steps = T+1
    Rewards = np.empty(T)
    for t in range(T):
        a = pi(s,seed)
        # env.render()
        s1,r,done,_ = env.step(np.array([a]))
        δ = V.update(a,r-rbar,s1)
        rbar += α*δ
        pi.update(s,a,δ)
        s = s1
        Rewards[t] = r
    return Rewards    
#%%

