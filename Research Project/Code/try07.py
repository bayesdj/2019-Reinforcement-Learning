import numpy as np
import torch.nn as nn
import torch
import gym
import math
import random
#%%
class ValueFunctionWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array,
                 λ:float,
                 α:float):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method
        dim = np.ceil((state_high-state_low)/tile_width).astype(int)+1
        self.tile_width = tile_width
        self.tile_dim = dim
        self.ntiles = dim.prod()
        self.d = self.ntiles*num_tilings
        self.w = np.zeros(self.d)
        tiling_index = np.arange(num_tilings)/num_tilings
        self.tiling_start = state_low - np.outer(tiling_index,tile_width)
        self.tiling_idx_start = np.array([i*self.ntiles for i in range(num_tilings)])
        self.z = np.zeros(self.d)
        self.α = α/num_tilings
        self.λ = λ

    def __call__(self,s):
        # TODO: implement this method      
        features = self.feat(s)
        return self.w[features].sum()
    
    def feat(self,s):
        positions = np.floor((s-self.tiling_start)/self.tile_width).astype(int)
        idx = np.ravel_multi_index(positions.T,dims=self.tile_dim)
        idx += self.tiling_idx_start
#        k = self.ntiles * self.num_tilings
#        feat = np.zeros(k)
#        feat[idx] = 1
        return idx        
        
    def update(self,s,δ):
        # TODO: implement this method
        f_ix = self.feat(s)
        grad = np.zeros(self.d)
        grad[f_ix] = 1
#        v = self.w[f_ix].sum()
        z = self.λ*self.z + grad       
        self.w += self.α*δ*z
        self.z = z

class PiApproximationWithGaussian():
    def __init__(self,
                 C:np.ndarray,
                 # action_limit:np.array,
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

    def update(self, s, a, δ):
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
        
def actor_critic(π:PiApproximationWithGaussian,
                 V:ValueFunctionWithTile,
                 env,
                 α:float,
                 T=200,
                 seed=None) -> float:
    rbar = 0
    if seed is not None:
        env.seed(seed)
    s = env.reset()
    env.spec.max_episode_steps = T+1
    Rewards = np.empty(T)
    for t in range(T):
        a = π(s,seed)
        # env.render()
        s1,r,done,_ = env.step(np.array([a]))
        δ = r-rbar+V(s1)-V(s)
        rbar += α*δ
        V.update(s,δ)
        π.update(s,a,δ)
        s = s1
        Rewards[t] = r
    return Rewards    
#%%

