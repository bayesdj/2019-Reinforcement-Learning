import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
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
        
    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        f_ix = self.feat(s_tau)
        grad = np.zeros(self.d)
        grad[f_ix] = 1
        v = self.w[f_ix].sum()
        self.w += alpha*(G-v)*grad
        return None
