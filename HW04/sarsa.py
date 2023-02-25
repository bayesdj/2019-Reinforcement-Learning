import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here        
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        
        dim = np.ceil((state_high-state_low)/tile_width) + 1
        self.tile_dim = dim.astype(int)        
        self.ntiles = int(dim.prod())
        tiling_index = np.arange(num_tilings)/num_tilings
        self.tiling_start = state_low - np.outer(tiling_index,tile_width)
        self.tiling_idx_start = np.array([i*self.ntiles for i in range(num_tilings)])

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.num_actions * self.num_tilings * self.ntiles

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        d = self.feature_vector_len()
        u = np.zeros(d)
        if not done:
            positions = np.floor((s-self.tiling_start)/self.tile_width).astype(int)
            idx = np.ravel_multi_index(positions.T,dims=self.tile_dim)
            idx += self.tiling_idx_start
            k = self.ntiles * self.num_tilings
            tiles = np.zeros(k)
            tiles[idx] = 1
            u[k*a:k*(a+1)] = tiles
        return u

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    nFeature = X.feature_vector_len()
    w = np.zeros(nFeature)
    epsilon = 0    
#    d = len(X.state_low)
    for i in range(num_episode):
        s, done = env.reset(), False
        a = epsilon_greedy_policy(s,done,w,epsilon)
        x = X(s,done,a)
        z = np.zeros(nFeature)
        Q0 = 0        
        while not done:
            s1,r,done,_ = env.step(a)
            a1 = epsilon_greedy_policy(s1,done,w,epsilon)
            x1 = X(s1,done,a1)
            Q = w@x
            Q1 = w@x1
            delta = r+gamma*Q1-Q
            z = gamma*lam*z+(1-alpha*gamma*lam*(z@x))*x
            w = w+alpha*(delta+Q-Q0)*z-alpha*(Q-Q0)*x
            Q0 = Q1
            x = x1
            a = a1        

    #TODO: implement this function
    return w
