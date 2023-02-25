from typing import Iterable
import numpy as np
import torch.nn as nn
import torch
#%%
class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        net = nn.Sequential(
            nn.Linear(state_dims,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,num_actions),
            nn.Softmax(dim=-1))        
        self.optimizer = torch.optim.Adam(net.parameters(), 
                    betas=(0.9,0.999),lr=alpha) 
        self.actions = np.arange(num_actions)     
        self.net = net
        
    def __call__(self,s) -> int:
        # TODO: implement this method
        # state to action 
        s = torch.from_numpy(s).float()
        prob = self.net(s).detach().numpy()
        return np.random.choice(self.actions, p=prob)

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optimizer.zero_grad()
        s = torch.from_numpy(s).float()
        loss = gamma_t*delta*torch.log(self.net(s)[a])
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        net = nn.Sequential(
            nn.Linear(state_dims,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1))        
        self.optimizer = torch.optim.Adam(net.parameters(), 
                    betas=(0.9,0.999),lr=alpha)
        self.loss_func = torch.nn.MSELoss()
        self.net = net

    def __call__(self,s) -> float:
        # TODO: implement this method
        s = torch.from_numpy(s).float()
        return self.net(s).item()

    def update(self,s,G):
        # TODO: implement this method
        self.optimizer.zero_grad()
        s = torch.from_numpy(s).float()
        v = self.net(s)
#        G = torch.tensor(G,requires_grad=True)
        G = torch.from_numpy(np.array([G])).float()
        loss = self.loss_func(v,G)
        loss.backward()
        self.optimizer.step()

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    ans = np.empty(num_episodes)
    for i in range(num_episodes):
#        s = torch.from_numpy(env.reset()).float()
        s = env.reset()
        t = 0
        Rewards = []
        S = [s]
        A = []
        done = False
        while done is False:
            a = pi(s)
            s,r,done,_ = env.step(a)
#            s = torch.from_numpy(s).float()
            Rewards.append(r)
            S.append(s)
            A.append(a)
        Rewards = -np.array(Rewards)
        T = len(Rewards)
        Gamma = gamma**np.arange(T)
        for t in range(T):
            G = Gamma[:T-t]@Rewards[t:]
            s = S[t]
            delta = G-V(s)
            V.update(s,delta)
            pi.update(s,A[t],gamma**t,delta)
        ans[i] = -Gamma@Rewards
    return ans

