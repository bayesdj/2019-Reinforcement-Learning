import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    for i in range(num_episode):
        t,T = (0,np.inf)
        done = False
        R = np.zeros(1)
        s = env.reset()
        S = [s]
        print(i)
        while True:
            if t < T:
                a = pi.action(s)
                s,r,done,_ = env.step(a)
                R = np.append(R,r)
                S.append(s)
                if done: T = t+1
            τ = t-n+1
            if τ >= 0:
                #k = min(τ+n,T)
                rng = np.arange(τ+1,min(τ+n,T)+1)
                G = R[rng]@gamma**(rng-τ-1)
                if τ+n<T: G += gamma**n*V(S[τ+n])
                V.update(alpha,G,S[τ])
            if τ == T-1:
                break
            else:
                t += 1
            
                
        
