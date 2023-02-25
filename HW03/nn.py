import numpy as np
from algo import ValueFunctionWithApproximation

#import tensorflow as tf
import torch.nn as nn
import torch

#%%
class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
#        model = tf.keras.models.Sequential([
#          tf.keras.layers.Flatten(input_shape=(state_dims,)),
#          tf.keras.layers.Dense(32, activation='relu'),
#          tf.keras.layers.Dense(32, activation='relu'),
#          tf.keras.layers.Dense(1)
#        ])       
        net = nn.Sequential(
                nn.Linear(state_dims,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,1)                
                )        
        self.optimizer = torch.optim.Adam(net.parameters(), 
                    betas=(0.9,0.999))
        self.loss_func = torch.nn.MSELoss() 
        self.net = net


    def __call__(self,s):
        # TODO: implement this method
        s = torch.from_numpy(s).float()
        return self.net(s).item()

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
#        s = np.array([s_tau])
#        y = np.array([[G]])
#        self.model.fit(x=s,y=y,batch_size=1,verbose=0)
#        return None
        self.optimizer.zero_grad()
        s = torch.from_numpy(s_tau).float()
        v = self.net(s)
        #G = torch.from_numpy(np.array([G])).float()
        G = torch.tensor(G,requires_grad=True)
        loss = self.loss_func(v,G)
        loss.backward()
        self.optimizer.step()
        
        
