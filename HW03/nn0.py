import numpy as np
from algo import ValueFunctionWithApproximation

import tensorflow as tf

#%%
class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(state_dims,)),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(1)
        ])
            
        model.compile(optimizer='adam',beta1=0.9,beta2=0.999,
                      loss='mean_squared_error')
        self.model = model

    def __call__(self,s):
        # TODO: implement this method
        s = np.array([s])
        p = self.model.predict(s,batch_size=1)
        return p[0,0]

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        s = np.array([s_tau])
        y = np.array([[G]])
        self.model.fit(x=s,y=y,batch_size=1,verbose=0)
        return None

