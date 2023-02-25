from typing import Iterable
import numpy as np

import collections
import itertools
import time
import tensorflow as tf

global_sess = 0

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
        start_time = time.time()

        self.state_dims = state_dims
        self.num_actions = num_actions
        scope = "policy_approx"

        self.state = tf.placeholder(dtype=tf.float32, shape=(None,self.state_dims), name="state")
        self.action = tf.placeholder(dtype=tf.int32, name="action")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        hiddensize = 32
        self.l1 = tf.layers.dense(self.state, units=hiddensize, activation=tf.nn.relu)
        self.l2 = tf.layers.dense(self.l1, units=hiddensize, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.l2, units=num_actions, activation=None)

        self.action_probs = tf.squeeze(tf.nn.softmax(self.logits))
        self.picked_action_prob = tf.gather(self.action_probs, self.action)

        # Loss and train op
        self.loss = -tf.log(self.picked_action_prob) * self.target

        self.alpha = 0.0003  # <= 3 * 10^-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.alpha,beta1=self.beta1,beta2=self.beta2)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self,s) -> int:
        if s.shape[0] == self.state_dims:
            s = np.expand_dims(s, axis=0)

        action_probs = self.sess.run(self.action_probs, { self.state: s })
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, s, a, gamma_t, delta, sess=None):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        if s.shape[0] == self.state_dims:
            s = np.expand_dims(s, axis=0)

        feed_dict = { self.state: s, self.target: delta, self.action: a  }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss

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
        self.state_dims = state_dims
        self.b = state_dims
        self.state = tf.placeholder(shape=(None, state_dims), dtype=tf.float32,name="state")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        hiddensize = 32
        self.l1 = tf.layers.dense(self.state, units=hiddensize, activation=tf.nn.relu)
        self.l2 = tf.layers.dense(self.l1, units=hiddensize, activation=tf.nn.relu)
        self.output_layer = tf.layers.dense(self.l2, units=1, activation=None)

        self.value_estimate = tf.squeeze(self.output_layer)
        self.loss = tf.squared_difference(self.value_estimate, self.target)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

        self.vsess = tf.Session()
        self.vsess.run(tf.global_variables_initializer())

    def __call__(self,s, sess=None) -> float:
        if s.shape[0] == self.state_dims:
            s = np.expand_dims(s, axis=0)
        return self.vsess.run(self.value_estimate, { self.state: s })

    def update(self, s, G, sess=None):
        if s.shape[0] == self.state_dims:
            s = np.expand_dims(s, axis=0)
        feed_dict = { self.state: s, self.target: G }
        _, loss = self.vsess.run([self.train_op, self.loss], feed_dict)
        return loss


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
    debug = False

    start_time = time.time()
    if debug:
        print("session initialize: ", time.time() - start_time)

    # simple way to tell which baseline i'm using
    if V.b == 0:
        #no baseline
        with_baseline = 0
    else:
        with_baseline = 1

    print("With Baseline:",with_baseline," V type",type(V))

    # Keeps track of useful statistics
    final_stats = {"episode_lengths":np.zeros(num_episodes), "episode_rewards":np.zeros(num_episodes)}

    for i_episode in range(num_episodes):
        start_ep = time.time()
        # Reset the environment and pick the first action
        state = env.reset()
        episode = []
        for t in itertools.count():

            # Take a step
            action = pi(np.expand_dims(state,axis=0))

            next_state, reward, done, _ = env.step(action)

            transaction = [state,action,reward,next_state,done]
            episode.append(transaction)

            final_stats["episode_rewards"][i_episode] += reward

            if done:
                end_collect = time.time()
                if debug:
                    print("\rStep {} @ Episode {}/{} ({})".format(
                        t, i_episode + 1, num_episodes, final_stats["episode_rewards"][i_episode - 1]))#, end="")
                    print("\rEpisode collection for ep{}=({})".format( i_episode, end_collect - start_ep))
                break

            state = next_state

        # Go through the episode and make policy/value updates
        for t, transition in enumerate(episode):
            # return after this time step
            G = sum(gamma**i * tr[2] for i, tr in enumerate(episode[t:]))

            # calculate baseline/advantage
            s = np.expand_dims(transition[0],axis=0) 
            a = transition[1]

            if with_baseline == 1:
                baseline_value = V(s) 
                delta = G - baseline_value
                if debug:
                    print("in with baseline G",G,"basline",baseline_value,"delta",delta)

                # Update our value approx
                V.update(s, delta)                  #update w
                pi.update(s, a, gamma, delta )   #update theta
            else:
                if debug:
                    print("in without baseline")
                # Update our policy estimator
                pi.update(s, a, gamma, G )   
        end_ep = time.time()
        if debug:
            print("\r--- Episode time for ep{}=({})".format( i_episode, end_ep - start_ep))


    #return list that includes the G_0 for every episodes.
    print(final_stats)
    end_time= time.time()
    print("Elapsed Time", end_time - start_time)
    return final_stats["episode_rewards"]
