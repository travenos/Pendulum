# A realization of a deep adaptive critic reinforcement learning algorithm
# DDPG (Deep Deterministic Policy Gradient) by Alexey Barashkov,
# master student of Moscow Technological University (MIREA)
from actor_critic import *


class ActorCriticDDPG(ActorCritic):
    """
    Goal of this class is learning actor and critic together with using of a target pair of actor and critic
    """
    def __init__(self, state_len=3, action_len=1, a_bound=2):
        DDPG_graph = tf.Graph()
        with DDPG_graph.as_default():
            self.TAU_CONST = 0.1  # Weights transfer rate
            super().__init__(state_len, action_len, a_bound)
            self.transfer_weights(1)

    def _construct_actor_critic(self):
        """
        Creating actor and critic models and initializing variables
        """
        # Main actor and critic networks
        super()._construct_actor_critic()
        # Target actor and critic networks
        self.actor2 = Actor(self.sess, **self.actor_param)
        self.critic2 = Critic(self.sess, self.actor2, **self.critic_param)

        self.weights_transfer_ops = []  # Weights transfer operations
        self.tau = tf.placeholder(tf.float64, (), "tau")    # Weights transfer rate placeholder

        def construct_weights_transfer(net1, net2):
            """Construct weights transfer computations for two nets"""
            for i in range(len(net1.weights)):
                new_weight = self.tau * net1.weights[i] + (1 - self.tau) * net2.weights[i]
                operation = tf.assign(net2.weights[i],  new_weight)
                self.weights_transfer_ops.append(operation)
                new_bias = self.tau * net1.biases[i] + (1 - self.tau) * net2.biases[i]
                operation = tf.assign(net2.biases[i], new_bias)
                self.weights_transfer_ops.append(operation)

        construct_weights_transfer(self.actor, self.actor2)
        construct_weights_transfer(self.critic, self.critic2)

    def transfer_weights(self, tau):
        """
        Transfer weights to target nets
        :param tau: transfer rate
        """
        feed_dict = {self.tau: tau}
        self.sess.run(self.weights_transfer_ops, feed_dict)

    def training(self, s, a, r, s1):
        """
        Training of the actor and the critic
        :param s: previous state vector
        :param a: action vector
        :param r: scalar reward, received for moving to state s1
        :param s1:  new state vector
        """
        super().training(s, a, r, s1)
        self.transfer_weights(self.TAU_CONST)

    def calculate_critic_goal(self, s, a, r, s1):
        """
        Compute Critic targets using Bellman equation
        :param s: state vector
        :param a: action vector
        :param r: scalar reward, received for moving to state s1
        :param s1:  new state vector
        :return: goal output for critic
        """
        # s, a, r, s1 are numpy arrays
        max_q1 = self.critic2.get_max_q(s1)
        if self.ALPHA == 1:  # Not necessary  to compute Q for initial state
            return r + self.GAMMA * max_q1
        q = self.critic.get_q(s, a)
        # Bellman equation
        return q + self.ALPHA * (r + self.GAMMA * max_q1 - q)
