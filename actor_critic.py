# A realization of a deep actor-critic reinforcement learning algorithm
# DDPG (Deep Deterministic Policy Gradient) by Alexey Barashkov,
# master student of Moscow Technological University (MIREA)
import tensorflow as tf
import numpy as np
from collections import deque


class Actor(object):
    """
    Actor model. It's goal is to give optimal action for every state
    """
    def __init__(self, sess, state_len, action_len, **kwargs):
        """
        Default constructor
        :param sess: TensorFlow session
        :param state_len: a number of coordinates in a state vector
        :param action_len: a number of coordinates in an action vector
        :param kwargs: other arguments:
         a_bound: a vector, containing bounds of every action coordinate
         hidden_neurons: a vector, containing a number of neurons in the hidden layers
         activations: a vector, containing activation functions for the hidden layers
        """
        self.lr = 0.001  # Actor learning rate
        self.sess = sess  # TensorFlow session
        self.STATE_LEN = state_len  # Length of a state vector
        self.ACTION_LEN = action_len  # Length of an action vector
        try:
            self.A_BOUND = kwargs['a_bound']
        except KeyError:  # If the argument was not given, using the default value
            self.A_BOUND = 1
        try:
            activations = list(kwargs['activations'])
        except KeyError:
            activations = []
        try:
            neurons_n = list(kwargs['hidden_neurons'])
        except KeyError:
            neurons_n = []
        activations.append(tf.nn.tanh)  # The last layer should have a tangential activation function
        self.ACTIVATIONS = activations
        # The last layer should have as many neurons as number of coordinates in action vector
        neurons_n.append(self.ACTION_LEN)
        self.NEURONS_N = neurons_n

        with tf.variable_scope('Actor'):
            self.s = tf.placeholder(tf.float64, [None, self.STATE_LEN], "state")  # Input state vector
            self.weights = []  # List of actor weights
            self.biases = []  # List of actor biases
            self._construct_net()  # Declare net variables and build computational graph

    def _construct_net(self):
        """
        Construction of the actor's neural net
        """
        for i in range(0, len(self.NEURONS_N)):
            if i == 0:
                w = tf.Variable(np.random.normal(0, 1, (self.STATE_LEN, self.NEURONS_N[i])), dtype=tf.float64)
            else:
                w = tf.Variable(np.random.normal(0, 1, (self.NEURONS_N[i - 1], self.NEURONS_N[i])), dtype=tf.float64)
            self.weights.append(w)
            b = tf.Variable(np.zeros((1, self.NEURONS_N[i])), dtype=tf.float64)
            self.biases.append(b)
        self._construct_computations(depth=len(self.NEURONS_N))  # Build computational graph

    def _construct_computations(self, depth):
        """
        Build computational graph
        :param depth: number of network layers
        """
        # First layer
        a = tf.matmul(self.s, self.weights[0]) + self.biases[0]
        if self.ACTIVATIONS[0] is not None:
            a = self.ACTIVATIONS[0](a)
        # Subsequent layers
        for i in range(1, depth):
            a = tf.matmul(a, self.weights[i]) + self.biases[i]
            if self.ACTIVATIONS[i] is not None:
                a = self.ACTIVATIONS[i](a)
        self.a = self.A_BOUND * a  # Output should lay in range from -A_BOUND to A_BOUND

    def get_action(self, s):
        """
        Get an optimal action vector prediction
        :param s: state vector
        """
        feed_dict = {self.s: s}
        return self.sess.run(self.a, feed_dict)


class Critic(object):
    """
    Critic model. It's goal is to approximate Q-function for every state and action
    """
    def __init__(self, sess, actor, hidden_neurons=tuple(), activations=tuple()):
        """
        Default constructor
        :param sess: TensorFlow session
        :param actor: object of an actor model
        :param hidden_neurons: a vector, containing a number of neurons in the hidden layers
        :param activations: a vector, containing activation functions for the hidden layers
        """
        self.sess = sess
        self.actor = actor

        self.GAMMA = 0.85  # Discount factor
        self.ALPHA = 1  # Training rate

        self.lr = 0.003  # Critic's optimizer's learning rate

        if len(hidden_neurons) != len(activations):
            raise ValueError("Every hidden layer should have exactly one activation function")

        self.ACTIVATIONS = list(activations)
        self.ACTIVATIONS.insert(0, tf.nn.tanh)  # Inserting the first layer for state preprocessing
        self.ACTIVATIONS.append(None)  # The output layer should have linear activation function
        self.NEURONS_N = list(hidden_neurons)
        self.NEURONS_N.insert(0, self.actor.STATE_LEN)  # Inserting the first layer for state preprocessing
        self.NEURONS_N.append(1)  # The critic network should have only one output

        # Variables for weights of the network
        with tf.variable_scope('Critic'):
            self.s = self.actor.s  # Using the same state input as an actor
            self.a_actor = self.actor.a  # Actor's output for connection to critic
            # Input for an external action vector (not from actor)
            self.a_ext = tf.placeholder(tf.float64, [None, self.actor.ACTION_LEN], "external_action")
            # A key for switching between actor's action and external action
            self.key = tf.placeholder(tf.bool, (), "action_switcher")  # 1 stands for actor action, 0 for extra
            nk = tf.logical_not(self.key)
            nk = tf.cast(nk, tf.float64)
            k = tf.cast(self.key, tf.float64)
            self.a = (self.a_actor * k + self.a_ext * nk)/actor.A_BOUND  # Action input to critic's net
            self.weights = []  # List of actor weights
            self.biases = []  # List of actor biases
            self._construct_net()  # Declare net variables and build computational graph

    def _construct_net(self):
        """
        Construction of the critic's neural net
        """
        for i in range(0, len(self.NEURONS_N)):
            if i == 0:  # First layer weights
                w = tf.Variable(np.random.normal(0, 1, (self.actor.STATE_LEN, self.NEURONS_N[i])), dtype=tf.float64)
            elif i == 1:  # Second layer weights
                w = tf.Variable(np.random.normal(0, 1,
                                (self.NEURONS_N[i - 1] + self.actor.ACTION_LEN, self.NEURONS_N[i])), dtype=tf.float64)
            else:  # Subsequent layers weights
                w = tf.Variable(np.random.normal(0, 1, (self.NEURONS_N[i - 1], self.NEURONS_N[i])), dtype=tf.float64)
            self.weights.append(w)
            b = tf.Variable(np.zeros((1, self.NEURONS_N[i])), dtype=tf.float64)
            self.biases.append(b)
        self._construct_computations(depth=len(self.NEURONS_N))   # Build computational graph

    def _construct_computations(self, depth):
        """
        Build computational graph
        :param depth: number of network layers
        """
        assert depth >= 2  # Critics network should have at least two layers

        # First layer
        q = tf.matmul(self.s, self.weights[0]) + self.biases[0]
        if self.ACTIVATIONS[0] is not None:
            q = self.ACTIVATIONS[0](q)
        # Second layer
        inp2 = tf.concat([q, self.a], 1)  # Adding actor input
        q = tf.matmul(inp2, self.weights[1]) + self.biases[1]
        if self.ACTIVATIONS[1] is not None:
            q = self.ACTIVATIONS[1](q)
        # Subsequent layers
        for i in range(2, depth):
            q = tf.matmul(q, self.weights[i]) + self.biases[i]
            if self.ACTIVATIONS[i] is not None:
                q = self.ACTIVATIONS[i](q)
        self.Q = q  # Critic's output
        with tf.variable_scope('squared_TD_error'):
            self.Q_goal = tf.placeholder(tf.float64, [None, 1], "Q_goal")  # Critic's output goal
            self.loss = tf.reduce_mean(tf.square(self.Q_goal - self.Q))  # Loss function for critic training
        with tf.variable_scope('critic_training'):
            # Critic training
            # Trainable variables are only critic's weights and biases
            self.critic_train_op = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss, var_list=self.weights+self.biases)
        with tf.variable_scope('actor_training'):
            # Actor training. Actor's goal is to maximize critic's output
            # Trainable variables are only actor's weights and biases
            self.actor_train_op = tf.train.AdamOptimizer(
                self.actor.lr).minimize(-self.Q, var_list=self.actor.weights+self.actor.biases)

    def critic_training(self, s, a, r, s1):
        """
        Train the critic on batch of experience
        :param s: previous state vector
        :param a: action vector
        :param r: scalar reward, received for moving to state s1
        :param s1:  new state vector
        """
        # s, a, r, s1 are numpy arrays
        q_goal = self._calc_target(s, a, r, s1)  # Calculate a butch of goal values for critic
        # While training critic, actor in not connected (key=0)
        feed_dict = {self.s: s, self.Q_goal: q_goal, self.key: 0, self.a_ext: a}
        _, loss = self.sess.run([self.critic_train_op, self.loss], feed_dict)  # Train the critic with one iteration
        return loss

    def actor_training(self, s):
        """
        Train the actor
        :param s: state vector
        """
        # The goal of an actor's training is maximization of critic's output
        # Thus actor is connected to critic (key=1)
        feed_dict = {self.s: s, self.key: 1, self.a_ext: np.zeros((s.shape[0], self.actor.ACTION_LEN))}
        self.sess.run(self.actor_train_op, feed_dict)  # Train the actor with one iteration

    def get_q(self, s, a):
        """
        Get value of Q-function for action a in state s
        :param s: state vector
        :param a: action vector
        :return: predicted value of Q-function
        """
        # Only critic is used (key=0)
        feed_dict = {self.s: s, self.key: 0, self.a_ext: a}
        return self.sess.run(self.Q, feed_dict)

    def get_max_q(self, s):
        """
        Get value of Q-function for optimal action, predicted by actor, in state s
        :param s: state vector
        :return: predicted maximum value of Q-function
        """
        # Actor predicts an optimal action. So it should be connected (key=1)
        feed_dict = {self.s: s, self.key: 1, self.a_ext: np.zeros((s.shape[0], self.actor.ACTION_LEN))}
        return self.sess.run(self.Q, feed_dict)

    def _calc_target(self, s, a, r, s1):
        """
        Compute Critic targets using Bellman equation
        :param s: previous state vector
        :param a: action vector
        :param r: scalar reward, received for moving to state s1
        :param s1:  new state vector
        :return: a goal for critic
        """
        # s, a, r, s1 are numpy arrays
        q = self.get_q(s, a)
        max_q1 = self.get_max_q(s1)
        # Bellman equation
        return q + self.ALPHA * (r + self.GAMMA * max_q1 - q)


class ActorCritic(object):
    """
    Goal of this class is learning with actor and critic connected together
    """
    def __init__(self, state_len=3, action_len=1, a_bound=2):
        """
        Default constructor
        """
        np.random.seed(42)    # Randomizers' initialization for getting repeatable results
        tf.set_random_seed(42)

        self.replay_memory = deque(maxlen=100000)  # History
        self.EPS_GREEDY = True  # Make random actions sometimes
        self.eps = 0.5  # Initial probability of random action
        self.EPS_DISCOUNT = 0.000008334  # By this value probability of random action is decreased by every step
        self.MIN_EPS = 0.05  # Minimum probability of random action
        self.BATCH_SIZE = 50  # Size of training batch on every step

        # Parameters for creation actor and critic models
        self.actor_param = {"state_len": state_len, "action_len": action_len, "a_bound": a_bound,
                            "hidden_neurons": (30, 7), "activations": (tf.nn.relu, tf.nn.sigmoid)}
        self.critic_param = {"hidden_neurons": (40, 19), "activations": (tf.nn.relu, tf.nn.sigmoid)}

        self.sess = tf.InteractiveSession()  # Starting a new TensorFlow session
        self._construct_actor_critic()  # Creating actor and critic models
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  # TensorFlow session saver

    def _construct_actor_critic(self):
        """
        Creating actor and critic models and initializing variables
        """
        self.actor = Actor(self.sess, **self.actor_param)
        self.critic = Critic(self.sess, self.actor, **self.critic_param)

    def save_to_file(self, file_name):
        """
        Saving session to hard drive
        :param file_name: name for model files
        """
        try:
            self.saver.save(self.sess, file_name)
        except tf.errors.PermissionDeniedError:
            raise PermissionError("Writing permission denied for file "+file_name)

    def load_from_file(self, file_name):
        """
        Loading session from hard drive
        :param file_name: model files' name
        """
        extension_len = 5
        try:
            self.saver.restore(self.sess, file_name[:-extension_len])
        except Exception:
            raise OSError("Can not restore TensorFlow session from this file")

    def reset_nn(self):
        self.sess.run(tf.global_variables_initializer())

    def compute_batch(self, s):
        """
        Predict an optimal action in every state from the batch s
        :param s: batch of states to be processed
        :return: batch of predicted optimal actions for every state
        """
        if type(s) != np.ndarray:
            raise TypeError("s should be a numpy array")
        a = self.actor.get_action(s)
        return a

    def compute(self, s):
        """
        Predict an optimal action in state s
        :param s: state vector
        :return: action vector
        """
        if type(s) != np.ndarray:
            raise TypeError("s should be a numpy array")
        if len(s) != self.get_inputs_count():
            raise ValueError("Length of s should be equal to number of inputs")
        rnd = np.random.sample(1)
        # Choose a random action
        if self.EPS_GREEDY and rnd < self.eps:
            a = np.random.uniform(-self.actor.A_BOUND, self.actor.A_BOUND, self.actor.ACTION_LEN)
        else:  # Choose predicted optimal action
            sa = np.array([s])
            qa = self.compute_batch(sa)
            a = qa[0]
        if self.EPS_GREEDY and self.eps > self.MIN_EPS:
            self.eps -= self.EPS_DISCOUNT  # Decreasing random action probability
        return a

    def training(self, s, a, r, s1):
        """
        Training of the actor and the critic
        :param s: previous state vector
        :param a: action vector
        :param r: scalar reward, received for moving to state s1
        :param s1:  new state vector
        """
        if type(s) != np.ndarray:
            raise TypeError("s should be a numpy array")
        if type(r) == np.ndarray or type(r) == list or type(r) == tuple:
            raise TypeError("r should be a scalar")
        if type(s1) != np.ndarray:
            raise TypeError("s1 should be a numpy array")

        self.replay_memory.append([s, a, r, s1])  # Adding new experience to history
        if len(self.replay_memory) < self.BATCH_SIZE:  # No training if there are not enough experience in the history
            return

        batch_indexes = set()  # Random indexes from the experience history
        batch_indexes.add(len(self.replay_memory)-1)
        while len(batch_indexes) < self.BATCH_SIZE:
            rnd = np.random.randint(0, len(self.replay_memory) - 1)
            batch_indexes.add(rnd)
        batch_indexes = list(batch_indexes)
        np.random.shuffle(batch_indexes)

        # Bunches of vectors for training
        sb = np.zeros((self.BATCH_SIZE, self.actor.STATE_LEN))
        ab = np.zeros((self.BATCH_SIZE, self.actor.ACTION_LEN))
        rb = np.zeros((self.BATCH_SIZE, 1))
        s1b = np.zeros((self.BATCH_SIZE, self.actor.STATE_LEN))

        for i, index in enumerate(batch_indexes):
            sb[i, :] = self.replay_memory[index][0]
            ab[i, :] = self.replay_memory[index][1]
            rb[i, :] = self.replay_memory[index][2]
            s1b[i, :] = self.replay_memory[index][3]

        self.critic.actor_training(s1b)  # Actor training with new state vectors
        self.critic.critic_training(sb, ab, rb, s1b)  # Critic training
        self.critic.actor_training(sb)  # Actor training with previous state vectors

    def get_inputs_count(self):
        """
        Get length of state vector
        :return: Number of state coordinates
        """
        return self.actor.STATE_LEN

    def get_outputs_count(self):
        """
        Get length of action vector
        :return: Number of action coordinates
        """
        return self.actor.ACTION_LEN

    def make_graph(self, folder_name):
        """
        Generate a TensorBoard computational graph
        :param folder_name: TensorBoard logs folder
        """
        tf.summary.FileWriter(folder_name, self.sess.graph)

    def close_session(self):
        """
        Close TensorFlow session
        """
        self.sess.close()
