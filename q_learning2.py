import numpy as np
import copy
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque

from q_learning import QLearningModel


class QLearningModel2(QLearningModel):
    def __init__(self, state_len=3, action_len=1, a_bound=2, actions=2, neurons_n=(40, 14),
                 activations=("relu", "sigmoid")):
        """Конструктор"""
        super().__init__(state_len, action_len, a_bound, actions, neurons_n, activations)

    def _construct_net(self, state_len, action_len, actions, neurons_n, activations):
        self._inputs_n = state_len + action_len
        self._neurons_n = list(neurons_n)
        self._neurons_n.append(1)
        self._activations = list(activations)
        self._activations.append("linear")
        self.neuralNet = Sequential()  # Нейронная сеть
        self._new_nn(self._inputs_n, self._neurons_n, self._activations)
