import gym
import numpy as np
from PyQt5 import QtWidgets

from q_learning import QLearningModel
from q_learning2 import QLearningModel2
from actor_critic import ActorCritic


class PendulumLearning(object):
    def __init__(self, GUI):
        self.learning_models_list = [QLearningModel(), QLearningModel2(), ActorCritic()]  # Используемы модели

        self.learning_model = self.learning_models_list[0]  # Ссылка на применяемую модель

        self.env = gym.make('Pendulum-v0')
        self.env.seed(1)  # reproducible

        self.s_len = self.env.observation_space.shape[0]
        self.a_bound = self.env.action_space.high

        self.s = np.zeros(self.s_len)   # Предыдущее состояние маятника
        self.reset_env()    # Перезапустить маятник

        self.is_learning = True

    def choose_learning_model(self, number):
        self.learning_model = self.learning_models_list[number]

    def step(self):
        self.env.render()
        a = self.learning_model.compute(self.s)
        s1, r, _, _ = self.env.step(a)
        if self.is_learning:
            self.learning_model.training(self.s, a, r, s1)
        self.s = s1
        theta = np.arctan2(s1[1], s1[0])
        omega = s1[2]
        moment = a
        return theta, omega, moment, r

    def reset_env(self):
        self.s = self.env.reset()

    def save_nn(self, filename):
        self.learning_model.save_to_file(filename)

    def load_nn(self, filename):
        self.learning_model.load_from_file(filename)

    def new_nn(self):
        self.learning_model.reset_nn()

    def start(self):
        QtWidgets.qApp.processEvents()