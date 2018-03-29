import gym
import numpy as np
import ctypes
from PyQt5 import QtWidgets

from q_learning import QLearningModel
from q_learning2 import QLearningModel2
from actor_critic import ActorCritic


class PendulumLearning(object):
    def __init__(self, GUI):
        self.GUI = GUI
        self.env = self.new_env()

        self._S_LEN = self.env.observation_space.shape[0]
        self._A_BOUND = self.env.action_space.high

        # Используемые модели
        self.learning_models_list = [QLearningModel(state_len=self._S_LEN, action_len=1, a_bound=self._A_BOUND),
                                     QLearningModel2(state_len=self._S_LEN, action_len=1, a_bound=self._A_BOUND),
                                     ActorCritic(state_len=self._S_LEN, action_len=1, a_bound=self._A_BOUND)]

        self.learning_model = self.learning_models_list[0]  # Ссылка на применяемую модель

        self.s = np.zeros(self._S_LEN)   # Предыдущее состояние маятника
        self.reset_env()    # Перезапустить маятник

        self.is_learning = True
        self.working = False
        self.endless = False

        self.max_ep_steps = 200

        self.steps = 0

    @staticmethod
    def new_env():
        env = gym.make('Pendulum-v0')
        env = env.unwrapped
        env.seed(1)  # reproducible
        return env

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
        if self.working:
            return
        self.working = True
        self._simulation_process()

    def _simulation_process(self):
        self.steps = 0
        t = 0
        thetas = []
        omegas = []
        moments = []
        times = []
        running_reward = None
        sum_reward = 0
        episodes = 0
        while self.working:
            try:
                theta, omega, moment, r = self.step()
                thetas.append(theta)
                omegas.append(omega)
                moments.append(moment)
                times.append(t)
                t += self.env.dt
                sum_reward += r
            except ctypes.ArgumentError:  # Закрытие пользователем окна с маятником
                self.env.close()
                self.env = self.new_env()
                self.reset_env()
                self.stop()
            else:
                QtWidgets.qApp.processEvents()
                self.steps += 1
                if self.steps >= self.max_ep_steps:  # Завершение эпизода
                    episodes += 1
                    if running_reward is None:
                        running_reward = sum_reward / self.steps * 20
                    else:
                        running_reward = running_reward * 0.9 + sum_reward / self.steps * 2
                    self.GUI.paint_scene(thetas, omegas, moments, times, running_reward, episodes)
                    t = 0
                    thetas.clear()
                    omegas.clear()
                    moments.clear()
                    times.clear()
                    sum_reward = 0
                    self.steps = 0
                    if not self.endless:  # Если эпизод в среде не бесконечный
                        self.reset_env()

    def stop(self):
        self.steps = 0
        self.working = False

    def exit(self):
        self.working = False
        for model in self.learning_models_list:
            model.close_session()
        self.env.close()

    def restart(self):
        if self.working:
            self.steps = self.max_ep_steps
        else:
            self.reset_env()

    def set_eps(self, eps):
        for model in self.learning_models_list:
            model.eps = eps

    def set_final_eps(self, eps):
        for model in self.learning_models_list:
            model.MIN_EPS = eps

    def set_eps_discount(self, eps_discount):
        for model in self.learning_models_list:
            model.EPS_DISCOUNT = eps_discount

    def get_eps(self):
        if not self.learning_model.EPS_GREEDY:
            return 0
        return self.learning_model.eps

    def set_eps_greedy(self, enable):
        for model in self.learning_models_list:
            model.EPS_GREEDY = enable

    def set_batch_size(self, batch_size):
        for model in self.learning_models_list:
            model.BATCH_SIZE = batch_size

