import numpy as np
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense
from collections import deque


class LearningModel(object):

    def __init__(self, inputs_n=None, neurons_n=(1,), activations=("linear",)):
        """Конструктор"""
        self.batch_size = 4  # Размер мини-выборки
        self.second_net_train_period = 1  # Период обновления весов в "запаздывающей" сети
        self.old_results_size = 3  # Число старых опытов в мини-выборке
        self.max_replay_memory_size = 100000  # Максимальный размер истории опыта

        self.replay_memory = deque(maxlen=self.max_replay_memory_size)  # История опыта

        self.batch_x = np.array([])  # Пачка значений для входа сети
        self.batch_y = np.array([])  # Пачка целевых выходов сети

        self.counter1 = 0  # Счётчик итераций для обновления весов в первой сети
        self.counter2 = 0  # Счётчик итераций для обновления весов в "запаздывающей" сети

        self.gamma = 0.8  # Discount factor

        self.eps = 0.8
        self.EPS_DISCOUNT = 0.99996
        self.EPS_DISCOUNT2 = 0.000008334

        self.neuralNet = Sequential()  # Нейронная сеть
        self.neuralNet2 = Sequential()  # Нейронная сеть для обучения с запаздыванием

        if inputs_n is None:
            return
        self.new_nn(inputs_n, neurons_n, activations)

    def new_nn(self, inputs_n, neurons_n, activations):
        """Создать новую нейронную сеть"""
        if type(inputs_n) != int:
            raise TypeError("inputs_n should be an integer")
        if inputs_n < 1:
            raise ValueError("inputs_n should be positive")
        if len(neurons_n) != len(activations):
            raise ValueError("neurons_n and activations should have the same length")
        del self.neuralNet
        del self.neuralNet2
        self.batch_x = np.zeros((self.batch_size, inputs_n))  # Пачка значений для входа сети
        self.batch_y = np.zeros((self.batch_size, neurons_n[-1]))  # Пачка целевых выходов сети
        self.counter1 = 0  # Счётчик итераций для обновления весов в первой сети
        self.counter2 = 0  # Счётчик итераций для обновления весов в "запаздывающей" сети
        self.replay_memory = deque(maxlen=self.max_replay_memory_size)
        self.neuralNet = Sequential()  # Нейронная сеть
        for i, (n, af) in enumerate(zip(neurons_n, activations)):  # Добавляем слои
            if i == 0:
                self.neuralNet.add(Dense(n, input_dim=inputs_n, kernel_initializer="normal", activation=af))
            else:
                self.neuralNet.add(Dense(n, kernel_initializer="normal", activation=af))
        self.neuralNet.compile(loss="mean_squared_error", optimizer="SGD", metrics=["mae"])   # Компилируем сеть
        # Используется стохастический градиентный метод оптимизации
        self.neuralNet2 = clone_model(self.neuralNet)  # Нейронная сеть для обучения с запаздыванием
        self.neuralNet2.set_weights(self.neuralNet.get_weights())

    def save_to_file(self, filename):
        """Сохранение сети вместе с весами в файл"""
        self.neuralNet.save(filename)

    def load_from_file(self, filename):
        """Загрузка сети вместе с весами из файла"""
        del self.neuralNet
        del self.neuralNet2
        self.neuralNet = load_model(filename)
        self.neuralNet2 = clone_model(self.neuralNet)  # Нейронная сеть для обучения с запаздыванием
        self.neuralNet2.set_weights(self.neuralNet.get_weights())

        self.batch_x = np.zeros((self.batch_size, self.get_inputs_count()))  # Пачка значений для входа сети
        self.batch_y = np.zeros((self.batch_size, self.get_outputs_count()))  # Пачка целевых выходов сети
        return self

    def compute_batch(self, sa):
        """Обработка каждого элемента из массива"""
        return list(map(self.action_weights, self.neuralNet.predict(sa)))

    def compute(self, s):
        """Обработка одного элемента"""
        if type(s) != np.ndarray:
            raise TypeError("s should be a numpy array")
        if len(s) != self.get_inputs_count():
            raise ValueError("Length of s should be equal to number of inputs")
        sa = np.array([s])
        qa = self.compute_batch(sa)
        a = qa[0]
        return a

    def action_weights(self, q):
        """Вычисление вектора действия (one-hot-encoding) по вектору q-занчений"""
        if type(q) != np.ndarray:
            raise TypeError("q should be a numpy array")
        a = np.zeros_like(q)
        a[q.argmax()] = 1
        # С вероятностью self.eps исследуем случайное действие
        rnd = np.random.sample(1)
        if rnd < self.eps:
            a = np.zeros_like(a)
            a[np.random.randint(0, len(a))] = 1
        # self.eps *= self.EPS_DISCOUNT
        self.eps -= self.EPS_DISCOUNT2

        assert np.all(0 <= a)
        assert np.all(a <= 1)
        assert np.any(a > 0)
        assert a.sum() == 1
        return a

    def get_q(self, s):
        """Получить значения Q-функции для состояния s"""
        sa = np.array([s])
        qa = self.neuralNet.predict(sa)
        q = qa[0]
        return q

    def get2_q(self, s):
        """Получить значения Q-функции для состояния s из сети с обучением с запаздыванием"""
        sa = np.array([s])
        qa = self.neuralNet2.predict(sa)
        q = qa[0]
        return q

    def update_nn2(self):
        """Перенести веса из обучаемой нейронной сети """
        self.neuralNet2.set_weights(self.neuralNet.get_weights())

    def training(self, s, a, r, s1):
        """
        Обучение нейронной сети.
        :param s: вектор предыдущего состояния
        :param a: вектор действия (one-hot-encoding)
        :param r: скалярная награда, полученная при переходе из s в s1
        :param s1:  вектор нового состояния
        """
        if type(s) != np.ndarray:
            raise TypeError("s should be a numpy array")
        if type(a) != np.ndarray:
            raise TypeError("a should be a numpy array")
        if type(r) == np.ndarray or type(r) == list or type(r) == tuple:
            raise TypeError("r should be a scalar")
        if type(s1) != np.ndarray:
            raise TypeError("s1 should be a numpy array")
        self.replay_memory.append([s, a, r, s1])
        self.batch_x[self.counter1] = s
        self.batch_y[self.counter1] = self.__calc_target(s, a, r, s1)
        self.counter1 += 1
        self.counter2 += 1
        if self.counter1 >= self.batch_size - self.old_results_size:    # Обновляем веса у первой сети
            if len(self.replay_memory) == self.counter1:    # Если в памяти ещё нет данных с предыдущих опытов
                self.neuralNet.fit(self.batch_x[:self.counter1], self.batch_y[:self.counter1],
                                   batch_size=int(self.counter1*0.4)+1,
                                   epochs=1, shuffle=True, verbose=0)  # Обучение сети
            else:   # Добавляем данные для старых ситуаций
                for i in range(0, self.old_results_size):
                    rnd = np.random.randint(0, len(self.replay_memory) - self.counter1)
                    old_result = self.replay_memory[rnd]
                    self.batch_x[self.counter1 + i] = old_result[0]
                    self.batch_y[self.counter1 + i] = self.__calc_target(*old_result)
                self.neuralNet.fit(self.batch_x, self.batch_y,
                                   batch_size=int(self.batch_size*0.4)+1, epochs=1, shuffle=True,
                                   verbose=0)  # Обучение сети
            self.counter1 = 0   # Обнуляем счётчик
        if self.counter2 == self.second_net_train_period:   # Обновляем веса у второй сети
            self.update_nn2()
            self.counter2 = 0  # Обнуляем счётчик

    def get_inputs_count(self):
        """Количество входов сети"""
        cfg = self.neuralNet.get_config()
        return cfg[0]['config']['batch_input_shape'][1]

    def get_outputs_count(self):
        """Количество выходов сети"""
        cfg = self.neuralNet.get_config()
        return cfg[-1]['config']['units']

    def __calc_target(self, s, a, r, s1):
        """Вычисление данных, подаваемых на выход нейронной сети (согласно уравнению Беллмана)"""
        q = self.get_q(s)
        max_q1 = max(self.get2_q(s1))
        alpha = 1
        return q + alpha * a * (r + self.gamma * max_q1 - q)
