import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from collections import deque


class QLearningModel(object):
    """Deep Q-learning"""
    def __init__(self, state_len=3, action_len=1, a_bound=2, actions=2, neurons_n=(30, 14),
                 activations=("relu", "sigmoid")):
        """Конструктор"""
        self.lr = 0.003  # Темп обучения нейросети
        self._state_len = state_len  # Длина вектора состояния
        self._action_len = action_len  # Длина вектора действия
        self._construct_net(state_len, action_len, actions, neurons_n, activations)  # Создать нейронную сеть
        if action_len == 1:
            self._alphabet = np.linspace(-a_bound, a_bound, actions)  # Алфавит действий
            self._alphabet.resize((self._alphabet.size, 1))
        else:
            raise NotImplementedError("Implemented only for 1-coordinate control signals")

        self.EPS_GREEDY = True  # Make random actions sometimes
        self.eps = 0.5  # Initial probability of random action
        self.EPS_DISCOUNT = 0.000008334  # By this value probability of random action is decreased by every step
        self.MIN_EPS = 0.05  # Minimum probability of random action
        self.BATCH_SIZE = 50  # Size of training batch on every step

        self.max_replay_memory_size = 100000  # Максимальный размер истории опыта
        self.replay_memory = deque(maxlen=self.max_replay_memory_size)  # История опыта

        self.GAMMA = 0.85  # Discount factor

    def _construct_net(self, state_len, action_len, actions, neurons_n, activations):
        """Создать нейронную сеть для заданных параметров"""
        self._inputs_n = state_len
        self._neurons_n = list(neurons_n)
        self._neurons_n.append(actions)
        self._activations = list(activations)
        self._activations.append("linear")
        self.neuralNet = Sequential()  # Нейронная сеть
        self._new_nn(self._inputs_n, self._neurons_n, self._activations)

    def _new_nn(self, inputs_n, neurons_n, activations):
        """Создать новую нейронную сеть"""
        if type(inputs_n) != int:
            raise TypeError("inputs_n should be an integer")
        if inputs_n < 1:
            raise ValueError("inputs_n should be positive")
        if len(neurons_n) != len(activations):
            raise ValueError("neurons_n and activations should have the same length")
        del self.neuralNet
        self.neuralNet = Sequential()  # Нейронная сеть
        for i, (n, af) in enumerate(zip(neurons_n, activations)):  # Добавляем слои
            if i == 0:
                self.neuralNet.add(Dense(n, input_dim=inputs_n, kernel_initializer="normal", activation=af))
            else:
                self.neuralNet.add(Dense(n, kernel_initializer="normal", activation=af))
        optimizer = optimizers.adam(lr=self.lr)
        self.neuralNet.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])   # Компилируем сеть
        # Используется метод оптимизации Adam

    def reset_nn(self):
        """Создать новую нейронную сеть с исходными параметрами"""
        self._new_nn(self._inputs_n, self._neurons_n, self._activations)

    def save_to_file(self, filename):
        """Сохранение сети вместе с весами в файл"""
        self.neuralNet.save(filename)

    def load_from_file(self, filename):
        """Загрузка сети вместе с весами из файла"""
        new_net = load_model(filename)
        if self.get_inputs_count(new_net) != self._inputs_n or self.get_outputs_count(new_net) != self._neurons_n[-1]:
            raise OSError("Unable to open file (wrong net configuration)")
        else:
            del self.neuralNet
            self.neuralNet = new_net
        return self

    def compute_batch_ohe(self, s_batch):
        """Обработка каждого элемента из массива"""
        return self.action_weights(self.neuralNet.predict(s_batch))

    def compute(self, s):
        """Обработка одного элемента"""
        if type(s) != np.ndarray:
            raise TypeError("s should be a numpy array")
        if len(s) != self._state_len:
            raise ValueError("Incorrect length of state vector")
        # С вероятностью self.eps исследуем случайное действие
        rnd = np.random.sample()
        if self.EPS_GREEDY and rnd < self.eps:
            a = np.zeros(len(self._alphabet))
            a[np.random.randint(0, len(a))] = 1
        else:
            s_batch = np.array([s])
            q_alph_batch = self.compute_batch_ohe(s_batch)
            a = q_alph_batch[0]
        if self.EPS_GREEDY:
            self.eps -= self.EPS_DISCOUNT  # Уменьшение вероятности случайного действия
            if self.eps < self.MIN_EPS:
                self.eps = self.MIN_EPS
        return self._alphabet[a.argmax()]

    @staticmethod
    def action_weights(q_alph_batch):
        """Вычисление вектора действия (one-hot-encoding) по вектору q-занчений"""
        if type(q_alph_batch) != np.ndarray:
            raise TypeError("q should be a numpy array")
        a_batch = np.zeros_like(q_alph_batch)
        for a_str, q_str in zip(a_batch, q_alph_batch):
            a_str[q_str.argmax()] = 1
        assert np.all(0 <= a_batch)
        assert np.all(a_batch <= 1)
        assert np.any(a_batch > 0)
        return a_batch

    def get_q(self, s):
        """Получить значения Q-функции для состояния s"""
        s_batch = np.array([s])
        q_alph_batch = self.neuralNet.predict(s_batch)
        q_alph = q_alph_batch[0]
        return q_alph

    def get_q_batch(self, s_batch):
        """Получить значения Q-функций для каждого состояния из пачки s_batch и каждого действия из алфавита"""
        return self.neuralNet.predict(s_batch)

    def training(self, s, a, r, s1):
        """
        Обучение нейронной сети.
        :param s: вектор предыдущего состояния
        :param a: действие
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

        # Перевод действия в one-hot-encoding
        a_mhe = np.float64(self._alphabet == a)
        a_ohe = a_mhe[:, 0].copy()
        for i in range(1, a_mhe.shape[1]):
            a_ohe *= a_mhe[:, i]
        self.replay_memory.append([s, a_ohe, r, s1])  # Добавление нового опыта в историю
        if len(self.replay_memory) < self.BATCH_SIZE:  # Если недостаточно опыта в истории, тренировки не происходит
            return

        batch_indexes = set()  # Случайные индексы элементов истории опыта
        batch_indexes.add(len(self.replay_memory)-1)
        while len(batch_indexes) < self.BATCH_SIZE:
            rnd = np.random.randint(0, len(self.replay_memory) - 1)
            batch_indexes.add(rnd)
        batch_indexes = list(batch_indexes)
        np.random.shuffle(batch_indexes)

        # Пачки векторов для тренировки
        sb = np.zeros((self.BATCH_SIZE, self.get_inputs_count()))
        ab = np.zeros((self.BATCH_SIZE, self.get_outputs_count()))
        rb = np.zeros((self.BATCH_SIZE, 1))
        s1b = np.zeros((self.BATCH_SIZE, self.get_inputs_count()))

        for i, index in enumerate(batch_indexes):
            sb[i, :] = self.replay_memory[index][0]
            ab[i, :] = self.replay_memory[index][1]
            rb[i, :] = self.replay_memory[index][2]
            s1b[i, :] = self.replay_memory[index][3]

        yb = self.__calc_target_batch(sb, ab, rb, s1b)
        self.neuralNet.fit(sb, yb, batch_size=self.BATCH_SIZE, epochs=1, shuffle=True, verbose=0)  # Обучение сети

    def get_inputs_count(self, net=None):
        """Количество входов сети"""
        if net is None:
            net = self.neuralNet
        cfg = net.get_config()
        try:
            return cfg['layers'][0]['config']['batch_input_shape'][1]
        except KeyError:
            return cfg[0]['config']['batch_input_shape'][1]

    def get_outputs_count(self, net=None):
        """Количество выходов сети"""
        if net is None:
            net = self.neuralNet
        cfg = net.get_config()
        try:
            return cfg['layers'][-1]['config']['units']
        except KeyError:
            return cfg[-1]['config']['units']

    def __calc_target_batch(self, s_batch, a_batch, r_batch, s1_batch):
        """Вычисление данных, подаваемых на выход нейронной сети (согласно уравнению Беллмана)"""
        Q = self.get_q_batch(s_batch)
        max_Q1 = np.max(self.get_q_batch(s1_batch), axis=1)
        max_Q1 = max_Q1.reshape((max_Q1.size, 1))
        alpha = 1
        return Q + alpha * a_batch * (r_batch + self.GAMMA * max_Q1 - Q)

    def clear_memory(self):
        """Очистить память опыта"""
        self.replay_memory.clear()

    def close_session(self):
        """Закрыть сессию (для унификации с классами, использующими TensorFlow)"""
        pass
