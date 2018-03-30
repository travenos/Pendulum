import numpy as np
from keras.models import Sequential

from q_learning import QLearningModel


class QLearningModel2(QLearningModel):
    """Q-learning. На вход нейросети подаются и состояние, и действие. У сети один выход"""
    def __init__(self, state_len=3, action_len=1, a_bound=2, actions=2, neurons_n=(30, 14),
                 activations=("relu", "sigmoid")):
        """Конструктор"""
        super().__init__(state_len, action_len, a_bound, actions, neurons_n, activations)

    def _construct_net(self, state_len, action_len, actions, neurons_n, activations):
        """Создать нейронную сеть для заданных параметров"""
        self._inputs_n = state_len + action_len
        self._neurons_n = list(neurons_n)
        self._neurons_n.append(1)
        self._activations = list(activations)
        self._activations.append("linear")
        self.neuralNet = Sequential()  # Нейронная сеть
        self._new_nn(self._inputs_n, self._neurons_n, self._activations)

    def compute_batch_ohe(self, s_batch):
        """Обработка каждого элемента из массива"""
        q_alph_batch = self.get_q_batch(s_batch)
        return self.action_weights(q_alph_batch)

    @staticmethod
    def action_weights(q):
        """Вычисление вектора действия (one-hot-encoding) по вектору q-занчений"""
        if type(q) != np.ndarray:
            raise TypeError("q should be a numpy array")
        a = np.zeros_like(q)
        for a_str, q_str in zip(a, q):
            a_str[q_str.argmax()] = 1
        assert np.all(0 <= a)
        assert np.all(a <= 1)
        assert np.any(a > 0)
        return a

    def get_q(self, s):
        """Получить значения Q-функции для состояния s"""
        s_batch = np.array([s])
        q_alph_batch = self.get_q_batch(s_batch)
        q_alph = q_alph_batch[0]
        return q_alph

    def get_q_batch(self, s_batch):
        """Получить значения Q-функций для каждого состояния из пачки s_batch и каждого действия из алфавита"""
        sa_batch = []
        for s in s_batch:
            for a in self._alphabet:
                sa_batch.append(np.concatenate((s, a)))
        sa_batch = np.array(sa_batch)
        q_alph_batch = self.neuralNet.predict(sa_batch)
        q_alph_batch.resize((int(q_alph_batch.size / self._alphabet.size), self._alphabet.size))
        return q_alph_batch
    
    def get_actions_q_batch(self, s_batch, a_batch):
        """Получить значения Q-функций для каждой ситуации (состояние из s_batch, действие из a_batch)"""
        sa_batch = np.concatenate((s_batch, a_batch), axis=1)
        q_batch = self.neuralNet.predict(sa_batch)
        return q_batch

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

        self.replay_memory.append([s, a, r, s1])  # Добавление нового опыта в историю
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
        sb = np.zeros((self.BATCH_SIZE, self._state_len))
        ab = np.zeros((self.BATCH_SIZE, self._action_len))
        rb = np.zeros((self.BATCH_SIZE, 1))
        s1b = np.zeros((self.BATCH_SIZE, self._state_len))

        for i, index in enumerate(batch_indexes):
            sb[i, :] = self.replay_memory[index][0]
            ab[i, :] = self.replay_memory[index][1]
            rb[i, :] = self.replay_memory[index][2]
            s1b[i, :] = self.replay_memory[index][3]

        yb = self.__calc_target_batch(sb, ab, rb, s1b)  # Пачка выходов нейросети
        xb = np.concatenate((sb, ab), axis=1)   # Пачка входов нейросети
        self.neuralNet.fit(xb, yb, batch_size=self.BATCH_SIZE, epochs=1, shuffle=True, verbose=0)  # Обучение сети

    def __calc_target_batch(self, s_batch, a_batch, r_batch, s1_batch):
        """Вычисление данных, подаваемых на выход нейронной сети (согласно уравнению Беллмана)"""
        Q = self.get_actions_q_batch(s_batch, a_batch)
        max_Q1 = np.max(self.get_q_batch(s1_batch), axis=1)
        max_Q1 = max_Q1.reshape((max_Q1.size, 1))
        alpha = 1
        return Q + alpha * (r_batch + self.GAMMA * max_Q1 - Q)
