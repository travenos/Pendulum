from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QTextBrowser, QSpinBox, QDoubleSpinBox, \
    QRadioButton, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from matplotlib import rc

# from simulation import PendulumLearning


class MainWindow(QWidget):
    """Окно графического интерфейса"""

    def __init__(self):
        super().__init__()

        # self.sim = PendulumLearning(self)  # Объект с моделью и обучающейся системой управления маятником

        # Элементы интерфейса
        self.canvas_width = 800  # Ширина холста
        self.canvas_height = 675  # Длина холста
        self.canvas = Plotter(self.canvas_width, self.canvas_height, dpi=100, parent=self)  # Холст с графиками
        # Кнопки
        self.start_button = QPushButton('Старт', self)
        self.stop_button = QPushButton('Стоп', self)
        self.restart_button = QPushButton('Рестарт', self)
        self.new_nn_button = QPushButton('Новая нейросеть', self)
        self.load_nn_button = QPushButton('Загрузить нейросеть', self)
        self.save_nn_button = QPushButton('Сохранить нейросеть', self)
        self.eps_button = QPushButton('Узнать вероятность случайного действия', self)
        # Надписи
        self.eps_start_label = QLabel('Начальная вероятность\nслучайного действия', self)
        self.eps_discount_label = QLabel('Уменьшение вероятности\nна каждом шаге', self)
        self.eps_final_label = QLabel('Конечная вероятность\nслучайного действия', self)
        self.episode_length_label = QLabel('Длина эпизода', self)
        self.batch_size_label = QLabel('Размер обучающей\nпачки', self)
        # Поля ввода чисел
        self.eps_start_spinbox = QDoubleSpinBox(self)
        self.eps_discount_spinbox = QDoubleSpinBox(self)
        self.eps_final_spinbox = QDoubleSpinBox(self)
        self.episode_length_spinbox = QSpinBox(self)
        self.batch_size_spinbox = QSpinBox(self)
        # Кнопки выбора обучающейся модели
        self.q_learning_rb = QRadioButton("Deep Q-learning", self)
        self.q_learning2_rb= QRadioButton("Q-learning, сеть с одним выходом", self)
        self.actor_critic_rb = QRadioButton("Actor-critic (DDPG)", self)
        # Чекбоксы
        self.eps_greedy_checkbox = QCheckBox("Использовать случайные действия", self)
        self.learning_checkbox = QCheckBox("Включить обучение", self)
        self.endless_episode_checkbox = QCheckBox("Бесконечный эпизод", self)
        # Вывод данных
        self.output_text_field = QTextBrowser(self)

        self.initUI()

    def initUI(self):
        """
        Формирование интерфейса
        """
        size = (1300, 715)  # Размер окна
        position = (100, 100)  # Начальная позиция окна
        canvas_position = (20, 20)  # Верхний левый угол холста
        buttons_indent = 10  # Расстояние между кнопками
        spinbox_indent = 8  # Расстояние между полями выбора чисел
        labels_indent = 5  # Расстояние между надписями
        # Параметры поля выбора начальной вероятности выбора случайного действия
        eps_start_min = 0
        eps_start_max = 1
        eps_start_step = 0.1
        eps_start_default_value = 0.5
        # Параметры поля выбора шага уменьшения вероятности случайного действия
        eps_discount_min = 0
        eps_discount_max = 1
        eps_discount_step = 0.000001
        eps_discount_default_value = 0.000008334
        eps_discount_decimals = 9
        # Параметры поля выбора конечной вероятности выбора случайного действия
        eps_final_min = 0
        eps_final_max = 1
        eps_final_step = 0.01
        eps_final_default_value = 0.05
        # Параметры поля выбора длины эпизода
        episode_length_min = 1
        episode_length_max = 2000
        episode_length_step = 100
        episode_length_default_value = 200
        # Параметры поля выбора размера пачки при обучении
        batch_size_min = 1
        batch_size_max = 300
        batch_size_step = 10
        batch_size_default_value = 50
        # Размер поля вывода данных
        text_field_size = (460, 150)

        buttons_left = canvas_position[0] + buttons_indent + self.canvas_width  # Координата левого края блока элементов управления
        buttons_up = canvas_position[1]  # Координата верхнего края блока элементов управления

        # Изменяем позицию и перемещаем окно
        self.resize(*size)
        self.move(*position)

        self.canvas.move(*canvas_position)

        # Кнопки старт, стоп, перезапуск, очистка
        button_up = buttons_up  # Верхняя позиция текущей кнопки
        self.start_button.resize(self.start_button.sizeHint())
        self.start_button.move(buttons_left, button_up)
        self.start_button.clicked.connect(self.start)

        buttons_distance = self.start_button.height() + buttons_indent  # Расстояние между верхними позициями кнопок
        spinbox_distance = self.eps_start_spinbox.height() + spinbox_indent
        button_up += buttons_distance

        self.stop_button.resize(self.start_button.sizeHint())
        self.stop_button.move(buttons_left, button_up)
        self.stop_button.clicked.connect(self.stop)
        button_up += buttons_distance

        self.restart_button.resize(self.start_button.sizeHint())
        self.restart_button.move(buttons_left, button_up)
        self.restart_button.clicked.connect(self.restart)

        # Элементы для настройки случайных действий
        # Координата левого края блока элементов для генерации случайной среды
        controls_left = buttons_left + self.start_button.width() + buttons_indent

        element_up = buttons_up  # Верхняя позиция текущего элемента управления
        self.eps_start_label.resize(self.eps_start_label.sizeHint())
        self.eps_start_label.move(controls_left, element_up)
        element_left = controls_left + self.eps_start_label.width() + buttons_indent
        self.eps_start_spinbox.move(element_left, element_up)
        self.eps_start_spinbox.setMinimum(eps_start_min)
        self.eps_start_spinbox.setMaximum(eps_start_max)
        self.eps_start_spinbox.setSingleStep(eps_start_step)
        self.eps_start_spinbox.setValue(eps_start_default_value)
        self.eps_start_spinbox.valueChanged.connect(self.change_start_eps)
        element_up += spinbox_distance

        self.eps_final_label.resize(self.eps_final_label.sizeHint())
        self.eps_final_label.move(controls_left, element_up)
        element_left = controls_left + self.eps_final_label.width() + buttons_indent
        self.eps_final_spinbox.move(element_left, element_up)
        self.eps_final_spinbox.setMinimum(eps_final_min)
        self.eps_final_spinbox.setMaximum(eps_final_max)
        self.eps_final_spinbox.setSingleStep(eps_final_step)
        self.eps_final_spinbox.setValue(eps_final_default_value)
        self.eps_final_spinbox.valueChanged.connect(self.change_final_eps)
        element_up += spinbox_distance

        self.eps_discount_label.resize(self.eps_discount_label.sizeHint())
        self.eps_discount_label.move(controls_left, element_up)
        element_left = controls_left + self.eps_discount_label.width() + buttons_indent
        self.eps_discount_spinbox.setDecimals(eps_discount_decimals)
        self.eps_discount_spinbox.resize(self.eps_discount_spinbox.sizeHint())
        self.eps_discount_spinbox.move(element_left, element_up)
        self.eps_discount_spinbox.setMinimum(eps_discount_min)
        self.eps_discount_spinbox.setMaximum(eps_discount_max)
        self.eps_discount_spinbox.setSingleStep(eps_discount_step)
        self.eps_discount_spinbox.setValue(eps_discount_default_value)
        self.eps_discount_spinbox.valueChanged.connect(self.change_eps_discount)
        element_up += spinbox_distance

        self.eps_greedy_checkbox.resize(self.eps_greedy_checkbox.sizeHint())
        self.eps_greedy_checkbox.move(controls_left, element_up)
        self.eps_greedy_checkbox.setChecked(True)
        self.eps_greedy_checkbox.stateChanged.connect(self.toggle_eps_greedy)

        labels_distance = self.eps_greedy_checkbox.height() + labels_indent

        element_up += labels_distance

        self.eps_button.resize(self.eps_button.sizeHint())
        self.eps_button.move(controls_left, element_up)
        self.eps_button.clicked.connect(self.get_eps)

        element_up += buttons_distance

        button_up = max([element_up, button_up])
        self.q_learning_rb.move(buttons_left, button_up)
        self.q_learning_rb.setChecked(True)
        self.q_learning_rb.toggled.connect(self.select_q_learning)
        button_up += labels_distance

        self.q_learning2_rb.move(buttons_left, button_up)
        self.q_learning2_rb.toggled.connect(self.select_q_learning2)
        button_up += labels_distance
        self.actor_critic_rb.move(buttons_left, button_up)
        self.actor_critic_rb.toggled.connect(self.select_actor_critic)
        button_up += labels_distance

        self.learning_checkbox.move(buttons_left, button_up)
        self.learning_checkbox.setChecked(True)
        self.learning_checkbox.stateChanged.connect(self.toggle_learning)

        button_up += labels_distance
        self.batch_size_label.resize(self.batch_size_label.sizeHint())
        self.batch_size_label.move(buttons_left, button_up)
        element_left = buttons_left + self.batch_size_label.width() + buttons_indent
        self.batch_size_spinbox.resize(self.batch_size_spinbox.sizeHint())
        self.batch_size_spinbox.move(element_left, button_up)
        self.batch_size_spinbox.setMinimum(batch_size_min)
        self.batch_size_spinbox.setMaximum(batch_size_max)
        self.batch_size_spinbox.setSingleStep(batch_size_step)
        self.batch_size_spinbox.setValue(batch_size_default_value)
        self.batch_size_spinbox.valueChanged.connect(self.change_batch_size)
        button_up += spinbox_distance

        self.new_nn_button.resize(self.save_nn_button.sizeHint())
        self.new_nn_button.move(buttons_left, button_up)
        self.new_nn_button.clicked.connect(self.new_nn)
        button_up += buttons_distance

        self.load_nn_button.resize(self.save_nn_button.sizeHint())
        self.load_nn_button.move(buttons_left, button_up)
        self.load_nn_button.clicked.connect(self.load_nn)
        button_up += buttons_distance

        self.save_nn_button.resize(self.save_nn_button.sizeHint())
        self.save_nn_button.move(buttons_left, button_up)
        self.save_nn_button.clicked.connect(self.save_nn)
        button_up += buttons_distance

        self.episode_length_label.resize(self.episode_length_label.sizeHint())
        self.episode_length_label.move(buttons_left, button_up)
        element_left = buttons_left + self.episode_length_label.width() + buttons_indent
        self.episode_length_spinbox.resize(self.episode_length_spinbox.sizeHint())
        self.episode_length_spinbox.move(element_left, button_up)
        self.episode_length_spinbox.setMinimum(episode_length_min)
        self.episode_length_spinbox.setMaximum(episode_length_max)
        self.episode_length_spinbox.setSingleStep(episode_length_step)
        self.episode_length_spinbox.setValue(episode_length_default_value)
        self.episode_length_spinbox.valueChanged.connect(self.change_episode_length)
        button_up += spinbox_distance

        self.endless_episode_checkbox.move(buttons_left, button_up)
        self.endless_episode_checkbox.setChecked(False)
        self.endless_episode_checkbox.stateChanged.connect(self.toggle_endless_episode)
        button_up += labels_distance

        self.output_text_field.resize(*text_field_size)
        self.output_text_field.move(buttons_left, button_up)
        self.output_text_field.setEnabled(False)
        self.setWindowTitle('Pendulum')
        self.show()

    def start(self):
        """
        Запуск моделирования
        """
        pass
        #self.sim.start()

    def stop(self):
        pass
        #self.sim.stop()

    def restart(self):
        pass
        #self.sim.stop()

    def change_start_eps(self):
        pass

    def change_final_eps(self):
        pass

    def change_eps_discount(self):
        pass

    def select_q_learning(self):
        pass

    def select_q_learning2(self):
        pass

    def select_actor_critic(self):
        pass

    def toggle_eps_greedy(self):
        pass

    def toggle_learning(self):
        pass

    def get_eps(self):
        pass

    def change_batch_size(self):
        pass
    
    def new_nn(self):
        pass
    
    def load_nn(self):
        pass
    
    def save_nn(self):
        pass

    def change_episode_length(self):
        pass

    def toggle_endless_episode(self):
        pass

    def paint_scene(self):
        """
        Рисовать на холсте цилиндр и роботов
        """
        # TODO Рисовать на холсте цилиндр и роботов
        pass


class Plotter(FigureCanvas):
    def __init__(self, width, height, dpi, parent=None):
        font = {'family': 'Verdana',
                'weight': 'normal'}
        rc('font', **font)
        scale_const = 100
        self.figure = Figure(figsize=(width/scale_const, height/scale_const), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        self.theta_plot = self.figure.add_subplot(311)
        self.theta_plot.set_title("Угол отклонения маятника от вертикальной оси")
        self.theta_plot.set_ylabel('θ, рад')
        self.theta_plot.grid(True)

        self.omega_plot = self.figure.add_subplot(312)
        self.omega_plot.set_title("Угловая скорость маятника")
        self.omega_plot.set_ylabel('ω, рад/с')
        self.omega_plot.grid(True)

        self.moment_plot = self.figure.add_subplot(313)
        self.moment_plot.set_title("Внешний момент силы, воздействующий на маятник")
        self.moment_plot.set_xlabel('t, с')
        self.moment_plot.set_ylabel('M, Н•м')
        self.moment_plot.grid(True)

        self.figure.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.425, wspace=0.35)

    # def set_title(self, title):
    #     self.ax.set_title(title)
    #
    # def plot(self, data=None, color='b'):
    #     if data is None:
    #         self.ax.plot()
    #     else:
    #         self.ax.plot(data, color)
    #     self.draw()
    #
    # def draw_circle(self, center=(0,0), radius=1, color='b', fill=False, rescale=None):
    #     circle = matplotlib.patches.Circle(center, radius=radius, color=color, fill=fill)
    #     self.ax.add_patch(circle)
    #     if rescale is not None:
    #         self.ax.set_xlim(xmin=center[0]-rescale*radius, xmax=center[0]+rescale*radius)
    #         self.ax.set_ylim(ymin=center[1]-rescale*radius, ymax=center[1]+rescale*radius)
