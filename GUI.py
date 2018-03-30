from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QTextBrowser, QSpinBox, QDoubleSpinBox, \
    QRadioButton, QCheckBox, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rc

from simulation import PendulumLearning


class MainWindow(QWidget):
    """Окно графического интерфейса"""

    def __init__(self):
        super().__init__()

        self.sim = PendulumLearning(self)  # Объект с моделью и обучающейся системой управления маятником

        # Элементы интерфейса
        self.canvas_width = 800  # Ширина холста
        self.canvas_height = 675  # Длина холста
        self.canvas = Plotter(self.canvas_width, self.canvas_height, dpi=100, parent=self)  # Холст с графиками
        # Кнопки
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.restart_button = QPushButton('Restart', self)
        self.new_nn_button = QPushButton('New neural net', self)
        self.load_nn_button = QPushButton('Load neural net', self)
        self.save_nn_button = QPushButton('Save neural net', self)
        self.save_plots_button = QPushButton('Save plots', self)
        self.clear_button = QPushButton('Clear', self)
        # Надписи
        self.eps_start_label = QLabel('Initial probability\nof random action', self)
        self.eps_discount_label = QLabel('Discount of probability\non each step', self)
        self.eps_final_label = QLabel('Final probability\nof random action', self)
        self.episode_length_label = QLabel('Episode length', self)
        self.batch_size_label = QLabel('Training batch\nsize', self)
        # Поля ввода чисел
        self.eps_start_spinbox = QDoubleSpinBox(self)
        self.eps_discount_spinbox = QDoubleSpinBox(self)
        self.eps_final_spinbox = QDoubleSpinBox(self)
        self.episode_length_spinbox = QSpinBox(self)
        self.batch_size_spinbox = QSpinBox(self)
        # Кнопки выбора обучающейся модели
        self.q_learning_rb = QRadioButton("Deep Q-learning", self)
        self.q_learning2_rb= QRadioButton("Q-learning, net with one output", self)
        self.actor_critic_rb = QRadioButton("Actor-critic (DDPG)", self)
        # Чекбоксы
        self.eps_greedy_checkbox = QCheckBox("Use random actions", self)
        self.learning_checkbox = QCheckBox("Turn training on", self)
        self.endless_episode_checkbox = QCheckBox("Endless episode", self)
        # Вывод данных
        self.output_text_field = QTextBrowser(self)

        self.initUI()

    def initUI(self):
        """Формирование интерфейса"""
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
        text_field_size = (460, 190)

        buttons_left = canvas_position[0] + buttons_indent + self.canvas_width  # Координата левого края блока элементов управления
        buttons_up = canvas_position[1]  # Координата верхнего края блока элементов управления

        # Изменяем позицию и перемещаем окно
        self.resize(*size)
        self.move(*position)

        self.canvas.move(*canvas_position)

        # Кнопки старт, стоп, перезапуск
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
        self.change_start_eps()
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
        self.change_final_eps()
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
        self.change_eps_discount()
        element_up += spinbox_distance

        self.eps_greedy_checkbox.resize(self.eps_greedy_checkbox.sizeHint())
        self.eps_greedy_checkbox.move(controls_left, element_up)
        self.eps_greedy_checkbox.setChecked(True)
        self.eps_greedy_checkbox.stateChanged.connect(self.toggle_eps_greedy)
        self.toggle_eps_greedy()

        labels_distance = self.eps_greedy_checkbox.height() + labels_indent

        element_up += labels_distance

        button_up = max([element_up, button_up])
        self.q_learning_rb.move(buttons_left, button_up)
        self.q_learning_rb.setChecked(True)
        self.q_learning_rb.toggled.connect(self.select_learning_model)
        button_up += labels_distance
        self.q_learning2_rb.move(buttons_left, button_up)
        self.q_learning2_rb.toggled.connect(self.select_learning_model)
        button_up += labels_distance
        self.actor_critic_rb.move(buttons_left, button_up)
        self.actor_critic_rb.toggled.connect(self.select_learning_model)
        self.select_learning_model()
        button_up += labels_distance

        self.learning_checkbox.move(buttons_left, button_up)
        self.learning_checkbox.setChecked(True)
        self.learning_checkbox.stateChanged.connect(self.toggle_learning)
        self.toggle_learning()

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
        self.change_batch_size()
        button_up += spinbox_distance

        element_up = button_up

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

        elements_left = buttons_left + self.save_nn_button.width() + buttons_indent

        self.save_plots_button.resize(self.save_plots_button.sizeHint())
        self.save_plots_button.move(elements_left, element_up)
        self.save_plots_button.clicked.connect(self.save_plots)
        element_up += buttons_distance

        self.clear_button.resize(self.save_plots_button.sizeHint())
        self.clear_button.move(elements_left, element_up)
        self.clear_button.clicked.connect(self.clear)
        element_up += buttons_distance

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
        self.change_episode_length()
        button_up += spinbox_distance

        self.endless_episode_checkbox.move(buttons_left, button_up)
        self.endless_episode_checkbox.setChecked(False)
        self.endless_episode_checkbox.stateChanged.connect(self.toggle_endless_episode)
        self.toggle_endless_episode()
        button_up += labels_distance

        self.output_text_field.resize(*text_field_size)
        self.output_text_field.move(buttons_left, button_up)
        self.output_text_field.setReadOnly(True)
        self.setWindowTitle('The Reverse Pendulum')
        self.show()

    def closeEvent(self, event):
        """Закрытие окна"""
        self.sim.exit()
        self.deleteLater()
        self.close()

    def start(self):
        """Запуск моделирования"""
        self.sim.start()

    def stop(self):
        """Остановка моделирования"""
        self.sim.stop()

    def restart(self):
        """Поместить маятник в случайную позицию"""
        self.sim.restart()

    def change_start_eps(self):
        """Изменить начальную вероятность случайного действия"""
        self.sim.set_eps(self.eps_start_spinbox.value())

    def change_final_eps(self):
        """Изменить конечную вероятность случайного действия"""
        self.sim.set_final_eps(self.eps_final_spinbox.value())

    def change_eps_discount(self):
        """Изменить шаг уменьшения вероятности случайного действия"""
        self.sim.set_eps_discount(self.eps_discount_spinbox.value())

    def select_learning_model(self):
        """Выбрать обучающуюся модель"""
        if self.q_learning_rb.isChecked():
            lm_number = 0
        elif self.q_learning2_rb.isChecked():
            lm_number = 1
        else:
            lm_number = 2
        self.sim.choose_learning_model(lm_number)

    def toggle_eps_greedy(self):
        """Включить или отключить совершение случайных действий"""
        self.sim.set_eps_greedy(self.eps_greedy_checkbox.isChecked())

    def toggle_learning(self):
        """Производить ли обучение модели, или просто включить управление"""
        enable = self.eps_greedy_checkbox.isChecked()
        self.sim.is_learning = enable

    def save_plots(self):
        """Сохранить графики в файл"""
        file_dialogue = QFileDialog()
        file_dialogue.setFileMode(QFileDialog.AnyFile)
        file_dialogue.setAcceptMode(QFileDialog.AcceptSave)
        name_filters = ["PNG images (*.png)", "All files (*.*)"]
        file_dialogue.setNameFilters(name_filters)
        if file_dialogue.exec():
            filename = file_dialogue.selectedFiles()[0]
            try:
                self.canvas.figure.savefig(filename)
            except PermissionError as e:
                QMessageBox.warning(self, "Error", str(e))
                self.canvas.draw()
        file_dialogue.deleteLater()

    def change_batch_size(self):
        """Изменить размер пачки при обучении нейросетей"""
        self.sim.set_batch_size(self.batch_size_spinbox.value())
    
    def new_nn(self):
        """Создать новую нейросеть со случайными коэффициентами"""
        self.sim.new_nn()
    
    def load_nn(self):
        """Загрузить нейросеть из файла"""
        file_dialogue = QFileDialog()
        file_dialogue.setFileMode(QFileDialog.ExistingFile)
        file_dialogue.setAcceptMode(QFileDialog.AcceptOpen)
        if self.actor_critic_rb.isChecked():
            name_filters = ["TensorFlow session (*.meta)"]
        else:
            name_filters = ["Hierarchical data format (*.hdf)", "All files (*.*)"]
        file_dialogue.setNameFilters(name_filters)
        if file_dialogue.exec():
            filename = file_dialogue.selectedFiles()[0]
            try:
                self.sim.load_nn(filename)
            except OSError or FileNotFoundError or FileNotFoundError as e:
                QMessageBox.warning(self, "Error", str(e))
                # self.new_nn()
        file_dialogue.deleteLater()
    
    def save_nn(self):
        """Сохранить нейросеть в файл"""
        file_dialogue = QFileDialog()
        file_dialogue.setFileMode(QFileDialog.AnyFile)
        file_dialogue.setAcceptMode(QFileDialog.AcceptSave)
        if self.actor_critic_rb.isChecked():
            name_filters = ["TensorFlow session (*.meta)"]
        else:
            name_filters = ["Hierarchical data format (*.hdf)", "All files (*.*)"]
        file_dialogue.setNameFilters(name_filters)
        if file_dialogue.exec():
            filename = file_dialogue.selectedFiles()[0]
            try:
                self.sim.save_nn(filename)
            except PermissionError as e:
                QMessageBox.warning(self, "Error", str(e))
        file_dialogue.deleteLater()

    def change_episode_length(self):
        """Изменить длину эпизода - количества шагов перед случайным выбором новой позиции маятника"""
        episode_length = self.episode_length_spinbox.value()
        self.sim.max_ep_steps = episode_length

    def toggle_endless_episode(self):
        """Никогда не выбирать случайную позицию для маятника"""
        enable = self.endless_episode_checkbox.isChecked()
        self.sim.endless = enable

    def clear(self):
        """Очистить графики и поле с информацией об обучении"""
        self.canvas.clear()
        self.canvas.draw()
        self.output_text_field.clear()

    def paint_scene(self, thetas, omegas, moments, times, running_reward, episode):
        """Рисовать графики"""
        self.canvas.clear()
        self.canvas.theta_plot.plot(times, thetas, 'b')
        self.canvas.omega_plot.plot(times, omegas, 'g')
        self.canvas.moment_plot.plot(times, moments, 'r')
        self.canvas.draw()
        eps = self.sim.get_eps()
        self.output_text_field.append("Episode %d: running reward: %d, random probability: %-10.5g"
                                      % (episode, running_reward, eps))


class Plotter(FigureCanvas):
    """Класс отображения графиков"""
    def __init__(self, width, height, dpi, parent=None):
        font = {'family': 'Verdana',
                'weight': 'normal'}
        rc('font', **font)
        scale_const = 100
        self.figure = Figure(figsize=(width/scale_const, height/scale_const), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        self.theta_plot = self.figure.add_subplot(311)
        self.omega_plot = self.figure.add_subplot(312)
        self.moment_plot = self.figure.add_subplot(313)
        self.figure.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.425, wspace=0.35)
        self.clear()

    def clear(self):
        """Очистить графики"""
        self.theta_plot.clear()
        self.theta_plot.set_title("A deviation angle of the pendulum from vertical axis")
        self.theta_plot.set_ylabel('θ, rad')
        self.theta_plot.grid(True)

        self.omega_plot.clear()
        self.omega_plot.set_title("An angular velocity of the pendulum")
        self.omega_plot.set_ylabel('ω, rad/s')
        self.omega_plot.grid(True)

        self.moment_plot.clear()
        self.moment_plot.set_title("An external moment of force acting on the pendulum")
        self.moment_plot.set_xlabel('t, s')
        self.moment_plot.set_ylabel('M, N•m')
        self.moment_plot.grid(True)

