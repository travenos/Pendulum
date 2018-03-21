from PyQt5.QtWidgets import QWidget, QPushButton
#from environment import Env
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches
import matplotlib.lines
import matplotlib.path
import numpy as np
import matplotlib.pyplot as plt
# TODO Интеграция с matplotlib


class MainWindow(QWidget):
    """Окно графического интерфейса"""

    def __init__(self):
        super().__init__()

        #self.env = Env(self)
        self.real_time = False

        self.canvas = Plotter(10, 6, dpi=100, parent=self)

        self.initUI()

    def initUI(self):
        """
        Формирование интерфейса
        """
        self.resize(1200, 600)
        self.move(300, 300)

        start = QPushButton('Start', self)
        start.resize(start.sizeHint())
        start.move(1000, 30)
        start.clicked.connect(self.on_start)

        stop = QPushButton('Stop', self)
        stop.resize(start.sizeHint())
        stop.move(1000, 90)
        #stop.clicked.connect(self.env.stop)

        self.canvas.move(0, 0)
        self.canvas.plot()
        self.canvas.draw_circle()

        # TODO Добавить холст
        # TODO Добавить переключатель реального времени
        # TODO Кнопка "Отрисовать текущий момент"
        # TODO Добавить список роботов
        # TODO Добавить интерфейс добавления и удаления роботов
        # TODO Интерфейс полного сохранения и загрузки роботов и нейронных сетей в файлы
        # TODO Кнопка переключатель обучения

        self.setWindowTitle('Cylinder transporting')
        self.show()

    def on_start(self):
        """
        Запуск моделирования
        """
        #t = Thread(target=self.env.start, args=(self.real_time,))
        #t.start()

    def paint_scene(self):
        """
        Рисовать на холсте цилиндр и роботов
        """
        # TODO Рисовать на холсте цилиндр и роботов
        pass


class Plotter(FigureCanvas):
    def __init__(self, width, height, dpi, parent=None):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)

    def set_title(self, title):
        self.ax.set_title(title)

    def plot(self, data=None, color='b'):
        if data is None:
            self.ax.plot()
        else:
            self.ax.plot(data, color)
        self.draw()

    def draw_circle(self, center=(0,0), radius=1, color='b', fill=False, rescale=None):
        circle = matplotlib.patches.Circle(center, radius=radius, color=color, fill=fill)
        self.ax.add_patch(circle)
        if rescale is not None:
            self.ax.set_xlim(xmin=center[0]-rescale*radius, xmax=center[0]+rescale*radius)
            self.ax.set_ylim(ymin=center[1]-rescale*radius, ymax=center[1]+rescale*radius)
