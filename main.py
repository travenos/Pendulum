import sys
import GUI
from PyQt5.QtWidgets import QApplication


def main():
    """
    Создание и вызов главного окна
    """
    app = QApplication(sys.argv)
    mw = GUI.MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
