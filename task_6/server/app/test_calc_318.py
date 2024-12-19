import sys
import math
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QSlider,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSplitter
)
from PyQt5.QtCore import Qt, QTimer

# Matplotlib для 3D графика
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MPlot3D(FigureCanvas):
    def __init__(self, parent=None):
        # Создаём фигуру и добавляем 3D-подграфик
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        super(MPlot3D, self).__init__(self.fig)
        self.setParent(parent)
        
        # Настройка осей
        self.ax.set_xlabel('Скорость u (м/с)')
        self.ax.set_ylabel('Угол a (°)')
        self.ax.set_zlabel('Момент M (Н·м)')
        self.ax.set_title('M = f(u, a)')
        
        # Установить лимиты осей (можно настроить по необходимости)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 90)
        self.ax.set_zlim(0, 1000)
        
        # Инициализация элементов для отображения
        # Линия для истории моментов
        self.past_moments_line, = self.ax.plot([], [], [], color='blue', label='История моментов')
        # Текущий момент
        self.current_point = None  # Изначально нет текущей точки
        
        # Добавляем легенду
        self.ax.legend()
        
        self.draw()

    def plot_past_moments(self, moments):
        """Отображает историю моментов на графике."""
        if moments:
            u_vals, a_vals, M_vals = zip(*moments)
            self.past_moments_line.set_xdata(u_vals)
            self.past_moments_line.set_ydata(a_vals)
            self.past_moments_line.set_3d_properties(M_vals)
        else:
            self.past_moments_line.set_xdata([])
            self.past_moments_line.set_ydata([])
            self.past_moments_line.set_3d_properties([])
        self.draw()

    def plot_current_point(self, u, a, M):
        """Отображает текущий момент на графике."""
        # Удаляем предыдущий маркер, если он существует
        if self.current_point:
            self.current_point.remove()
        # Добавляем новый маркер
        self.current_point = self.ax.scatter(u, a, M, color='red', s=50, label='Текущий момент')
        self.ax.legend()
        self.draw()

    def clear_past_moments(self):
        """Очищает историю моментов и текущий момент с графика."""
        self.past_moments_line.set_xdata([])
        self.past_moments_line.set_ydata([])
        self.past_moments_line.set_3d_properties([])
        if self.current_point:
            self.current_point.remove()
            self.current_point = None
        self.draw()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование момента")

        # Параметры: (начальное значение, минимум, максимум, шаг)
        self.param_info = {
            "Kп": (0.13, 0, 1, 0.001),
            "Cx": (0.62, 0, 1, 0.01),
            "Плотность (p)": (1000, 500, 2000, 10),
            "Скорость (u, м/с)": (2.05, 0, 10, 0.05),
            "Площадь (S)": (0.2, 0.01, 1, 0.01),
            "Угол (a, градусы)": (35, 0, 90, 1),
        }

        # Параметры радиуса штурвала: (начальное, минимум, максимум, шаг)
        self.r_info = (0.5, 0.1, 2.0, 0.01)
        self.param_values = {p: info[0] for p, info in self.param_info.items()}
        self.r_value = self.r_info[0]

        self.recording = False
        self.file_name = "moment_log.txt"
        self.past_moments = []  # Список для хранения истории моментов

        # Инициализация таймера для оптимизации обновления графика
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_3d_plot)

        self.init_ui()
        self.update_current_moment()
        self.update_3d_plot()  # начальный вызов

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Создание слайдеров для параметров
        self.param_sliders = {}
        self.param_labels = {}

        for label, (initial, minimum, maximum, step) in self.param_info.items():
            row_layout = QHBoxLayout()
            lbl_name = QLabel(f"{label}:")
            lbl_value = QLabel(f"{initial:.3f}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(minimum / step))
            slider.setMaximum(int(maximum / step))
            slider.setValue(int(initial / step))
            slider.setSingleStep(1)
            slider.setPageStep(10)
            # Подключаем сигнал изменения значения слайдера
            slider.valueChanged.connect(
                lambda val, l=label, s=step, lv=lbl_value: self.on_param_change(l, val * s, lv)
            )

            row_layout.addWidget(lbl_name)
            row_layout.addWidget(slider)
            row_layout.addWidget(lbl_value)
            main_layout.addLayout(row_layout)

            self.param_sliders[label] = slider
            self.param_labels[label] = lbl_value

        # Создание слайдера для радиуса штурвала
        r_layout = QHBoxLayout()
        r_lbl_name = QLabel("Радиус штурвала (r):")
        self.r_lbl_value = QLabel(f"{self.r_value:.3f}")
        r_slider = QSlider(Qt.Horizontal)
        r_min, r_max, r_step = self.r_info[1], self.r_info[2], self.r_info[3]
        r_slider.setMinimum(int(r_min / r_step))
        r_slider.setMaximum(int(r_max / r_step))
        r_slider.setValue(int(self.r_value / r_step))
        r_slider.setSingleStep(1)
        r_slider.setPageStep(10)
        # Подключаем сигнал изменения значения слайдера
        r_slider.valueChanged.connect(lambda val: self.on_r_change(val * r_step))

        r_layout.addWidget(r_lbl_name)
        r_layout.addWidget(r_slider)
        r_layout.addWidget(self.r_lbl_value)
        main_layout.addLayout(r_layout)

        # Создание кнопок Старт и Стоп
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        # Создание меток для отображения текущего и среднего момента
        self.current_m_label = QLabel("Текущий момент: - Н·м")
        self.average_m_label = QLabel("Средний момент: - Н·м")
        main_layout.addWidget(self.current_m_label)
        main_layout.addWidget(self.average_m_label)

        # Создание графика
        self.plot_widget = MPlot3D()

        # Размещение панелей и графика с помощью QSplitter
        splitter = QSplitter(Qt.Horizontal)
        params_widget = QWidget()
        params_widget.setLayout(main_layout)
        splitter.addWidget(params_widget)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(1, 1)  # Позволяет графику растягиваться

        # Упаковка в контейнерный макет
        container_layout = QVBoxLayout()
        container_layout.addWidget(splitter)
        container_widget = QWidget()
        container_widget.setLayout(container_layout)
        self.setCentralWidget(container_widget)

    def on_param_change(self, param_name, value, value_label):
        """Обработчик изменения значения параметра."""
        self.param_values[param_name] = value
        value_label.setText(f"{value:.3f}")
        self.update_current_moment()
        self.schedule_plot_update()

    def on_r_change(self, value):
        """Обработчик изменения значения радиуса штурвала."""
        self.r_value = value
        self.r_lbl_value.setText(f"{self.r_value:.3f}")
        self.update_current_moment()
        self.schedule_plot_update()

    def calculate_moment(self, Kp, Cx, p, u, S, a_deg, r):
        """Вычисляет момент M на основе заданных параметров."""
        a_rad = math.radians(a_deg)  # Преобразование угла в радианы
        F_sopr = (Cx * p * (u ** 2) * S * math.sin(a_rad)) / 2.0
        M = Kp * r * F_sopr
        return M

    def calculate_moment_vectorized(self, Kp, Cx, p, u, S, a_deg, r):
        """Векторизованное вычисление момента M для массивов u и a."""
        a_rad = np.radians(a_deg)  # Преобразование угла в радианы
        F_sopr = (Cx * p * (u ** 2) * S * np.sin(a_rad)) / 2.0
        M = Kp * r * F_sopr
        return M

    def get_current_moment(self):
        """Получает текущий момент на основе текущих значений параметров."""
        Kp = self.param_values["Kп"]
        Cx = self.param_values["Cx"]
        p = self.param_values["Плотность (p)"]
        u = self.param_values["Скорость (u, м/с)"]
        S = self.param_values["Площадь (S)"]
        a_deg = self.param_values["Угол (a, градусы)"]
        r = self.r_value
        return self.calculate_moment(Kp, Cx, p, u, S, a_deg, r)

    def update_current_moment(self):
        """Обновляет отображение текущего момента и добавляет его в историю при записи."""
        M = self.get_current_moment()
        self.current_m_label.setText(f"Текущий момент: {M:.3f} Н·м")
        if self.recording:
            # Записываем текущий момент в файл
            with open(self.file_name, 'a', encoding='utf-8') as f:
                f.write(f"{self.param_values['Скорость (u, м/с)']},{self.param_values['Угол (a, градусы)']},{M}\n")
            # Добавляем текущий момент в список истории
            self.past_moments.append((
                self.param_values['Скорость (u, м/с)'],
                self.param_values['Угол (a, градусы)'],
                M
            ))
            # Планируем обновление графика
            self.schedule_plot_update()

    def update_3d_plot(self):
        """Обновляет 3D-график с учетом истории моментов и текущего момента."""
        # Обновляем историю моментов на графике
        self.plot_widget.plot_past_moments(self.past_moments)

        # Отображение текущего момента на графике
        current_u = self.param_values["Скорость (u, м/с)"]
        current_a = self.param_values["Угол (a, градусы)"]
        current_M = self.get_current_moment()
        self.plot_widget.plot_current_point(current_u, current_a, current_M)

    def schedule_plot_update(self):
        """Запускает таймер для обновления графика через 100 мс."""
        self.update_timer.start(100)

    def start_recording(self):
        """Начинает запись моментов."""
        self.recording = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Очистка предыдущей записи
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        self.past_moments.clear()
        self.plot_widget.clear_past_moments()
        self.update_current_moment()

    def stop_recording(self):
        """Останавливает запись моментов и вычисляет среднее значение."""
        self.recording = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if os.path.exists(self.file_name):
            with open(self.file_name, 'r', encoding='utf-8') as f:
                moments = [line.strip().split(',') for line in f if line.strip()]
            moments = [(float(u), float(a), float(M)) for u, a, M in moments]
            if moments:
                avg_m = sum(m[2] for m in moments) / len(moments)
                self.average_m_label.setText(f"Средний момент: {avg_m:.3f} Н·м")
            else:
                self.average_m_label.setText("Средний момент: нет данных")
        else:
            self.average_m_label.setText("Средний момент: файл не найден")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
