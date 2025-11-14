# gui/execution_tab.py
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from gui.live_chart import LiveChart
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.dates import DateFormatter

from utils.renko_utils import colonnesRko
from utils.utils import BUY, SELL, NONE, load_config
from matplotlib.patches import Polygon, Rectangle
import numpy as np



class ExecutionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()

        # === BOUTONS CÔTE À CÔTE ===
        btn_layout = QHBoxLayout()  # ← HORIZONTAL
        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_start.setFixedWidth(100)
        self.btn_stop.setFixedWidth(100)
        self.btn_start.clicked.connect(parent.start_execution)
        self.btn_stop.clicked.connect(parent.stop_execution)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()  # Pousse à gauche
        layout.addLayout(btn_layout)
        # ---- config ----------
        """
        self.btn_load = QPushButton("Charger Config Live")
        self.btn_save = QPushButton("Sauver Config Live")
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        self.btn_load.clicked.connect(self.load_live_config)
        self.btn_save.clicked.connect(self.save_live_config)
        """
        # === GRAPHIQUE PRINCIPAL (BRIQUES + EMA) ===
        self.chart = LiveChart()
        layout.addWidget(self.chart)

        # === SOUS-GRAPHIQUES : RSI + MACD ===
        self.fig_ind, (self.ax_rsi, self.ax_macd) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        self.canvas_ind = FigureCanvasQTAgg(self.fig_ind)
        layout.addWidget(self.canvas_ind)

        # === LÉGENDE ===
        legend = QLabel(
            "Signaux : ▲ sigo BUY | ▼ sigo SELL | ■ sigc CLOSE | ★ LSTM"
        )
        layout.addWidget(legend)
        self.setLayout(layout)
        self.last_brick = None

    # gui/execution_tab.py
    def update_display(self, data):
        df_full = data.get('df')
        if df_full is None or df_full.empty:
            return

        # --- 10 dernières pour le graphique principal ---
        df_chart = df_full.tail(10)
        chart_data = {
            'df': df_chart,
            'current_bid': data['current_bid']
        }
        self.chart.update_display(chart_data)

        if self.last_brick is not None and self.last_brick == df_full.index[-1]:
            return
        # --- 80 dernières pour RSI/MACD (cohérent avec 10 bougies = 2h) ---
        df_ind = df_full.tail(80)

        self.ax_rsi.clear()
        self.ax_macd.clear()

        # RSI
        rsi = df_ind['RSI'].dropna()
        if not rsi.empty:
            self.ax_rsi.plot(rsi.index, rsi.values, color='purple', linewidth=1.2)
            self.ax_rsi.axhline(70, color='r', linestyle='--', alpha=0.5)
            self.ax_rsi.axhline(30, color='g', linestyle='--', alpha=0.5)
            self.ax_rsi.set_ylim(0, 100)
            self.ax_rsi.set_ylabel('RSI')
            self.ax_rsi.grid(True, alpha=0.3)

        # MACD
        if 'MACD_line' in df_ind.columns:
            macd = df_ind['MACD_line'].dropna()
            signal = df_ind['MACD_signal'].dropna()
            hist = df_ind['MACD_hist'].dropna()
            if not macd.empty:
                self.ax_macd.plot(macd.index, macd.values, color='blue', linewidth=1)
                self.ax_macd.plot(signal.index, signal.values, color='orange', linewidth=1)
                # self.ax_macd.bar(hist.index, hist.values, color='gray', alpha=0.6, width=0.8)
                self.ax_macd.axhline(0, color='k', linewidth=0.5)
                self.ax_macd.set_ylabel('MACD')
                self.ax_macd.grid(True, alpha=0.3)

        # --- SYNCHRONISER AXE X ---
        if not df_chart.empty:
            start = df_chart.index[0]
            end = df_chart.index[-1] + pd.Timedelta(minutes=2)
            self.ax_rsi.set_xlim(start, end)
            self.ax_macd.set_xlim(start, end)

        # --- FORMATAGE DATE ---
        self.fig_ind.autofmt_xdate()
        self.ax_macd.xaxis.set_major_formatter(DateFormatter('%H:%M'))

        self.canvas_ind.draw()
