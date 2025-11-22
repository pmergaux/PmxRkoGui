# gui/execution_tab.py
from datetime import datetime

import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt6 import uic
from mt5linux import MetaTrader5

from gui.live_chart import LiveChart
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.dates import DateFormatter
from strategy.pmxRko import PmxRkoStrategy
from simulation.simulator import RenkoSimulator
from utils.utils import BUY, SELL


class ExecutionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        uic.loadUi("ui/live_chart.ui", self)
        self.parent = parent

        self.btn_start.clicked.connect(self.start_execution)
        self.btn_stop.clicked.connect(self.stop_execution)
        # === GRAPHIQUE PRINCIPAL (BRIQUES + EMA) ===
        self.chart = LiveChart(self)
        self.view.addWidget(self.chart)

        # === SOUS-GRAPHIQUES : RSI + MACD ===
        self.fig_ind, (self.ax_rsi, self.ax_macd) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        self.canvas_ind = FigureCanvasQTAgg(self.fig_ind)
        self.view.addWidget(self.canvas_ind)

        # === LÉGENDE ===
        self.legend_text = "Signaux : ▲ sigo BUY | ▼ sigo SELL | ■ sigc CLOSE | ★ LSTM --|-- "
        # self.. permet une réutilisation comme position : sens, prix ouv. profit
        self.legend = QLabel(
            self.legend_text
        )
        self.view.addWidget(self.legend)

        self.last_brick = None
        self.simulator = None
        self.cfg = None

    def start_execution(self):
        self.cfg = self.parent.get_config_live()
        self.status_label.setText('Trading en cours')
        if self.cfg.get('live', None) is None:
            self.simulator = RenkoSimulator(self, self.cfg)  # ← CORRIGÉ : 3 args OK
            self.parent.start_execution(self.simulator, 100)
        else:
            self.simulator = PmxRkoStrategy(self, self.cfg)
            self.simulator.lance()
            self.parent.start_execution(self.simulator,1000)

    def stop_execution(self):
        if self.simulator is not None:
            self.simulator.timer.stop()
        self.status_label.setText("Trading stop")

    # gui/execution_tab.py
    def update_display(self, data):
        df_full = data.get('df')
        if df_full is None or df_full.empty:
            return

        # --- 10 dernières pour le graphique principal ---
        df_chart = df_full.tail(15)
        chart_data = {
            'df': df_chart,
            'current_bid': data['current_bid']
        }
        self.chart.update_display(chart_data)

        if self.last_brick is not None and self.last_brick == df_full.index[-1]:
            return
        positionStatut = ''
        strategy = data.get("strategy", None)
        if strategy is not None and strategy.positions is not None and len(strategy.positions) > 0:
            position = strategy.positions[0]
            ls = BUY if position.type == MetaTrader5.POSITION_TYPE_BUY else SELL
            cp = position.price_current
            pc = position.price_open
            TP = (pc + strategy.stp * ls) if strategy.stp != 0 else 0
            SL = (pc - strategy.ssl * ls) if strategy.ssl != 0 else 0
            date = pd.Timestamp(position.time, unit='s')
            # print(type(date), date)  #.strftime("yyyy-MM-dd hh:mm:ss"))
            positionStatut = (f" position : {'buy' if ls==BUY else 'sell'} le : {date} open : {pc:.2f}, "
                              f"tp : {TP:.2f} sl : {SL:.2f} pf : {position.profit:.2f}")
        self.legend.setText(self.legend_text + positionStatut)
        # --- 80 dernières pour RSI/MACD (cohérent avec 10 bougies = 2h) ---
        df_ind = df_full.tail(80)

        self.ax_rsi.clear()
        self.ax_macd.clear()

        # RSI
        rsi = df_ind['RSI'].dropna()
        if not rsi.empty:
            self.ax_rsi.plot(rsi.index, rsi.values, color='purple', linewidth=1.2)
            self.ax_rsi.axhline(self.cfg.get('rsi_high',70), color='r', linestyle='--', alpha=0.5)
            self.ax_rsi.axhline(self.cfg.get('rsi_low',30), color='g', linestyle='--', alpha=0.5)
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
                self.ax_macd.plot(hist.index, hist.values, color='gray', linewidth=1)
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
