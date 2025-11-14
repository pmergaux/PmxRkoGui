# gui/live_chart.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from collections import deque
from datetime import datetime
import random
import pandas as pd
import numpy as np

from strategy.pmxRko import PmxRkoStrategy


class LiveRenkoChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.brick_size = 0.00010
        self.max_bricks = 50
        self.target_bricks = 45
        self.renko_bricks = deque(maxlen=self.max_bricks)
        self.bid_prices = deque(maxlen=self.max_bricks)
        self.ema_values = deque(maxlen=self.max_bricks)
        self.rsi_values = deque(maxlen=self.max_bricks)
        self.macd_line = deque(maxlen=self.max_bricks)
        self.macd_signal = deque(maxlen=self.max_bricks)
        self.closes = []
        self.price = 1.08500
        self.strategy = None

        self.setup_ui()
        self.start_simulation()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Bouton lancement
        self.btn_start = QPushButton("LANCER LE LIVE")
        self.btn_start.setStyleSheet("background:#006400;color:white;font-weight:bold;padding:10px;")
        self.btn_start.clicked.connect(self.start_live)

        self.status_label = QLabel("Statut : En attente...")
        self.status_label.setStyleSheet("color:orange;font-weight:bold;")

        # Canvas
        self.figure = Figure(facecolor='black', figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
        self.ax_renko = self.figure.add_subplot(gs[0], facecolor='black')
        self.ax_rsi = self.figure.add_subplot(gs[1], facecolor='black')
        self.ax_macd = self.figure.add_subplot(gs[2], facecolor='black')

        # Tooltip
        self.tooltip = QLabel(self)
        self.tooltip.setStyleSheet("background:#1a1a1a;color:white;border:1px solid gray;padding:6px;font-size:10pt;")
        self.tooltip.hide()

        layout.addWidget(self.btn_start)
        layout.addWidget(self.status_label)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def start_live(self):
        if self.timer and self.timer.isActive():
            return
        self.btn_start.setEnabled(False)
        self.status_label.setText("Statut : Live démarré - Génération des briques...")
        self.status_label.setStyleSheet("color:#00FF00;font-weight:bold;")
        self.start_simulation()

    def start_simulation(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(300)

    def renko_add(self, price, time_str):
        if not self.renko_bricks:
            open_r = round(price - (price % self.brick_size), 5)
            self.renko_bricks.append({'open': open_r, 'close': open_r, 'up': True, 'time': time_str})
            return True
        last = self.renko_bricks[-1]['close']
        diff = price - last
        bricks = int(abs(diff) // self.brick_size)
        if bricks == 0: return False
        is_up = diff > 0
        for _ in range(bricks):
            open_r = last
            close_r = round(open_r + self.brick_size if is_up else open_r - self.brick_size, 5)
            self.renko_bricks.append({'open': open_r, 'close': close_r, 'up': is_up, 'time': time_str})
            last = close_r
        return True

    def ema(self, prices, period=9):
        if len(prices) < period: return prices[-1] if prices else 0
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

    def rsi(self, prices, period=14):
        if len(prices) < period + 1: return 50.0
        deltas = np.diff(prices)
        up = deltas.clip(min=0)
        down = -deltas.clip(max=0)
        ma_up = pd.Series(up).rolling(period).mean().iloc[-1]
        ma_down = pd.Series(down).rolling(period).mean().iloc[-1]
        rs = ma_up / ma_down if ma_down != 0 else 100
        return 100 - (100 / (1 + rs))

    def macd(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow: return 0, 0
        ema_f = pd.Series(prices).ewm(span=fast, adjust=False).mean().iloc[-1]
        ema_s = pd.Series(prices).ewm(span=slow, adjust=False).mean().iloc[-1]
        macd_l = ema_f - ema_s
        macd_s = macd_l  # simplifié
        return macd_l, macd_s

    def update_chart(self):
        if not hasattr(self, 'strategy'):
            cfg = self.parent.load_config_from_file()
            self.strategy = PmxRkoStrategy(cfg)
            self.strategy.cl = None  # Simu
        """
        self.strategy.run("ETHUSD")
        self.price += random.gauss(0, 0.00008)
        bid = self.price + random.uniform(0, 0.00005)
        close = self.price
        time_str = datetime.now().strftime("%H:%M:%S")
        self.closes.append(close)

        new_brick = self.renko_add(close, time_str)
        if new_brick:
            ema9 = self.ema(self.closes[-9:], 9)
            self.bid_prices.append(bid)
            self.ema_values.append(ema9)
            if len(self.closes) > 14:
                self.rsi_values.append(self.rsi(self.closes[-15:]))
            if len(self.closes) > 26:
                ml, ms = self.macd(self.closes[-30:])
                self.macd_line.append(ml)
                self.macd_signal.append(ms)

        if len(self.renko_bricks) >= self.target_bricks and not self.strategy:
            cfg = self.parent.load_config_from_file()
            self.brick_size = cfg.get("brick_size", 0.00010)
            self.strategy = True
            self.status_label.setText(f"Statut : Stratégie lancée avec brick_size={self.brick_size}")
        """
        # === AFFICHAGE ===
        for ax in [self.ax_renko, self.ax_rsi, self.ax_macd]:
            ax.clear()
            ax.grid(True, color='gray', alpha=0.2)
            ax.set_facecolor('black')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_xticklabels([])

        n = len(self.renko_bricks)
        if n > 0:
            x = list(range(n))
            for i, brick in enumerate(self.renko_bricks):
                rect = patches.Rectangle((x[i]-0.4, brick['open']), 0.8,
                    self.brick_size if brick['up'] else -self.brick_size,
                    facecolor=(0,0.8,0) if brick['up'] else (0.8,0,0), linewidth=0, alpha=0.9)
                self.ax_renko.add_patch(rect)

            min_len = min(n, len(self.bid_prices), len(self.ema_values))
            if min_len > 0:
                x_lines = x[-min_len:]
                self.ax_renko.plot(x_lines, list(self.bid_prices)[-min_len:], color='cyan', lw=0.8, label='Bid')
                self.ax_renko.plot(x_lines, list(self.ema_values)[-min_len:], color='orange', lw=1.5, label='EMA 9')
            self.ax_renko.set_xlim(-0.5, self.max_bricks-0.5)
            self.ax_renko.set_title(f"Renko Live - {n}/50 briques", color='white')
            self.ax_renko.legend(loc='upper left', facecolor='black', labelcolor='white', fontsize=8)

        if len(self.rsi_values) > 0:
            x_rsi = list(range(len(self.rsi_values)))
            self.ax_rsi.plot(x_rsi, self.rsi_values, color='purple', lw=1.2)
            self.ax_rsi.axhline(70, color='red', ls='--', alpha=0.5)
            self.ax_rsi.axhline(30, color='green', ls='--', alpha=0.5)
            self.ax_rsi.set_ylim(0, 100)

        if len(self.macd_line) > 0:
            x_macd = list(range(len(self.macd_line)))
            self.ax_macd.plot(x_macd, self.macd_line, color='blue', lw=1.2, label='MACD')
            self.ax_macd.plot(x_macd, self.macd_signal, color='red', lw=1.2, label='Signal')
            self.ax_macd.legend(loc='upper left', facecolor='black', labelcolor='white', fontsize=8)

        self.canvas.draw()

    def on_mouse_move(self, event):
        if not event.inaxes or event.inaxes != self.ax_renko:
            self.tooltip.hide()
            return
        x_mouse = int(event.xdata + 0.5)
        n = len(self.renko_bricks)
        if 0 <= x_mouse < n:
            brick = self.renko_bricks[x_mouse]
            ema_val = self.ema_values[x_mouse] if x_mouse < len(self.ema_values) else "N/A"
            text = f"Time: {brick['time']}\nOpen: {brick['open']:.5f}\nClose: {brick['close']:.5f}\nEMA: {ema_val:.5f}"
            self.tooltip.setText(text)
            self.tooltip.adjustSize()
            self.tooltip.move(int(event.x)+15, int(event.y)-40)
            self.tooltip.show()
        else:
            self.tooltip.hide()
