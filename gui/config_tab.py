# gui/config_tab.py
from PyQt5.QtWidgets import QWidget, QFormLayout, QLineEdit, QPushButton, QFileDialog, QVBoxLayout
import json
import os

from PyQt6 import uic

from utils.utils import load_config, save_config


class ConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.config_file = "../config.json"
        uic.loadUi("ui/main_window.ui", self)
        """
        self.layout = QFormLayout()

        # === PARAMÈTRES ===
        self.renko_size = QLineEdit("17.1")
        self.ema_period = QLineEdit("9")
        self.rsi_period = QLineEdit("14")
        self.macd_fast = QLineEdit("12")
        self.macd_slow = QLineEdit("26")
        self.macd_signal = QLineEdit("9")
        self.lstm_seq = QLineEdit("30")
        self.lstm_units = QLineEdit("50")

        self.layout.addRow("Renko Size", self.renko_size)
        self.layout.addRow("EMA Period", self.ema_period)
        self.layout.addRow("RSI Period", self.rsi_period)
        self.layout.addRow("MACD Fast", self.macd_fast)
        self.layout.addRow("MACD Slow", self.macd_slow)
        self.layout.addRow("MACD Signal", self.macd_signal)
        self.layout.addRow("LSTM Seq Len", self.lstm_seq)
        self.layout.addRow("LSTM Units", self.lstm_units)

        # === BOUTONS ===
        btn_layout = QVBoxLayout()
        btn_load = QPushButton("CHARGER")
        btn_save = QPushButton("SAUVEGARDER")
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_save)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.layout)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
        """
        self.btn_load.clicked.connect(self.load)
        self.btn_save.clicked.connect(self.save)
        self.btn_validate.clicked.connect(self.cntrl)
        self.cfg = self.paren.get_config()
        if self.cfg is not None and len(self.cfg) > 0:
            self.set_config(self.cfg)

    def get_config(self):
        return {
            'renko_size': float(self.renko_size.text()),
            'ema_period': int(self.ema_period.text()),
            'rsi_period': int(self.rsi_period.text()),
            'macd': [int(self.macd_fast.text()), int(self.macd_slow.text()), int(self.macd_signal.text())],
            'lstm': {'seq_len': int(self.lstm_seq.text()), 'units': int(self.lstm_units.text())}
        }

    def set_config(self, cfg):
        self.renko_size.setText(str(cfg.get('renko_size', 17.1)))
        self.ema_period.setText(str(cfg.get('ema_period', 9)))
        self.rsi_period.setText(str(cfg.get('rsi_period', 14)))
        macd = cfg.get('macd', [12, 26, 9])
        self.macd_fast.setText(str(macd[0]))
        self.macd_slow.setText(str(macd[1]))
        self.macd_signal.setText(str(macd[2]))
        self.cci_period.setText(str(cfg.get('cci_period', 14)))
        lstm = cfg.get('lstm', {'seq_len': 30, 'units': 50, 'threshold_buy': 0.6, 'threshold_sell':0.4})
        self.lstm_seq_len.setText(str(lstm.get('seq_len', 30)))
        self.lstm_units.setText(str(lstm.get('units', 50)))
        self.threshold_buy.setText(str(lstm.get('threshold_buy', 0.6)))
        self.threshold_sell.setText-str(lstm.get('threshold_sell', 0.4))

    def load(self):
        cfg = load_config(self)
        if cfg:
            self.set_config(cfg)

    def save(self):
        cfg = self.get_config()
        save_config(self, "config_live.json", cfg)

    def cntrl(self):
        print('cfg futur controle')