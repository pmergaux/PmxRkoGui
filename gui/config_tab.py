# gui/config_tab.py
from PyQt6.QtWidgets import QTabWidget ,QWidget, QFormLayout, QLineEdit, QPushButton, QFileDialog, QVBoxLayout
import json
import os

from PyQt6 import uic

from utils.utils import load_config, save_config


class ConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        uic.loadUi("ui/config_tab.ui", self)
        self.parent = parent
        self.config_file = "../config.json"
        self.btn_load.clicked.connect(self.load)
        self.btn_save.clicked.connect(self.save)
        self.btn_validate.clicked.connect(self.cntrl)
        self.cfg = self.parent.get_config_live()
        self.varFeat = [self.feat_1, self.feat_2, self.feat_3, self.feat_4]
        if self.cfg is not None and len(self.cfg) > 0:
            self.set_config(self.cfg)

    def get_config(self):
        try:
            features = []
            for i in range(len(self.varFeat)):
                if self.varFeat[i].currentIndex() != 0:
                    features.append(self.varFeat[i].currentText())
            if self.target_col.currentIndex() == 0 and 'LSTM' in features:
                raise ("Configuration IA et cible manque !")
            return {
                'renko_size': float(self.renko_size.text()),
                'ema_period': int(self.ema_period.text()),
                'rsi_period': int(self.rsi_period.text()),
                'macd': [int(self.macd_fast.text()), int(self.macd_slow.text()), int(self.macd_signal.text())],
                'lstm': {'seq_len': int(self.lstm_seq_len.text()), 'units': int(self.lstm_units.text()),
                         'threshold_buy': float(self.thresold_buy.text()), 'threshold_sell': self.threshold_sell.text()},
                'features':features,
                'target': {'target_col': self.target_col.currentText(), 'target_type': self.target_type.currentText(),
                           'target_include':True if self.target_include.isChecked() else False}
            }
        except Exception as e:
            print(f"Erreur dans la coniguration {e}")
            return None

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
        self.threshold_sell.setText(str(lstm.get('threshold_sell', 0.4)))
        features = cfg.get('features', [])
        for i in range(len(self.varFeat)):
            self.varFeat[i].setCurrentIndex(0)
        for i in range(len(features)):
            if i > 3:
                print(f"features trop nombreuses {features}")
                break
            if len(features[i]) > 2:
                self.varFeat[i].setCurrentText(features[i])
        self.target_col.setCurrentIndex(0)
        target = cfg.get('target', None)
        if target is not None:
            self.target_col.setCurrentText(target.get('target_col', ''))
            self.target_type.setCurrentText(target.get('target_type', 'direction'))
            self.target_include.setChecked(target.get('target_include', False))

    def load(self):
        cfg = load_config(self)
        if cfg:
            self.set_config(cfg)

    def save(self):
        cfg = self.cntrl()
        if cfg is not None:
            save_config(self, "config_live.json", cfg)

    def cntrl(self):
        cfg = self.get_config()
        if cfg is not None:
            features = cfg.get('features', [])
            if len(features) == 0:
                print(f"Attention sans indicateurs pas de trading")
                return None
        return cfg