# gui/config_tab.py
import json
import os

from PyQt5.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QWidget, QFileDialog, QComboBox, QLineEdit
from PyQt6 import uic

from utils.config_utils import indVal, tarVal
from utils.qwidget_utils import set_widget_from_dict, get_widget_from_dict, get_widget_from_list
from utils.utils import load_config, save_config, to_number

class ConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        uic.loadUi("ui/config_tab.ui", self)
        self.parent = parent
        self.btn_load.clicked.connect(self.load)
        self.btn_save.clicked.connect(self.save)
        self.btn_validate.clicked.connect(self.cntrl)
        self.cfg = self.parent.get_config_live()
        self.mapping = {"features":
            [self.feat_1, self.feat_2, self.feat_3, self.feat_4, self.feat_5, self.time_live]
                        }
        for value in self.mapping["features"]:
            if value.objectName() != "time_live":
                value.addItems(indVal)
        self.target_col.addItems(tarVal)
        self.set_config(self.cfg)

    def get_config(self):
        try:
            self.cntrl()
            get_widget_from_dict(self, self.cfg["target"])
            get_widget_from_dict(self, self.cfg["open_rules"])
            get_widget_from_dict(self, self.cfg["close_rules"])
            get_widget_from_dict(self, self.cfg["live"])
            get_widget_from_dict(self, self.cfg("parameters"))
            get_widget_from_dict(self, self.cfg["lstm"])
            return self.cfg
        except Exception as e:
            print(f"Erreur dans la configuration {e}")
            return None

    def set_config(self, cfg):
        set_widget_from_dict(self, cfg)
        self.cfg = cfg

    def load(self):
        path, _ = QFileDialog.getOpenFileName(None, "Charger config", "", "JSON (*.json)")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.cfg = json.load(f)
            self.set_config(self.cfg)

    def save(self):
        cfg = self.cntrl()
        path, _ = QFileDialog.getSaveFileName(parent=None, caption="Sauver config", directory='config_live.json',
                                              filter="JSON (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            self.parent.statusBar().showMessage(f"Config sauvegardée : {path}")

    def cntrl(self):
        features = get_widget_from_list(self, "features")
        if self.target_col.currentIndex() == 0 and 'LSTM' in features:
            raise ("Configuration IA et cible manque !")
        self.cfg["features"] = features
        return  self.cfg