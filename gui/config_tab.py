# gui/config_tab.py
import json
import os

from PyQt6.QtWidgets import QWidget, QFileDialog, QComboBox, QLineEdit
from PyQt6 import uic

from utils.config_utils import indVal, tarVal
from utils.qwidget_utils import set_widget_from_dict, get_widget_from_dict, get_widget_from_list

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
        self.strategy_name = "pmxRKO" # Default or detected
        self.set_config(self.cfg)

    def get_config(self):
        try:
            self.cntrl()
            # On détermine où sauvegarder les données (dans la sous-section ou à la racine)
            target_cfg = self.cfg.get(self.strategy_name, self.cfg)
            
            get_widget_from_dict(self, target_cfg["target"])
            get_widget_from_dict(self, target_cfg["open_rules"])
            get_widget_from_dict(self, target_cfg["close_rules"])
            get_widget_from_dict(self, target_cfg["live"])
            get_widget_from_dict(self, target_cfg["parameters"])
            get_widget_from_dict(self, target_cfg["lstm"])
            return self.cfg
        except Exception as e:
            print(f"Erreur dans la configuration {e}")
            return None

    def set_config(self, cfg):
        self.cfg = cfg
        # Détection de la stratégie
        target_cfg = cfg
        if "parameters" not in cfg:
            for key, value in cfg.items():
                if isinstance(value, dict) and "parameters" in value:
                    self.strategy_name = key
                    target_cfg = value
                    break
        set_widget_from_dict(self, target_cfg)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(None, "Charger config", "", "JSON (*.json)")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.cfg = json.load(f)
            self.set_config(self.cfg)

    def save(self):
        cfg = self.get_config()
        if cfg is None:
            print("err recup cfg")
            return
        path, _ = QFileDialog.getSaveFileName(parent=None, caption="Sauver config", directory='config_live.json',
                                              filter="JSON (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            self.parent.statusBar().showMessage(f"Config sauvegardée : {path}")
            self.parent.set_config_live(cfg)

    def cntrl(self):
        features = get_widget_from_list(self, "features")
        if self.target_col.currentIndex() == 0 and 'LSTM' in features:
            raise Exception("Configuration IA et cible manque !")
        target_cfg = self.cfg.get(self.strategy_name, self.cfg)
        target_cfg["features"] = features
        return  self.cfg