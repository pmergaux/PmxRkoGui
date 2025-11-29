# utils/config_utils.py
import hashlib
import json
import os

from PyQt6.QtWidgets import QFileDialog, QWidget

indVal = ['', 'LSTM', 'EMA', 'RSI', 'MACD_line', 'ATR', 'CCI', 'Stoch RSI']
tarVal = ['', 'close', 'EMA', 'RSI']

def config_to_hash(config: dict) -> str:
    """Hash unique pour une config → nom de modèle"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]

def load_config(qui, filename=None, required=False):
    if filename is None:
        path, _ = QFileDialog.getOpenFileName(qui, "Charger config", "", "JSON (*.json)")
    else:
        path = filename
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    elif required:
        raise FileNotFoundError(f"CONFIG OBLIGATOIRE MANQUANTE : {filename}")
    else:
        return {}

def save_config(qui: QWidget, filename, cfg):
    path, _ = QFileDialog.getSaveFileName(parent=None, caption="Sauver config", directory=filename, filter="JSON (*.json)")
    if path:
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        qui.parent.statusBar().showMessage(f"Config sauvegardée : {path}")

def prepare_to_hashcode(config:dict):
    parameters = config["parameters"]
    features = config["features"]
    lstm = config["lstm"]
    target = config["target"]
    live = config["live"]
    params = {"renko_size":parameters["renko_size"]}
    trans = {"EMA":["ema_period"], "RSI": ["rsi_period", "rsi_high", "rsi_low"], "MACD_histo": ["macd"], "CCI": ["cci__period", "cci_high", "cci_low"]}
    for name in features:
        if name in trans.keys():
            for value in trans[name]:
                params[value] = parameters[value]
    return {"parameters":params, "features": features, "target": target, "lstm": lstm, "symbol":live["symbol"]}
