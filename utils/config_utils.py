# utils/config_utils.py
import hashlib
import json
import os

from PyQt6.QtWidgets import QFileDialog, QWidget

from utils.model_utils import nn_servers

indVal = ['', 'EMA', 'RSI', 'MACD_hist', 'ATR', 'CCI', 'Stoch RSI']
tarVal = ['', 'close', 'EMA', 'RSI']
features_base = ["EMA", "RSI", "MACD_hist", "close", "time_live", "ATR", "CCI", "Stoch RSI"]
params_base = {"renko_size": 17.1, "ema_period": 9, "rsi_period": 14, "rsi_high": 70, "rsi_low": 30,
                "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
               "cci_period": 14, "cci_high": 100, "cci_low": -100, "atr_period": 14,
               "threshold_buy": 0.6, "threshold_sell": 0.4, "close_buy": 0.53, "close_sell": 0.47}
lstm_base = {"lstm_seq_len": 48, "lstm_units": 96}
mlp_base = {"mlp_unit1": 128, "mlp_unit2": 64,"mlp_dropout": 0.3,
                    "mlp_lr": 0.001, "mlp_batch_size": 256, "mlp_patience": 10 }
lgbm_base = {"lgbm_learning_rate": 0.05, "lgbm_num_leaves": 31, "lgbm_n_estimators": 1000,
             "lgbm_feature_fraction": 0.8, "lgbm_bagging_fraction": 0.8,
             "lgbm_min_child_samples": 20, "lgbm_early_stop_rounds": 20}
xgb_base = {"xgb_learning_rate": 0.05, "xgb_max_depth": 6, "xgb_n_estimators": 1000,
            "xgb_subsample": 0.8, "xgb_colsample_bytree": 0.8, "xgb_early_stop_rounds":50}
config_base = {"parameters": params_base, "features": features_base, "lstm": lstm_base, "mlp": mlp_base,
               "lgbm": lgbm_base, "xgb": xgb_base}
trans = {"EMA": ["ema_period"], "RSI": ["rsi_period", "rsi_high", "rsi_low"], "MACD_hist": ["macd"],
         "CCI": ["cci__period", "cci_high", "cci_low"], "ATR}": ["atr_period"], "close":["close"], "time_live": ["time_live"]}


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
    # les rules sont éliminées
    parameters = config["parameters"]
    features = config["features"]
    lstm = config.get("lstm", None)
    mlp = config.get("mlp", None)
    lgbm = config.get("lgbm", None)
    xgb = config.get("xgb", None)
    target = config["target"]
    live = config["live"]
    params = {"renko_size":parameters["renko_size"]}
    for name in features:
        if name in trans.keys():
            for value in trans[name]:
                if value in parameters.keys():
                    params[value] = parameters[value]
                else:
                    params[name] = value
    version = live.get('version', [])
    ret = {"parameters":params, "features": features, "target": target,
            "symbol":live["symbol"], 'version': version}
    for vs in version:
        if vs in ['SIMPLE', 'ULTRA', 'LSTM']:
            if lstm is None:
                lstm = lstm_base.copy()
            ret['lstm'] = lstm
            break
        if vs == 'MLP':
            if mlp is None:
                mlp = mlp_base.copy()
            ret['mlp'] = mlp
            break
        if vs == 'LGBM':
            if lgbm is None:
                lgbm = lgbm_base.copy()
            ret['lgbm'] = lgbm
            break
        if vs == 'XGB':
            if xgb is None:
                xgb = xgb_base.copy()
            ret['xgb'] = xgb
            break
    return ret

def to_config_std(config):
    try:
        config_live = load_config(None, "../config_live.json")
    except BaseException as e:
        print("err chargement config live", e)
        return None
    try:
        if config_live is None:
            print("No config No Optim")
            return None
        config_std = config_live.copy()
        params_live = config_std.get("parameters")
        if params_live is None:
            print("err cfg no live param")
            return None
        features_cfg = config.get('features')        # features obligatoires
        if features_cfg is None:
            print("No features No optim")
            return None
        # print("feat", features_cfg)
        params = {"renko_size": config.get("renko_size", config_live.get("renko_size", 36.1))}
        features = []
        for name in features_cfg:
            if name in trans.keys():
                features.append(name)
                for value in trans[name]:
                    # print("val", name, value)
                    if value == name:
                        continue
                    params[value] = config.get(value, params_live.get(value, params_base[value]))
        # print("pam", params)
        version = config.get("VERSION", [])
        for vs in version:
            if vs in nn_servers:
                params["threshold_buy"] = config.get("threshold_buy",
                                                     params_live.get("threshold_buy",
                                                                                params_base["threshold_buy"]))
                params["threshold_sell"] = config.get("threshold_sell",
                                                 params_live.get("threshold_sell",
                                                                                params_base["threshold_sell"]))
                params["close_buy"] = config.get("close_buy",
                                                     params_live.get("close_buy",
                                                                                params_base["close_buy"]))
                params["close_sell"] = config.get("close_sell",
                                                 params_live.get("close_sell",
                                                                                params_base["close_sell"]))
                break
        config_std["parameters"] = params
        config_std['features'] = features
    except BaseException as e:
        print(f"création config std paramètres {e}")
        return None
    try:
        lstm = None
        mlp = None
        lgbm = None
        xgb = None
        for vs in version:
            if vs in ['SIMPLE', 'ULTRA', 'LSTM']:
                if lstm is not None:
                    continue
                lstm = {}
                for key in lstm_base.keys():
                    lstm[key] = config.get(key, lstm_base[key])
            if vs == 'MLP':
                mlp = {}
                for key in mlp_base.keys():
                    mlp[key] = config.get(key, mlp_base[key])
            if vs == 'LGBM':
                lgbm = {}
                for key in lgbm_base.keys():
                    lgbm[key] = config.get(key, lgbm_base[key])
            if vs == 'XGB':
                xgb = {}
                for key in xgb_base.keys():
                    xgb[key] = config.get(key, xgb_base[key])
        if lstm is not None:
            config_std['lstm'] = lstm
        if mlp is not None:
            config_std['mlp'] = mlp
        if lgbm is not None:
            config_std['lgbm'] = lgbm
        if xgb is not None:
            config_std['xgb'] = xgb
    except BaseException as e:
        print(f"err create config std réseaux {e}")
        return None
    try:
        open_rules = config.get("open_rules", {})
        if not open_rules:
            config_std["open_rules"] = {"rule_ema":False if 'EMA' not in features else True, "rule_rsi":False if 'RSI' not in features else True,
                          "rule_macd":False if 'MACD_hist' not in features else True}
        close_rules = config.get("close_rules", {})
        if not close_rules:
            config_std["close_rules"] = {"close_sens":True}
        for key in config_std['live'].keys():
            if key == 'version' or config.get(key, None) is None:
                continue
            config_std['live'][key] = config[key]
        config_std['live']['version'] = version
        return config_std
    except BaseException as e:
        print("err create config_std", e)
        return None
