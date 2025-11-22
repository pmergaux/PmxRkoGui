# utils/config_utils.py
import hashlib
import json

indVal = ['', 'LSTM', 'EMA', 'RSI', 'MACD_line', 'ATR', 'CCI', 'Stoch RSI']
tarVal = ['', 'close', 'EMA', 'RSI']

def config_to_hash(config: dict) -> str:
    """Hash unique pour une config → nom de modèle"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]
