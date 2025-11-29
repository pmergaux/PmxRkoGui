# live_scaler_qt.py
import numpy as np
import joblib
from PyQt6.QtCore import QObject, QMutex, QMutexLocker

class LiveScaler(QObject):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path=None):
        if hasattr(self, "_ready"): return
        super().__init__()
        self.mutex = QMutex()
        data = joblib.load(path)
        self.min = np.asarray(data['min'], dtype=np.float32)
        self.range = np.asarray(data['range'], dtype=np.float32)
        self._ready = True

    def transform(self, X):
        with QMutexLocker(self.mutex):
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            return (arr - self.min) / self.range

_scaler = None
def get_scaler(path="models/scaler_live.pkl"):
    global _scaler
    if _scaler is None:
        _scaler = LiveScaler(path)
    return _scaler

def save_live_scaler(train_df, feature_cols, path="models/scaler_live.pkl"):
    import os
    arr = train_df[feature_cols].to_numpy(np.float32)
    data = {'min': arr.min(0), 'range': arr.max(0) - arr.min(0) + 1e-12}
    os.makedirs("models", exist_ok=True)
    joblib.dump(data, path)
    print(f"Live scaler sauvé → {path}")
