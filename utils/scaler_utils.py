import numpy as np
import joblib
import os

"""
#usage
# Pendant l'optimisation (une seule fois) pensez à calculer le hash_code save et load
scaler_data = train_and_save_minmax_scaler(
    train_df[feature_cols].values,
    "models/scaler_a7f3e9c1.pkl"
)
# En live
new_scaled = load_and_transform(latest_brick[feature_cols].values, "models/scaler_a7f3e9c1.pkl")
"""

scaler_data = None


# =============================================
# 4. SAUVEGARDE SCALER LIVE
# =============================================
def train_save_live_scaler(data, feature_cols, path="models/scaler_live.pkl"):
    global scaler_data
    arr = np.asarray(data, dtype=np.float32)
    # print("shape train", arr.shape)
    scaler_data = {
        'min': arr.min(axis=0),
        'range': arr.max(axis=0) - arr.min(axis=0) + 1e-12,
        'feature_cols': feature_cols
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler_data, path)
    print(f"Live scaler sauvé → {path}")
    return (arr - scaler_data['min']) / scaler_data['range']

def load_and_transform(data: np.ndarray, scaler_path: str):
    """
    Chargement + transform en live → < 2 ms garanti
    """
    global scaler_data
    if scaler_data is None:
        scaler_data = joblib.load(scaler_path)
    arr = np.asarray(data, dtype=np.float32)
    return (arr - scaler_data['min']) / scaler_data['range']
