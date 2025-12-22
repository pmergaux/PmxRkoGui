import pickle

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

# =============================================
# SCALER LIVE
# =============================================
def train_fit_transform_scaler(df, cols):
    data = df[cols].to_numpy(dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    # print("shape train", arr.shape)
    scaler_data = {
        'min': arr.min(axis=0),
        'range': arr.max(axis=0) - arr.min(axis=0) + 1e-12,
        'cols': cols    # pour usage ulterieur
    }
    return scaler_data, (arr - scaler_data['min']) / scaler_data['range']

def load_and_transform(scaler_data, df):
    """
    Chargement + transform en live → < 2 ms garanti
    """
    data = df[scaler_data['cols']].to_numpy(dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    return (arr - scaler_data['min']) / scaler_data['range']

def load_scaler(scaler_path: str):
    """
    Charge un scaler (StandardScaler, MinMaxScaler, etc.).
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler non trouvé : {scaler_path}")
    scaler_data = joblib.load(scaler_path)
    return scaler_data

def save_scaler(scaler_data, scaler_path: str):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler_data, scaler_path)

def save_scaler_std(scaler, scaler_path):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "wb") as f: # "wb" -> Write Binary
        pickle.dump(scaler, f)

def load_scaler_std(scaler_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler non trouvé : {scaler_path}")
    with open(scaler_path, "rb") as f: # "rb" -> Read Binary
        scaler = pickle.load(f)
    return scaler


#
