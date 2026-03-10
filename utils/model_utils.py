# utils/model_utils.py
import datetime
import pickle

import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow import keras
import lightgbm as lgb
import xgboost as xgb
from typing import List, Tuple, Dict
from numba import njit, prange
from sklearn.preprocessing import MinMaxScaler
from utils.scaler_utils import train_fit_transform_scaler, load_and_transform
from utils.utils import get_extension

nn_servers = ['SIMPLE', 'ULTRA', 'LSTM', 'GRU', 'TRANSFORMER', 'TFT', 'N_BRICKS', 'XGB', 'LGBM', 'MLP']
kr_servers = ['simple', 'ultra', 'lstm', 'gru', 'transformer', 'mlp']

def generate_param_combinations(grid):
    import itertools
    keys = grid.keys()
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

# ================================
# 1. FONCTIONS UTILITAIRES
# ================================
def clean_features(df, cols):
    dfc = df[cols].copy()
    dfc = dfc.replace([np.inf, -np.inf], np.nan).fillna(0)
    return dfc

def config_to_features(config:dict):
    features = config["features"]
    target = config["target"]
    features_cols = []
    total_cols = []
    target_cols = target["target_col"]
    if not isinstance(target_cols, list):
        target_cols = [target_cols]
    for col in features:
        if len(col) > 1:
            if col not in target["target_col"]:
                features_cols.append(col)
            total_cols.append(col)
    for col in target_cols:
        if col not in features_cols:
            if target.get("target_include", False):
                features_cols.append(col)
        if col not in total_cols:
            total_cols.append(col)
    return features_cols, target_cols, total_cols

# ====================== TARGETS SIMPLES ======================
def prepare_target_column(df, target_col, target_type, horizon=1):
    """
    Prépare la colonne cible en fonction du type demandé.
    Retourne le DataFrame avec une nouvelle colonne 'target'.
    """
    if not target_col in df.columns:
        print(f"target {target_col} not in columns")
        raise f"target not in columns {target_col}"
    df = df.copy()
    if target_type == 'direction':
        # On prédit le signe du changement futur
        # Le futur est défini par un décalage négatif
        future_change = df[target_col].diff(periods=-1)
        # On crée la cible : 1 si le futur est positif (le prix va monter), 0 sinon
        df['target'] = (future_change > 0).astype(int)
    elif target_type == 'value':
        # Exemple pour un problème de régression : on prédit la valeur future
        df['target'] = df[target_col].shift(-1)
    elif target_type == 'return':
        df['target'] = df[target_col].pct_change(horizon).shift(-horizon)
    # ... (vous pouvez ajouter d'autres types de cibles ici) ...
    # On supprime les dernières lignes où la cible est NaN car inconnue
    df = df.dropna(subset=['target']).reset_index(drop=True)
    return df

def prepare_targets_simple(df, horizon=5):
    df = df.copy()
    df['target'] = df['close'].pct_change(horizon).shift(-horizon)
    df['target'] = np.sign(df['target'])
    df['target'] = df['target'].replace(-1, 0)  # pour n'avoir que 0 ou 1
    df = df.dropna(subset=['target']).reset_index(drop=True)
    return df

# =============================================
# 1. SCALING COLS (features or targets seulement)
# =============================================
def scale_cols_only(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str],
):
    scaler, train = train_fit_transform_scaler(train_df, cols)
    val   = load_and_transform(scaler, val_df)
    test  = load_and_transform(scaler, test_df)
    return scaler, train, val, test

# =============================================
# 2. ASSEMBLAGE + TARGETS BRUTS
# =============================================
def assemble_targets(
    X_train_scaled, X_val_scaled, X_test_scaled,
    train_df, val_df, test_df,
    target_cols: List[str]
):
    y_train = train_df[target_cols].to_numpy(dtype=np.float32)
    y_val   = val_df[target_cols].to_numpy(dtype=np.float32)
    y_test  = test_df[target_cols].to_numpy(dtype=np.float32)

    train_ready = np.hstack([X_train_scaled, y_train])
    val_ready   = np.hstack([X_val_scaled,   y_val])
    test_ready  = np.hstack([X_test_scaled,  y_test])

    return train_ready, val_ready, test_ready

# =============================================
# 2b. ASSEMBLAGE + TARGETS scaled
# =============================================
def assemble_with_targets(
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train_scaled, y_val_scaled, y_test_scaled,
):
    train_ready = np.hstack([X_train_scaled, y_train_scaled])
    val_ready   = np.hstack([X_val_scaled,   y_val_scaled])
    test_ready  = np.hstack([X_test_scaled,  y_test_scaled])

    return train_ready, val_ready, test_ready

# =============================================
# 3. CREATE SEQUENCES — NUMBA ULTRA-RAPIDE
# =============================================
"""
pour seq = 5 et len = 11
n_samples = 7
i = 0,1,2,3,4,5,6
X 0-4 à 6-10            la lim haute est exclue
y 4,5,6,7,8,9,10 
soit y avec -1 si on veut symchroniser sinon on anticipe y par rapport à X
"""
@njit(fastmath=True)
def create_sequences_numba(data: np.ndarray, seq_len: int, n_features: int, horizon: int = 0):
    n_samples = len(data) - seq_len - horizon + 1
    if n_samples <= 0:  # ← garde-fou
        return (np.empty((0, seq_len, n_features), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32))

    n_samples = len(data) - seq_len - horizon + 1   # nombre d'échantillons
    n_targets = data.shape[1] - n_features  # nombre total de colonnes - celles des features = nombre colonnes target

    X = np.empty((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.empty((n_samples, n_targets), dtype=np.float32)

    # Attn normalement la dernière ligne aurait dû être enlevée car on ne veut que des bougies closes (valides)
    # ici on retourne tout !
    for i in range(n_samples):
        X[i] = data[i:i + seq_len, :n_features]
        y[i] = data[i + horizon + seq_len-1, n_features:]

    return X, y

# ======================load and save version simplifiée ==============================
def save_model(model, path: str, model_type: str = None):
    """
    Sauvegarde un modèle (Keras, XGBoost ou LightGBM).
    Parameters
    ----------
    model : objet modèle
    path : str
        Chemin complet avec extension (.keras, .json, .txt)
    model_type : str, optional
        'keras', 'xgb' ou 'lgb'. Si None, déduit de l'extension.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = get_extension(path)
    if model_type is None and ext != "" and ext != ".":
        if ext in ['.keras', '.h5']:
            model_type = 'keras'
        elif ext in ['.json', '.xgb']:
            model_type = 'xgb'
        elif ext in ['.txt', '.lgb']:
            model_type = 'lgb'
        else:
            raise ValueError(f"Extension {ext} non reconnue pour sauvegarde modèle")
    else:
        model_type = model_type.lower()
    print(f"Sauvegarde modèle {model_type.upper()} → {path}")
    if model_type == 'keras' or model_type in kr_servers:
        if ext == "":
            path += ".keras"
        elif ext == ".":
            path += "keras"
        model.save(path)  # .keras ou .h5 selon l'extension
    elif model_type == 'xgb':
        if ext == "":
            path += ".json"
        elif ext == ".":
            path += "json"
        model.save_model(path)  # .json recommandé (lisible)
    elif model_type == 'lgb' or model_type == 'lgbm':
        if ext == "":
            path += ".txt"
        elif ext == ".":
            path += "txt"
        save_lgbm_model(model, path)  # .txt recommandé (lisible)
    else:
        raise ValueError(f"Type de modèle {model_type} non supporté")

def load_model(path: str, model_type: str = None):
    """
    Charge un modèle sauvegardé.
    Returns
    -------
    objet modèle chargé
    """
    ext = get_extension(path)
    if model_type is None:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.keras', '.h5']:
            model_type = 'keras'
        elif ext in ['.json', '.xgb']:
            model_type = 'xgb'
        elif ext in ['.txt', '.lgb']:
            model_type = 'lgb'
        else:
            raise ValueError(f"Extension {ext} non reconnue")
    else:
        model_type = model_type.lower()
    if model_type == 'keras' or model_type in kr_servers:
        if ext == "":
            path += ".keras"
        elif ext == ".":
            path += "keras"
    elif model_type == 'xgb':
        if ext == "":
            path += ".json"
        elif ext == ".":
            path += "json"
    elif model_type == 'lgb' or model_type == 'lgbm':
        if ext == "":
            path += ".txt"
        elif ext == ".":
            path += "txt"
    else:
        raise ValueError(f"Type de modèle {model_type} non supporté")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle non trouvé : {path}")
    print(f"Chargement modèle {model_type.upper()} ← {path}")
    if model_type == 'keras' or model_type in kr_servers:
        return keras.models.load_model(path)
    elif model_type == 'xgb':
        model = xgb.XGBClassifier()  # ou XGBRegressor selon ton cas
        model.load_model(path)
        return model
    elif model_type == 'lgb' or model_type == 'lgbm':
        return lgb.Booster(model_file=path)  # LightGBM utilise Booster pour charger
    else:
        return None

def save_lgbm_model(model, path: str = None, model_type: str = 'auto'):
    """
    Sauvegarde un modèle LightGBM (Booster ou LGBMClassifier)
    - Si Booster → .txt (lisible, recommandé)
    - Si LGBMClassifier → extrait le Booster et sauvegarde .txt
    - Option pickle (.pkl) si tu veux tout garder (mais déconseillé)

    Retourne le chemin final utilisé
    """
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = f"models/lgbm_best_{timestamp}.txt"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(model, lgb.Booster):
        model.save_model(path)
        print(f"Modèle Booster sauvegardé → {path}")

    elif isinstance(model, lgb.LGBMClassifier) or isinstance(model, lgb.LGBMRegressor):
        # On extrait le Booster interne
        booster = model.booster_
        booster.save_model(path)
        print(f"Modèle LGBMClassifier/Regressor → Booster extrait et sauvegardé → {path}")

        # Option : sauvegarde complète via pickle (si tu veux les params scikit-learn)
        pickle_path = path + ".pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Sauvegarde complète (pickle) → {pickle_path}")

    else:
        raise TypeError("Modèle non reconnu comme LightGBM")

    return path

def load_lgbm_model(path: str):
    """
    Charge un modèle LightGBM depuis un fichier .txt ou .pkl
    Retourne un Booster ou un LGBMClassifier selon le fichier
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle non trouvé : {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in ['.txt', '.model']:
        # Charge comme Booster (API native)
        booster = lgb.Booster(model_file=path)
        print(f"Modèle Booster chargé ← {path}")
        return booster

    elif ext == '.pkl':
        # Charge le modèle complet (scikit-learn API)
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modèle complet (pickle) chargé ← {path}")
        return model

    else:
        raise ValueError(f"Extension non reconnue : {ext} (attendu .txt ou .pkl)")
