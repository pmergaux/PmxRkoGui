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

from train.trainer import JepaPredictor
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
    target_cols = target["target_col"]
    # print(f"config_to_features : {features}, target : {target_cols}")
    if not isinstance(target_cols, list):
        target_cols = [target_cols]
    total_cols = [col for col in features]
    for tg in target_cols:
        if tg not in total_cols:
            total_cols.append(tg)
    features_cols = [col for col in features if col not in target_cols]
    for tg in target_cols:
        if tg not in features_cols:
            if target.get("target_include", False):
                features_cols.append(tg)
    return features_cols, target_cols, total_cols

# ====================== TARGETS SIMPLES ======================
def prepare_target_column(dfr, target_col, target_type, horizon=1):
    """
    Prépare la colonne cible en fonction du type demandé.
    Retourne le DataFrame avec une nouvelle colonne 'target'.
    """
    if not target_col in dfr.columns:
        print(f"ptc target {target_col} not in columns {dfr.columns.tolist()}")
        raise ValueError(f"target {target_col} not in columns")
    df = dfr.copy()

    if target_type == 'return':
        df['target'] = df[target_col].pct_change(periods=-horizon).shift(-horizon)
        # Écrêtage à 3 ou 5 écarts-types pour éviter que les outliers ne polluent l'entraînement
        mean = df['target'].mean()
        std = df['target'].std()
        df['target'] = df['target'].clip(lower=mean - 3 * std, upper=mean + 3 * std)
    elif target_type == 'diff_scaled':
        # Très efficace pour les Renko : différence normalisée
        # On calcule la différence et on divise par l'écart-type local (volatilité)
        diff = df[target_col].diff(periods=-horizon).shift(-horizon)
        #df['target'] = diff / df[target_col].rolling(window=100).std()
        vol_ema = df[target_col].ewm(span=100).std()
        df['target'] = diff / vol_ema
    # --- ON ÉVITE 'direction' POUR LA RÉGRESSION ---
    elif target_type == 'direction':
        # On prédit le signe du changement futur
        # Le futur est défini par un décalage négatif
        future_change = df[target_col].diff(periods=-1)
        # On crée la cible : 1 si le futur est positif (le prix va monter), 0 sinon
        df['target'] = (future_change > 0).astype(int)
    elif target_type == 'value':
        # Exemple pour un problème de régression : on prédit la valeur future
        df['target'] = df[target_col].shift(-1)

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
@njit(fastmath=True, cache=True)
def create_sequences_numba(data: np.ndarray, seq_len: int, n_features: int, horizon: int = 0):
    n_samples = len(data) - seq_len - horizon + 1   # nombre d'échantillons
    if n_samples <= 0:  # ← garde-fou
        return (np.empty((0, seq_len, n_features), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32))

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
    elif model_type == 'cat':
        if ext == "":
            path += ".joblib"
        elif ext == ".":
            path+= "joblib"
    elif model_type == 'tab':
        if ext == "":
            path += ".joblib"
        elif ext == ".":
            path+= "joblib"
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
        #return lgb.Booster(model_file=path)  # LightGBM utilise Booster pour charger
        return joblib.load(path)
    elif model_type == 'cat':
        return joblib.load(path)
    elif model_type == 'tab':
        return joblib.load(path)
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

def save_model_artifact(model, filepath="tabicl_trading_model.pkl"):
    """
    Sauvegarde l'objet modèle complet pour un usage ultérieur en live.
    """
    #joblib.dump(model, filepath)
    # Au lieu de joblib.dump, TabICL gère sa propre sérialisation
    model.save(
        "my_trading_model.pkl",
        save_model_weights=True,  # Sauvegarde les poids fine-tunés
        save_training_data=False,
        # False en production pour ne pas alourdir le fichier avec l'historique d'entraînement
        save_kv_cache=True
    )
    print(f"Modèle sérialisé avec succès vers : {filepath}")


def load_model_artifact(filepath="tabicl_trading_model.pkl"):
    """
    Charge le modèle fine-tuné depuis le disque pour l'inférence en live.
    """
    print(f"Chargement du modèle depuis : {filepath}")
    #model = joblib.load(filepath)
    from tabicl import FinetunedTabICLRegressor  # ou la classe correspondante

    # Utilisation de la méthode de classe native .load()
    model = FinetunedTabICLRegressor.load("my_trading_model.pkl")
    return model

import torch


def cntrl_jepa(state_dict, jepa_cfg):
    # 1. Déduction automatique de l'architecture depuis le checkpoint
    weight_ih = state_dict["encoder.weight_ih_l0"]
    hidden_3x, input_dim = weight_ih.shape
    hidden_dim = hidden_3x // 3

    # proj_latent.weight a pour shape [latent_dim, hidden_dim] en PyTorch (Linear out_features, in_features)
    proj_weight = state_dict["proj_latent.weight"]
    latent_dim = proj_weight.shape[0]

    # Si par construction votre proj_latent est inversée, la dimension d'entrée de la linear est hidden_dim
    # On peut vérifier les deux axes pour être 100% sûr :
    # proj_weight.shape[1] correspond au hidden_dim d'entrée de la projection.

    # Nombre de couches GRU
    num_layers = sum(1 for k in state_dict.keys() if k.startswith("encoder.weight_ih_l"))

    # 2. Récupération des valeurs de configuration attendues
    cinput_dim = jepa_cfg.get("INPUT_DIM")
    chidden_dim = jepa_cfg.get("HIDDEN_DIM")
    clatent_dim = jepa_cfg.get("LATENT_DIM")
    cnum_layers = jepa_cfg.get("NUM_LAYERS")

    # 3. Vérification et synchronisation intelligente
    mismatch = False
    if cinput_dim != input_dim or chidden_dim != hidden_dim or clatent_dim != latent_dim or cnum_layers != num_layers:
        print(f"⚠️ Ajustement de la config JEPA (détecté vs configuré) :")
        print(f"   - input_dim  : {cinput_dim} -> {input_dim}")
        print(f"   - hidden_dim : {chidden_dim} -> {hidden_dim}")
        print(f"   - latent_dim : {clatent_dim} -> {latent_dim}")
        print(f"   - num_layers : {cnum_layers} -> {num_layers}")

        # Plutôt que de planter brutalement, on met à jour la config avec la stricte vérité du modèle sauvegardé
        jepa_cfg["INPUT_DIM"] = input_dim
        jepa_cfg["HIDDEN_DIM"] = hidden_dim
        jepa_cfg["LATENT_DIM"] = latent_dim
        jepa_cfg["NUM_LAYERS"] = num_layers

    return jepa_cfg

def save_jepa( model, stats, model_save_path, jepa_cfg):
    # 5. Sauvegarde du modèle entraîné pour exploitation en trading live
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stats": stats,
        "hyperparameters": {
            "input_dim": jepa_cfg["INPUT_DIM"],
            "hidden_dim": jepa_cfg["HIDDEN_DIM"],
            "latent_dim": jepa_cfg["LATENT_DIM"],
            "num_layers": jepa_cfg["NUM_LAYERS"]
        }
    }
    torch.save(checkpoint, model_save_path)


def load_jepa(model_path, jepa_cfg=None):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint["model_state_dict"]
    stats = checkpoint["stats"]

    # On récupère les hyperparams exacts du best trial d'Optuna s'ils existent,
    # sinon on déduit automatiquement des poids (comme vu juste avant)
    hp = checkpoint.get("hyperparameters", {})
    if len(hp) == 0:
        weight_ih = state_dict["encoder.weight_ih_l0"]
        hidden_3x, input_dim = weight_ih.shape
        hidden_dim = hidden_3x // 3
        # On déduit les autres dimensions
        latent_dim = state_dict["proj_latent.weight"].shape[0]
        # On compte le nombre de couches (si weight_ih_l1 existe, il y en a 2, etc.)
        num_layers = sum(1 for k in state_dict.keys() if k.startswith("encoder.weight_ih_l"))
    else:
        input_dim = hp.get("input_dim", 6)
        hidden_dim = hp.get("hidden_dim", 128)
        latent_dim = hp.get("latent_dim", 64)
        num_layers = hp.get("num_layers", 2)

    model = JepaPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, stats
