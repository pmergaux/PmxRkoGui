# pipeline_manager.py
import copy
import os
import numpy as np
import pandas as pd
from tabicl import TabICLRegressor
import torch
from torch.utils.data import Dataset, DataLoader

from decision.candle_decision import add_indicators_optimized, choix_features_numba, calculate_atr_4sl, calculate_atr
from train.prediction import prediction, tabicl_predict
from utils.rate_utils import ticks2rates
from utils.renko_utils import tick21renko

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # ← à mettre TOUT EN HAUT du fichier
# 2. Optionnel mais recommandé : limite la verbosité de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warning, 3=error only
# Optionnel : désactive les protections Lightning qui tuent les processus
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import gc
import tensorflow as tf

from train.trainer import lstm_train_model, tabicl_train, xgb_train, lgbm_train, gru_train, mlp_train, \
    lstm_train_simple, lstm_train_ultra, catboost_train, tabicl_fine_train, jepa_train
from utils.model_utils import scale_cols_only, config_to_features, prepare_target_column, assemble_with_targets, \
    create_sequences_numba

def reset_tf_memory():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

renko_tab = {'1h': 1.72, "15m": 4.5, "4h": 0.8}

def get_stabilized_renko_size(new_atr, last_renko_size):
    """
    Stabilise la taille du Renko pour éviter la volatilité des paramètres.
    """
    # 1. Bornes strictes (Clamping)
    MIN_SIZE, MAX_SIZE = 16.0, 26.0

    # 2. Lissage (EMA pour réactivité + stabilité)
    # On applique un coefficient de lissage (ex: 0.3) pour amortir les chocs
    smoothed_atr = (new_atr * 0.1) + (last_renko_size * 0.9)

    # 3. Hystérésis (Seuil de tolérance de 15%)
    # Si le changement est trop faible, on garde l'ancienne taille pour éviter le bruit
    if abs(smoothed_atr - last_renko_size) / last_renko_size < 0.15:
        return last_renko_size

    # 4. Application finale des bornes
    final_size = max(MIN_SIZE, min(MAX_SIZE, smoothed_atr))

    # Arrondi à 1 décimale (votre contrainte)
    return round(final_size, 1)

def prepare_renko(config_std, df_ticks_complet):
    timeframe = config_std.get("live", {}).get("timeframe", "1h")
    atr_window = config_std.get("indicators_and_filters", {}).get("atr", {}).get("window", 14)
    dj = ticks2rates(df_ticks_complet, timeframe, 'bid')
    atr_ser = calculate_atr(dj, atr_window)[atr_window:]
    atr_mean = atr_ser.mean()
    atr_max = atr_ser.max()
    atr_min = atr_ser.min()
    atr_std = atr_ser.std()
    print(f"atr_mean {atr_mean:.2f} max {atr_max:.2f} min {atr_min:.2f} std {atr_std:.2f}")
    # Assurez-vous que l'index est bien en datetime
    df_ticks_complet.index = pd.to_datetime(df_ticks_complet.index)
    #multiplier = 23.2 / atr_mean
    # Grouper par date (annule l'heure, ne garde que l'année-mois-jour)
    daily_groups = df_ticks_complet.groupby(df_ticks_complet.index.date)

    # Convertir les groupes en une liste ordonnée de DataFrames (blocs d'une journée)
    blocks = [group for _, group in daily_groups]
    #old_size = config_std.get("parameters", {}).get("renko_size", 23.2)
    old_size = atr_mean
    processed_blocks = []
    print(f"Processing {len(blocks)} blocks")
    #multiplier = 1
    for i, block in enumerate(blocks):
        dj = ticks2rates(block, timeframe, 'bid')
        if i == 0:
            if len(dj) > atr_window:
                #current_size  = calculate_atr_4sl(dj.iloc[:atr_window+1], multiplier=multiplier, window=atr_window)
                #old_size = get_stabilized_renko_size(current_size, old_size)
                old_size = calculate_atr(dj, atr_window)[atr_window:].mean()

        # 2. Application de vos routines (Renko + indicateurs)
        df_renko = tick21renko(block, None, old_size, 'bid')
        df_renko["renko_volatility_ratio"] = old_size
        # 3. Stockage du bloc traité
        processed_blocks.append(df_renko)
        # 1. Calcul dynamique de l'ATR (en incluant potentiellement le bloc précédent pour le buffer)
        # Exemple de calcul basé sur votre logiqueS actuelle
        if len(dj) > atr_window:
            #current_size = calculate_atr_4sl(dj, multiplier=multiplier, window=atr_window)
            #old_size = get_stabilized_renko_size(current_size, old_size)
            old_size = calculate_atr(dj, atr_window)[atr_window:].mean()
        print(f"block {i} size {old_size:.2f}")
    # 4. Reconstruction finale
    final_df = pd.concat(processed_blocks)
    config_std["parameters"]["renko_size"] = old_size
    return final_df

def prepare_jepa_data(df, brick_size, stats):
    # 1. Direction (+1 / -1)
    direction = np.where(df["close_renko"] >= df["open_renko"], 1.0, -1.0).astype(np.float32)
    # 2. Corps réel normalisé
    real_body = ((df["close"] - df["open"]) / brick_size).values.astype(np.float32)
    # 3. Mèches haute et basse normalisées
    max_oc = np.maximum(df["open"].values, df["close"].values)
    min_oc = np.minimum(df["open"].values, df["close"].values)
    high_wick = (np.maximum(0.0, df["high"].values - max_oc) / brick_size).astype(np.float32)
    low_wick = (np.maximum(0.0, min_oc - df["low"].values) / brick_size).astype(np.float32)
    # 4. Durée de formation normalisée
    time_s = pd.to_datetime(df["time"]).astype("int64") // 10 ** 9
    duration_s = time_s.diff().fillna(time_s.diff().median()).values
    log_dur = np.log1p(np.maximum(0.0, duration_s))
    if stats is None:
        stats = {
            "mean_log_dur": float(log_dur.mean()),
            "std_log_dur": float(log_dur.std() + 1e-6)
        }
    elif isinstance(stats, dict):
        mean_log_dur = stats.get("mean_log_dur", 0.0)
        std_log_dur = stats.get("std_log_dur", 1.0)
    else:
        # Si 'stats' est un float ou autre chose par erreur, on définit des valeurs par défaut
        print(
            f"⚠️ [JEPA] Attention: 'stats' n'est pas un dictionnaire mais un {type(stats)} (valeur: {stats}). Utilisation des valeurs par défaut.")
        mean_log_dur = 0.0
        std_log_dur = 1.0
    # --- Normalisation globale via les stats du train ---
    # Sécurité blindée contre les types inattendus (float, None, etc.)
    norm_dur = ((log_dur - stats["mean_log_dur"]) / stats["std_log_dur"]).astype(np.float32)
    # --- 🚨 DÉTECTION D'EMBALLEMENT (MARKET RUSH) ---
    features = np.stack([direction, real_body, high_wick, low_wick, norm_dur], axis=1)
    # --- 6ème feature : déplacement relatif en briques ---
    raw_closes = df["close"].values.astype(np.float32)
    """
    # =========================================================================
    # 🆕 5. NOUVELLES FEATURES : Volatilité glissante (ex: fenêtre de 15 briques)
    # =========================================================================
    window_vol = 15

    # Volatilité de la vitesse (écart-type glissant du log_dur)
    s_log_dur = pd.Series(log_dur)
    rolling_dur_std = s_log_dur.rolling(window=window_vol, min_periods=1).std().fillna(0.0).values.astype(np.float32)
    # Volatilité des mèches (somme des mèches lissée sur la fenêtre)
    total_wicks = high_wick + low_wick
    s_wicks = pd.Series(total_wicks)
    rolling_wick_mean = s_wicks.rolling(window=window_vol, min_periods=1).mean().fillna(0.0).values.astype(np.float32)
    # On empile le tout (passe de 5 à 7 features d'entrée)
    # N'oubliez pas d'ajuster input_dim=7 dans votre config si vous faites ça !
    features = np.stack([
        direction, 
        real_body, 
        high_wick, 
        low_wick, 
        norm_dur, 
        rolling_dur_std, 
        rolling_wick_mean
    ], axis=1)
    raw_closes = df["close"].values.astype(np.float32)
    """
    return features, raw_closes, stats

def create_jepa_windows(data: np.ndarray, context_len: int = 50, target_len: int = 10):
    """
    Découpe une série temporelle (ex: prix Renko, indicateurs) en fenêtres glissantes.

    Args:
        data: Array numpy de forme [N_samples, N_features]
        context_len: Longueur de l'historique passé (le contexte)
        target_len: Longueur du futur à prédire (la cible)

    Returns:
        context_windows: Tenseur [Batch, context_len, features]
        target_windows: Tenseur [Batch, target_len, features]
    """
    contexts = []
    targets = []

    total_len = len(data)
    window_size = context_len + target_len

    for i in range(total_len - window_size + 1):
        # Le bloc de contexte (passé)
        ctx = data[i: i + context_len]
        # Le bloc cible (futur proche)
        tgt = data[i + context_len: i + window_size]

        contexts.append(ctx)
        targets.append(tgt)

    return torch.tensor(np.array(contexts), dtype=torch.float32), \
        torch.tensor(np.array(targets), dtype=torch.float32)


class NextCloseRenkoDataset(Dataset):
    def __init__(
            self,
            df,
            seq_len=64,
            split="train",
            train_ratio=0.8,
            stats=None
    ):
        super().__init__()
        self.seq_len = seq_len

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        self.brick_size = float(abs(df["open_renko"].iloc[0] - df["close_renko"].iloc[0]))
        df = df.reset_index(drop=True).sort_values(by="time").reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)

        if split == "train":
            self.df = df.iloc[:split_idx].copy().reset_index(drop=True)
        else:
            self.df = df.iloc[split_idx:].copy().reset_index(drop=True)

        self.stats = stats
        self.features, self.raw_closes = self._prepare_data(self.df)
        self.num_samples = max(0, len(self.features) - self.seq_len)
        # À la fin de __init__ ou _prepare_data +1 pour le rel_price de getitem
        self.input_dim = self.features.shape[1] + 1
        if self.num_samples == 0:
            raise ValueError(f"Pas assez de briques ({len(self.features)}) pour séquence {self.seq_len}")

    def _prepare_data(self, df):
        features, raw_closes, self.stats = prepare_jepa_data(df, self.brick_size, self.stats)
        return features, raw_closes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 64 briques visibles : [idx : idx + 64]
        window_feat = self.features[idx: idx + self.seq_len].copy()
        window_closes = self.raw_closes[idx: idx + self.seq_len].copy()

        # 6e feature stationarisée : déplacement en briques par rapport à la première brique de la fenêtre
        rel_price = ((window_closes - window_closes[0]) / self.brick_size)[:, np.newaxis]
        visible_seq = torch.tensor(np.concatenate([window_feat, rel_price], axis=-1), dtype=torch.float32)

        # 65ème brique (l'avenir) : idx + 64
        last_visible_close = window_closes[-1]
        target_65_close = self.raw_closes[idx + self.seq_len]

        # Target = variation en briques de la 65ème par rapport à la 64ème
        target_delta_bricks = torch.tensor(
            (target_65_close - last_visible_close) / self.brick_size,
            dtype=torch.float32
        )

        return visible_seq, target_delta_bricks, last_visible_close, target_65_close


def prepare_jepa(df_bricks, jepa):
    # 1. Chargement des données (Split chronologique 80/20)
    SEQ_LEN = jepa["SEQ_LEN"]
    print(f"Chargement des séquences de {SEQ_LEN} briques...")
    BATCH_SIZE = jepa["BATCH_SIZE"]
    train_dataset = NextCloseRenkoDataset(
        df=df_bricks,
        seq_len=SEQ_LEN,
        split="train",
        train_ratio=0.8
    )
    test_dataset = NextCloseRenkoDataset(
        df=df_bricks,
        seq_len=SEQ_LEN,
        split="test",
        train_ratio=0.8,
        stats=train_dataset.stats
    )
    jepa["INPUT_DIM"] = train_dataset.input_dim
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

    return train_dataset, train_loader, test_loader

def train_all_models(config_std, df, trial):
    """
    Fonction centralisée : Entraîne tous les modèles requis par la config.
    """
    try:
        cfg = copy.deepcopy(config_std)
        VERSION = cfg['live']['version']

        # ====================== FEATURE ENGINEERING ======================
        if "renko_volatility_ratio" in cfg["features"] and "renko_volatility_ratio" not in df.columns:
            raise ValueError("renko_volatility_ratio missing")
        df_renko = add_indicators_optimized(df, cfg["parameters"])
        if "renko_volatility_ratio" in cfg["features"] and "renko_volatility_ratio" not in df_renko.columns:
            raise ValueError("renko_volatility_ratio missing after add_indic")
        df_renko = choix_features_numba(df_renko, cfg)
        if "renko_volatility_ratio" in cfg["features"] and "renko_volatility_ratio" not in df_renko.columns:
            raise ValueError("renko_volatility_ratio missing choix add_indic")
        if df_renko is None or len(df_renko) < 200:
            print("❌ Pas assez de données après feature engineering")
            return {}

        # ====================== TARGET ======================
        features_cols, target_cols, _ = config_to_features(cfg)
        if "renko_volatility_ratio" in df.columns and "renko_volatility_ratio" not in features_cols:
            features_cols.append("renko_volatility_ratio")
        cfg["features"] = features_cols

        target_col = target_cols[0] if isinstance(target_cols, list) else target_cols

        if target_col not in df_renko.columns:
            print(f"❌ Target '{target_col}' non trouvée")
            return {}

        target_type = cfg['target'].get('target_type', 'diff_scaled')
        df_renko = prepare_target_column(df_renko, target_col, target_type).reset_index(drop=True)
        df_renko = df_renko.ffill().dropna()

        # ====================== SPLIT & SCALE ======================
        train_len = int(len(df_renko) * 0.65)
        val_len = int(len(df_renko) * 0.15)

        train_df = df_renko.iloc[:train_len]
        val_df = df_renko.iloc[train_len:train_len + val_len]
        test_df = df_renko.iloc[train_len + val_len:]

        X_scaler, X_train, X_val, X_test = scale_cols_only(train_df, val_df, test_df, features_cols)

        # IMPORTANT : y en 1D pour XGB / LGBM (régression)
        y_train = train_df[[target_col]].values.astype(np.float32).ravel()
        y_val   = val_df[[target_col]].values.astype(np.float32).ravel()
        #y_test  = test_df[[target_col]].values.astype(np.float32).ravel()
        # ici les targets sont selon leur dim (en générale 1 col)
        # l'assemblage est en préparation de create-seq
        train_r, val_r, test_r = assemble_with_targets(X_train, X_val, X_test,
                                                       train_df[[target_col]].values,
                                                       val_df[[target_col]].values,
                                                       test_df[[target_col]].values)

        if 'GRU' in VERSION:
            seq_len = cfg.get('gru', {}).get('gru_seq_len', 24)
        else:
            seq_len = cfg.get('lstm', {}).get('lstm_seq_len', 24)

        X_train_seq, y_train_seq = create_sequences_numba(train_r, seq_len, len(features_cols))
        X_val_seq, y_val_seq = create_sequences_numba(val_r, seq_len, len(features_cols))
        models = {}
        # ====================== ENTRAÎNEMENT ======================
        for vs in VERSION:
            reset_tf_memory()
            if 'SIMPLE' == vs:
                models[vs] = lstm_train_simple(X_train_seq, y_train_seq, X_val_seq, y_val_seq, seq_len, len(features_cols))
            elif 'ULTRA' == vs:
                units = cfg['lstm'].get('lstm_units', 96)
                models[vs] = lstm_train_ultra(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
            elif 'LSTM' == vs:
                units = cfg['lstm'].get('lstm_units', 128)
                models[vs] = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
            elif 'GRU' == vs:
                gru_params = cfg.get('gru', {})
                models[vs] = gru_train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, gru_params)
            elif 'MLP' == vs:
                mlp_params = cfg.get('mlp', {})
                models[vs] = mlp_train(X_train, y_train, X_val, y_val, len(features_cols), mlp_params)
                """
            # version classifier
            elif 'XGB' == vs:
                xgb_params = cfg.get('xgb', {})
                model_rnn = xgb_train(
                    X_train, y_train, X_val, y_val,
                    learning_rate=xgb_params.get('xgb_learning_rate', 0.05),
                    max_depth=xgb_params.get('xgb_max_depth', 6),
                    objective='binary:logistic'
                )
            elif 'LGBM' == vs:
                lgbm_params = cfg.get('lgbm', {})
                model_rnn = lgbm_train(
                    X_train, y_train, X_val, y_val,
                    learning_rate=lgbm_params.get('lgbm_learning_rate', 0.05),
                    num_leaves=lgbm_params.get('lgbm_num_leaves', 31),
                    objective='binary'
                )
            elif 'CAT' == vs:
                cat_params = cfg.get('cat', {})
                model_rnn = catboost_train(X_train, y_train, X_val, y_val,
                                           iterations=cat_params.get('iterations', 500),
                                           depth=cat_params.get('depth', 6),
                                           learning_rate=cat_params.get('learning_rate', 0.05),
                                           objective='Logloss')
                """
            elif 'XGB' == vs:
                xgb_params = cfg.get('xgb', {})
                models[vs] = xgb_train(
                    X_train, y_train, X_val, y_val,
                    learning_rate=xgb_params.get('xgb_learning_rate', 0.05),
                    max_depth=xgb_params.get('xgb_max_depth', 6),
                    objective='reg:squarederror'  # Changé de 'binary:logistic' à 'reg:squarederror'
                )
            elif 'LGBM' == vs:
                lgbm_params = cfg.get('lgbm', {})
                models[vs] = lgbm_train(
                    X_train, y_train, X_val, y_val,
                    learning_rate=lgbm_params.get('lgbm_learning_rate', 0.05),
                    num_leaves=lgbm_params.get('lgbm_num_leaves', 31),
                    objective='regression'        # Changé de 'binary' à 'regression'
                )
            elif 'CAT' == vs:
                cat_params = cfg.get('cat', {})
                models[vs] = catboost_train(
                    X_train, y_train, X_val, y_val,
                    iterations=cat_params.get('iterations', 500),
                    depth=cat_params.get('depth', 6),
                    learning_rate=cat_params.get('learning_rate', 0.05),
                    objective='RMSE')              # Changé de 'Logloss' à 'RMSE' (Root Mean Squared Error))
            elif 'TAB' == vs:
                from tabicl import TabICLRegressor
                models[vs] = TabICLRegressor()
            elif 'TABFF' == vs:
                process_id = os.getpid()
                output_dir = f"./ckpts_{process_id}"
                models[vs] = tabicl_fine_train(X_train, y_train, X_val, y_val, output_dir)
            elif 'TABFIN' == vs:
                from tabicl import TabICLRegressor
                # Supposons que vous avez votre train_df et vos features de base
                # 1. Obtenir les embeddings de la JEPA sous forme de tableau 2D [N_samples, embed_dim]
                # un modèle fin_jepa doit avoir déjà existe
                predictor = models['FINJEPA']
                # assert hasattr(predictor, 'model') and predictor.model is not None, "Model must exist before using TABFIN"
                predictor.model.eval()
                # 🚀 RESTRICTION : On ne garde que les 2000 dernières lignes pour soulager le CPU
                max_bars = 2000
                if len(X_train) > max_bars:
                    X_train_subset = X_train[-max_bars:]
                    train_df_subset = train_df.iloc[-max_bars:]
                    y_train_subset = y_train[-max_bars:]  # Important si y_train est un tableau numpy/série aligné
                else:
                    X_train_subset = X_train
                    train_df_subset = train_df
                    y_train_subset = y_train
                with torch.no_grad():
                    # On transforme tout X_train (ou train_df) en tenseur pour récupérer les embeddings latents
                    full_tensor = torch.tensor(X_train_subset, dtype=torch.float32).to(predictor.device)
                    # Pour chaque ligne, on extrait son embedding via l'encodeur de contexte de la JEPA
                    # (Si X_train est en 2D [N, Features], on peut l'encapsuler en fenêtres ou l'encoder directement selon la structure de votre encodeur)
                    latent_embeddings = predictor.model.context_encoder(full_tensor).cpu().numpy()

                # 2. Convertir les embeddings en DataFrame Pandas avec des noms de colonnes explicites
                embed_dim = latent_embeddings.shape[1]
                jepa_col_names = [f"jepa_latent_{i}" for i in range(embed_dim)]
                df_jepa_features = pd.DataFrame(latent_embeddings, columns=jepa_col_names, index=train_df_subset.index)

                # 3. Fusionner les features tabulaires classiques et les embeddings JEPA
                # On s'assure d'aligner les index proprement
                X_train_augmented = pd.DataFrame(X_train_subset, columns=features_cols, index=train_df_subset.index)
                X_train_augmented = pd.concat([X_train_augmented, df_jepa_features], axis=1)

                # Idem pour la validation (X_val) si vous utilisez TabICL Fine-Tuning (TABFF)
                # ... (répéter l'extraction pour X_val avec le même modèle JEPA gelé)
                model_tabicl = TabICLRegressor()
                model_tabicl.fit(X_train_augmented, y_train_subset)
                models[vs] = model_tabicl
            elif 'JEPA' == vs:
                jepa = cfg.get("jepa", None)

                # 🚀 RESTRICTION : On extrait la portion globale, puis on limite aux 2000 dernières lignes pour le CPU
                df_subset = df.iloc[:train_len + val_len]
                max_bars = 2000
                if len(df_subset) > max_bars:
                    df_jepa_subset = df_subset.iloc[-max_bars:]
                else:
                    df_jepa_subset = df_subset
                train_dataset, train_loader, test_loader = prepare_jepa(df_jepa_subset, jepa)
                model = jepa_train(train_loader, test_loader, jepa, trial)
                models[vs] = (model, train_dataset.stats)                #print(f"train_all_models: model={type(model)} stats={train_dataset.stats}")
            elif 'FINJEPA' == vs:
                from train.trainer import FinJepaPredictor
                max_bars = 2000
                # Paramètres configurables (ou valeurs par défaut)
                jepa_cfg = cfg.get('finjepa', {})
                ctx_len = jepa_cfg.get('context_len', 60)
                tgt_len = jepa_cfg.get('target_len', 15)
                batch_size = jepa_cfg.get('batch_size', 32)
                epochs = jepa_cfg.get('epochs', 3)
                lr = jepa_cfg.get('lr', 1e-4)
                # 1. Utilisation de X_train (qui est déjà scalé via X_scaler)
                # X_train est un tableau numpy de forme [N_samples, len(features_cols)]
                X_train_val = np.concatenate([X_train, X_val], axis=0)
                if len(X_train_val) > max_bars:
                    X_train_val = X_train_val[-max_bars:]
                # y_train_val = np.concatenate([y_train, y_val], axis=0)[-2000:]
                input_dim = X_train_val.shape[1]
                # 1. Initialiser le prédicteur FinJEPA
                predictor = FinJepaPredictor(input_dim=input_dim, lr=lr)
                # 2. Créer les fenêtres sur l'ensemble combiné (80% des données)
                context_batch, target_batch = create_jepa_windows(X_train_val, context_len=ctx_len, target_len=tgt_len)
                # 4. Boucle d'entraînement sur plusieurs époques ou une passe simple
                predictor.model.train()
                for epoch in range(epochs):
                    total_loss = 0.0
                    n_batches = 0
                    # Mélange optionnel des batchs à chaque époque pour plus de robustesse
                    indices = torch.randperm(len(context_batch))
                    for i in range(0, len(context_batch) - batch_size, batch_size):
                        batch_idx = indices[i: i + batch_size]
                        ctx_b = context_batch[batch_idx]
                        tgt_b = target_batch[batch_idx]
                        loss = predictor.train_step(ctx_b, tgt_b)
                        total_loss += loss
                        n_batches += 1
                    avg_loss = total_loss / max(1, n_batches)
                    print(f"FinJEPA [Epoch {epoch + 1}/{epochs}] - Loss moyenne: {avg_loss:.4f}")
                # 5. Stockage du modèle entraîné dans le dictionnaire de sortie
                models[vs] = predictor
        result = {'models': models,
                  'scaler': X_scaler,
                  'renko_test': test_df,
                  'config': cfg
                  }
        print(f"train_all_models result: {len(result)}")
        return result

    except Exception as e:
        print(f"Erreur dans train_all_models: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        reset_tf_memory()
        gc.collect()

def run_evaluation(models, data_test, config):
    pass
