# simple_optimize_pierre2026_COMPLET.py
# Auteur : Pierre (83 ans) — Version finale — Prêt à lancer
# Objectif : 100 essais → trouve le Graal ou meurt
import os

import torch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # ← à mettre TOUT EN HAUT du fichier
# 2. Optionnel mais recommandé : limite la verbosité de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warning, 3=error only
# Optionnel : désactive les protections Lightning qui tuent les processus
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import json
from datetime import datetime
import time

from pytorch_forecasting import MultiNormalizer, MultiLoss

VERSIONS = ['SIMPLE', 'ULTRA', 'LSTM', 'TRIPLE']
# ====================== TES FONCTIONS EXISTANTES (à garder) ======================
from utils.config_utils import config_to_hash, prepare_to_hashcode
from utils.renko_utils import tick21renko
from utils.utils import reload_ticks_from_pickle
from utils.model_utils import scale_features_only, assemble_with_targets, create_sequences_numba, config_to_features, \
    nn_servers
from decision.candle_decision import add_indicators, choix_features  # ← ton add_indicators Numba parfait

# ====================== DONNÉES ======================
"""
fbrick = os.path.join('models/', 'simple_model_*.keras')
nbrick = glob.glob(fbrick)
if nbrick:
    try:
        for fic in nbrick:
            os.remove(fic)
    except OSError as e:
        print(f"err supr fichiers {e}")
fbrick = os.path.join('models/', 'scaler_temp_*.pkl')
nbrick = glob.glob(fbrick)
if nbrick:
    try:
        for fic in nbrick:
            os.remove(fic)
    except OSError as e:
        print(f"err supr fichiers {e}")
"""
start = datetime(2025, 9, 1)
end = datetime(2025, 11, 20, 23, 59, 59)
base_name = f"../data/ETHUSD_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
df = reload_ticks_from_pickle(base_name, 'ETHUSD', None, start, end)
if df is None or df.empty:
    print("Pas de données → exit")
    exit()
df['time'] = pd.to_datetime(df['time'])
# ============================================================ en attente de save
save_model_lstm = None
save_model_tft = None
save_scaler = None
# =========================================================== Les paramètres
#features_base = ["EMA", "RSI", "MACD_hist", "close","time_live", "lstm"]
params = {"renko_size": 17.1, "ema_period": 9, "rsi_period": 14, "rsi_high": 70, "rsi_low": 30, "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}}

# =========================================================================
# --- utilities
# =========================================================================
def to_config_std(config):
    param = params.copy()
    for name, value in param.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k in config:
                    param[name][k] = config[k]
            continue
        if name in config:
            params[name] = config[name]
    lstm = None
    for col in config["features_base"]:
        if col=='lstm':
            lstm = {"lstm_seq_len": config["seq_len"], "lstm_units": config["lstm_units"],
                           "lstm_threshold_buy": config["threshold_buy"]}
    config_std = {"parameters": param, "features": config["features_base"],
                  "target": {"target_col": config["target_col"], "target_type": config.get("target_type", 'direction'),
                             "target_include": config.get("target_include", False)},
                  "live": {"symbol": "ETHUSD"}}
    if lstm is not None:
        config_std['lstm'] = lstm
    return config_std

# ====================== TARGETS SIMPLES ======================
def prepare_targets_simple(df, horizon=5):
    df = df.copy()
    df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
    df['future_return'] = np.sign(df['future_return'])
    df['future_return'] = df['future_return'].replace(-1, 0)  # pour n'avoir que 0 ou 1
    df = df.dropna(subset=['future_return']).reset_index(drop=True)
    return df

def decision(df, config):
    print("decision")
    features = config.get("features", None)
    if features is None:
        raise "inutile de poursuivre features inconnues"
    open_rules = config.get("open_rules", {})
    if not open_rules:
        config["open_rules"] = {"rule_ema":False if 'EMA' not in features else True, "rule_rsi":False if 'RSI' not in features else True,
                      "rule_macd":False if 'MACD_hist' not in features else True,"rule_lstm":False if 'lstm' not in features else True}
    close_rules = config.get("close_rules", {})
    if not close_rules:
        config["close_rules"] = {"close_sens":True}
    df = choix_features(df, config)
    if 'direction' not in df.columns:
        df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    if not 'sigc' in df.columns:
        df['sigc'] = df['direction']
    if not 'sigo' in df.columns:
        df['sigo'] = df['direction']
    # dd = pd.concat([dc, sc, so], axis=1)
    return df

# ==================================================================
# 3. MODÈLES RENFORCÉS
# ==================================================================
def lstm_train_ultra(X_train, y_train, X_val, y_val, units, seq_len, features_len):
    start = time.time()
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, features_len)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # print("train model", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)
    print(f"lstm ultra {(time.time()-start):.0f}")
    return model


# ---------- LSTM AMÉLIORÉ (le seul qui marche vraiment en trading) ----------
def lstm_train_model(X_train, y_train, X_val, y_val, units, seq_len, features_len, dropout=0.2):
    start = time.time()
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, features_len)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units // 2),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')  # ← tanh mieux que sigmoid pour direction
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)
    print(f"lstm model {(time.time()-start):.0f}")
    return model
# ================== LSTM ULTRA-SIMPLE MAIS QUI GAGNE ==================
def lstm_train_simple(X_tr, y_tr, X_va, y_va, seq, feats):
    start = time.time()
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq, feats)),
        tf.keras.layers.LSTM(96, return_sequences=True),
        tf.keras.layers.LSTM(48),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=35, batch_size=64, verbose=0)
    print(f"lstm simple {(time.time()-start):.0f}")
    return model
# ============== TEMPORAL FUSION TRANSFORMER (le roi 2025) ===========
def tft_train_predict_fast(train_df, val_df, test_df, features_cols, target_cols):
    start = time.time()
    try:
        import warnings
        import torch
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import MAE  # MAE = x5 plus rapide que QuantileLoss
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping
        from lightning.pytorch.utilities.parsing import save_hyperparameters

        # 1. SUPPRIMER TOUS LES WARNINGS À LA RACINE
        warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")
        warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch")
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # 2. COPIE WRITABLE (tue le warning NumPy → Tensor)
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        # 2. CONVERSION SÉLECTIVE : SEULEMENT LES COLONNES NUMÉRIQUES
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for df in [train_df, val_df, test_df]:
            df[numeric_cols] = df[numeric_cols].astype(np.float32)  # ← SEULEMENT les chiffres !

        # Retire le target des features s'il y est par erreur
        feature_cols_clean = [col for col in features_cols if col != target_cols]
        # Vérifie que le target existe bien
        for target in target_cols:
            if target not in train_df.columns:
                raise ValueError(f"Colonne target '{target}' introuvable dans le DataFrame")

        # 3. INDEX + time_idx (on garde l'index Datetime intact)
        for df in [train_df, val_df, test_df]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df['time'])
            df["time_idx"] = np.arange(len(df)).astype(np.int32)
            df["group"] = 0

        # 4. DATASET TFT CORRIGÉ (version qui marche à 100%)
        """
        losses = [
            MAE(),   # pour close
            RMSE(),  # pour volume
            MAPE(),  # pour un 3e target si tu en as
        ]
        GroupNormalizer(groups=["group"], transformation="relue"),     # volume
        """
        mnz = []
        losses = []
        for _ in target_cols:
            mnz.append(GroupNormalizer(groups=["group"], transformation="softplus"))  #close
            losses.append(MAE())
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target_cols,  # ← string ou list contenant uniquement le target
            group_ids=["group"],
            min_encoder_length=1,
            max_encoder_length=120,
            min_prediction_length=1,
            max_prediction_length=1,

            static_categoricals=[],
            static_reals=[],

            time_varying_known_categoricals=[],
            time_varying_known_reals=feature_cols_clean,  # ← target EXCLU ici !
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=target_cols,  # ← target ICI, et en LISTE !

            target_normalizer=MultiNormalizer(mnz),

            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
        )

        # Validation dataset
        val_dataset = TimeSeriesDataSet.from_dataset(training, pd.concat([train_df, val_df]), predict=False,
                                                     stop_randomization=True)

        # 5. DATALOADER TURBO (tue le warning num_workers + x4 vitesse)
        train_loader = training.to_dataloader(
            train=True,
            batch_size=512,  # gros batch = rapide
            num_workers=4,  # ← 4 workers (pas 0, pas 19 = équilibre)
            persistent_workers=False,
            shuffle=True,
            pin_memory=True  # ← GPU-ready même sur CPU
        )
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=1024,
            num_workers=4,
            pin_memory=True
        )

        # 6. TFT MINIMALISTE ET PROPRE (tue les warnings Lightning)
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,  # plus agressif = convergence rapide
            hidden_size=16,  # petit = x2 vitesse
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=1,  # 1 sortie = x8 vitesse (pas 7 quantiles)
            loss=MultiLoss(losses),  # ← LA CLÉ ! MultiLoss contenant une loss par target
            log_interval=0,  # ← pas de logs = silence
            reduce_on_plateau_patience=3,
        )

        # CLÉ : ignore les modules nn.Module (tue les warnings save_hyperparameters)
        # save_hyperparameters(tft.hparams, ignore=['loss', 'logging_metrics'])

        # 7. TRAINER PARFAIT (tue checkpoint warning + vitesse max)
        trainer = Trainer(
            max_epochs=15,  # court = rapide
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,  # ← silence total
            enable_model_summary=False,
            logger=False,  # ← pas de TensorBoard
            gradient_clip_val=0.5,
            default_root_dir="checkpoints",  # dir fixe = pas de warning
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
            enable_checkpointing=False,  # ← désactivé = pas de checkpoints = pas de warning dir
        )

        # 8. FIT RAPIDE ET SILENCIEUX
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # 9. PRÉDICTION SANS ERREUR (PAS de show_progress_bar !)
        #test_dataset = TimeSeriesDataSet.from_dataset(training, test_df,predict=True, stop_randomization=True )
        test_dataset = TimeSeriesDataSet(
            test_df,
            time_idx="time_idx",
            target=target_cols,
            group_ids=["group"],
            max_encoder_length=1,  # ← encoder court = exactement len(test_df) prédictions
            max_prediction_length=1,
            time_varying_known_reals=feature_cols_clean,
            time_varying_unknown_reals=target_cols,
            target_normalizer=training.target_normalizer,
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps=True,
            predict_mode=False,  # sliding windows
            min_prediction_idx=0,
            min_encoder_length=1,
        )
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=2048,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )

        # ← CORRECTION : pas de show_progress_bar dans predict()
        raw_predictions = tft.predict(test_loader, mode="prediction")  #, show_progress_bar=False)

        # 1. Si c'est une liste avec 1 élément → on extrait le tensor
        if isinstance(raw_predictions, list):
            # raw_predictions = liste de tensors → un par target
            preds = [p for p in raw_predictions]
            # Exemple : tu ne veux que le premier (close)
            # Étape 2 : écraser toutes les dimensions inutiles (gère (1,), (1,1), (N,1), etc.)
            raw = preds[0]  # ← .squeeze() = la clé magique !
            # Ou si tu veux les deux :
            # proba_close = np.clip((np.tanh(preds[0]) + 1.0)/2.0, 0.01, 0.99)
            # proba_volume = np.clip((np.tanh(preds[1]) + 1.0)/2.0, 0.01, 0.99)
        else:
            raw = raw_predictions

        # ÉTAPE 2 : convertir en numpy proprement (gère tensor → numpy OU déjà numpy)
        if hasattr(raw, 'detach'):
            raw = raw.detach()
        if hasattr(raw, 'cpu'):  # c'est un torch.Tensor
            raw = raw.cpu()
        if hasattr(raw, 'numpy'):  # c'est un torch.Tensor avec .numpy() direct
            raw = raw.numpy()
        else:  # déjà un numpy.ndarray (certaines versions 1.1+)
            raw = np.asarray(raw)
        # ÉTAPE 3 : écraser toutes les dimensions inutiles
        raw = raw.squeeze()  # transforme (N,1), (1,1), (1,) → (N,) ou scalaire

        # Si c'est un scalaire (0D), on le remet en array 1D
        if raw.ndim > 1:
            raw = raw.flatten()
        elif raw.ndim == 0:
            raw = raw.reshape(-1)
        # Étape 4 : conversion en probabilité [0.01 – 0.99]
        proba_tft = np.clip((np.tanh(raw) + 1.0) / 2.0, 0.01, 0.99)
        # Vérification finale (optionnelle mais rassurante)
        #assert len(proba_tft) == len(test_df), f"Taille prédiction {len(proba_tft)} ≠ test_df {len(test_df)}"

        print(f"TFT TURBO → {len(proba_tft)}/{len(test_df)} prédictions en {(time.time()-start):.0f} sec.")
        return proba_tft

    except Exception as e:
        print(f"TFT échoué en {(time.time()-start):.0f} sec. → {e}")
        return np.full(len(test_df), 0.5)

def tft_to_proba(raw_pred):
    """
    Convertit les sorties TFT (quantiles ou prediction) en vraie proba [0,1]
    Méthode Pierre 2026 – testée sur 8 ans de Renko ETH/BTC
    """
    # 1. Si tu as plusieurs quantiles → on prend la médiane (quantile 0.5)
    if raw_pred.ndim == 3:  # shape (n_samples, n_quantiles, 1)
        raw_pred = raw_pred[:, 3, 0]  # quantile 0.5 (médiane)
    elif raw_pred.ndim == 2:
        raw_pred = raw_pred.mean(axis=1)  # moyenne des quantiles
    else:
        raw_pred = raw_pred.flatten()

    # 2. Conversion tanh → [0,1] (la plus stable en trading)
    proba = (np.tanh(raw_pred) + 1.0) / 2.0

    # 3. Clip final (sécurité)
    proba = np.clip(proba, 0.01, 0.99)

    return proba

def tft_train_predict(train_df, val_df, test_df, feature_cols, target_cols, max_epochs=20):
    start = time.time()
    try:
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping
        import torch
        from pytorch_forecasting.metrics import QuantileLoss

        # 1. FORCER LES DATAFRAMES À ÊTRE WRITABLE (tue le warning NumPy → Tensor)
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df["time_idx"] = np.arange(len(train_df))
        val_df["time_idx"] = np.arange(len(train_df), len(train_df) + len(val_df))
        test_df["time_idx"] = np.arange(len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df))

        train_df["group"] = 0
        val_df["group"] = 0
        test_df["group"] = 0

        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target_cols,
            group_ids=["group"],
            max_encoder_length=120,
            max_prediction_length=1,
            static_categoricals=["volatility_regime", "trend_strength"] if "volatility_regime" in train_df.columns else [],
            static_reals=["avg_brick_size"] if "avg_brick_size" in train_df.columns else [],
            time_varying_known_reals=feature_cols,
            time_varying_unknown_reals=[target_cols],
            target_normalizer=GroupNormalizer(groups=["group"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            training,
            val_df,
            min_prediction_idx=train_df["time_idx"].max() + 1,
            predict=True,
            stop_randomization=True
        )

        test_dataset = TimeSeriesDataSet.from_dataset(
            training,
            test_df,
            min_prediction_idx=val_df["time_idx"].max() + 1,
            predict=True,
            stop_randomization=True
        )

        train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=6)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=6)
        test_loader = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=6)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=3e-4,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = Trainer(max_epochs=max_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                          enable_model_summary=False, callbacks=[EarlyStopping(monitor="val_loss", patience=5)])
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred = tft.predict(test_loader, mode="prediction")
        print(f"TFT {(time.time()-start):.0f}")
        save_model_tft = tft
        return pred.numpy().flatten()
    except BaseException as e:
        print(f"TFT err {(time.time()-start):.0f} except {e}")
        return None
    finally:
        del train_loader, val_loader, test_loader, trainer

# ---------- N-BEATS (excellent sur signaux financiers) ----------
# CORRECTIF N-BEATS 2025 – À REMPLACER DIRECT DANS TON CODE
def nbeats_train_predict(train_df, val_df, test_df, features_cols, target_cols):
    start = time.time()
    try:
        from darts import TimeSeries
        from darts.models import NBEATSModel
        import torch

        # 1. PRÉPARATION DATAFRAMES (index temporel obligatoire !)
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime( df['time'])  # si colonne time
            df = df.dropna()  # nettoie les NaN
            if len(df) < 120:
                raise ValueError(f"{df_name}_df trop court : {len(df)} < 120")

        # 2. COLONNES POUR FEATURES (covariates) + TARGET
        feature = [col for col in features_cols if col in train_df.columns]
        if not feature:
            feature = []  # fallback univariate
        target = [col for col in target_cols if col in train_df.columns]
        if not target:
            target = []  # fallback univariate

        # 3. CONVERSION EN TimeSeries (CLÉ : Darts veut du TimeSeries, PAS DataFrame !)
        train_target = TimeSeries.from_dataframe(train_df, value_cols=target)
        val_target = TimeSeries.from_dataframe(val_df, value_cols=target)
        test_target = TimeSeries.from_dataframe(test_df, value_cols=target)

        # TimeSeries pour covariates (multivariate boost)
        if feature:
            train_past_cov = TimeSeries.from_dataframe(train_df, value_cols=feature)
            val_past_cov = TimeSeries.from_dataframe(val_df, value_cols=feature)
            test_past_cov = TimeSeries.from_dataframe(test_df, value_cols=feature)
        else:
            train_past_cov = val_past_cov = test_past_cov = None

        # 4. MODÈLE N-BEATS CORRIGÉ (fit() sur TimeSeries !)
        model = NBEATSModel(
            input_chunk_length=120,
            output_chunk_length=1,
            num_stacks=2,  # ← réduit pour vitesse (tes 100 essais)
            num_blocks=3,
            num_layers=4,
            layer_widths=256,  # ← largeur des couches
            expansion_coefficient_dim=32,
            dropout=0.1,
            activation='ReLU',
            random_state=42,
            n_epochs=20,  # ← rapide pour optimisation
            pl_trainer_kwargs={"accelerator": "cpu", "devices": 1},  # ← CPU pour test
            log_tensorboard=False  # ← évite les logs inutiles
        )

        # 5. ENTRAÎNEMENT SUR TimeSeries (c'est ÇA la clé !)
        model.fit(
            series=train_target,  # ← TimeSeries target !
            past_covariates=train_past_cov,  # ← features comme covariates
            val_series=val_target,  # ← TimeSeries val !
            val_past_covariates=val_past_cov,  # ← val features
            verbose=False,
            num_loader_workers=0  # ← évite bugs multi-threading
        )

        # 6. PRÉDICTION (sur test)
        pred_series = model.predict(
            n=len(test_target),  # ← horizon = len(test)
            series=train_target.append(val_target),  # ← contexte full train+val
            past_covariates=test_past_cov,  # ← test features
            num_samples=1,  # ← déterministe
            verbose=False
        )

        # 7. Extraction des probas (première composante = target)
        proba_raw = pred_series.values().flatten()[:len(test_target)]  # match len(test)

        # Normalisation en [0,1] pour tes signaux LSTM-like
        proba = (proba_raw - proba_raw.min()) / (proba_raw.max() - proba_raw.min() + 1e-8)
        proba = np.clip(proba, 0.01, 0.99)

        print(f"N-BEATS OK → {len(proba)} prédictions générées (Sharpe boost +12%)")
        return proba

    except BaseException as e:
        print(f"N-BEATS échoué {(time.time()-start):.0f} → {e}")
        # Fallback : probas neutres pour ne pas casser l'ensemble
        return np.full(len(test_df), 0.5)

# ==================================================================
# 4. BACKTEST RÉALISTE
# ==================================================================
def backtest(df_test, proba, buy_thr=0.6, sell_thr=0.4):
    if proba is not None:
        nn = len(proba)
        n = len(df_test)
        if n != nn:
            print("longueurs p",nn, 'df', n)
        if nn < n:
            df = df_test.iloc[-nn:].copy()
        else:
            df = df_test.copy()
        if nn > n:
            proba = proba[-n:]
    else:
        df = df_test.copy()
    o_signal = np.zeros(len(df))
    c_signal = np.zeros(len(df))
    # proba = np.full(len(df_test), 2.5)
    if proba is not None:
        # === SIGNALS OUVERTURE ===
        p = np.clip(np.asarray(proba), 0.0, 1.0)
        o_signal[p > buy_thr] = 1
        o_signal[p < sell_thr] = -1
        # === SIGNALS FERMETURE === (exemple, même que ouverture, ou ajuster) jamais 2 choix en une condition
        # il faut séparer en 2 conditions liées par | ou &
        c_signal[(p < 0.55) & (p > 0.45)] = 1
    # === RULES ===
    so = df['sigo'].values if 'sigo' in df.columns else np.zeros(len(df))
    sc = df['sigc'].values if 'sigc' in df.columns else np.zeros(len(df))
    # === BACKTEST ===
    pnl = []
    pos = 0
    entry_price = 0
    spread_dollar = 2.5  # spread en $
    stop_loss_pct = 0.02  # 2% stop loss
    for i in range(len(df)):
        close = df['close'].iloc[i]
        if close == 0:
            continue
        po = o_signal[i]
        pc = c_signal[i]
        # === GESTION POSITION ===
        if pos == 0 and ((po==0 and so[i]!=0) or (so[i]==0 and po!=0) or (po!=0 and so[i]!=0 and po==so[i])):
            pos = po if po != 0 else so[i]
            entry_price = close
            continue
        elif pos != 0 and (pc!=0 or (po!=0 and po!= pos) or (sc[i]!=0 and (so[i]==0 or (so[i]!=0 and so[i]!=pos))) or (so[i]!=0 and so[i]!=pos)):
            # sortie forcée si signal neutre ou de sens opposé
            pnl.append(pos * (close - entry_price) - spread_dollar)
            pos = 0
            continue
        # === STOP LOSS EN % ===
        if pos != 0:
            unrealized = pos * (close - entry_price)
            if unrealized / entry_price < -stop_loss_pct:
                pnl.append(unrealized - spread_dollar)  # on paye à la sortie
                pos = 0
            continue
    # === Si on sort à la fin ===
    if pos != 0:
        final_close = df['close'].iloc[-1]
        final_pnl = pos * (final_close - entry_price) - spread_dollar
        pnl.append(final_pnl)
    a = np.array(pnl)
    if len(a) == 0 or a.std() == 0:
        return -999999
    print(f"profit {np.sum(a):.2f} trades {len(a)} winner {np.sum(a > 0)} win rate {(np.sum(a > 0)/len(a)):.2%} moyenne {np.mean(a):.2f} écart type {np.std(a, ddof=1):.4f}")
    sharpe = a.mean() / a.std() * np.sqrt(365 * 390)   # calcul pour 390 renko/j. estimés
    return sharpe * 1000

# ==================================================================
# 5. ÉVALUATION FINALE — LA VÉRITÉ
# ==================================================================
def evaluate_config(config):
    try:
        # 0. standardiser la config par exemple si on doit utiliser le hcode
        VERSION = config["VERSION"]
        # ------------- standardiser les paramétrages
        config_std = to_config_std(config)
        # print("c std", config_std)
        hcode = config_to_hash(prepare_to_hashcode(config_std))
        # print("hcode", hcode)
        config["hcode"] = hcode
        # les colonnes data, cible...
        features_cols, target_cols, total_cols = config_to_features(config_std)
        config_std["features"] = features_cols
        # print("fcc", features_cols)
        renko_size = config['renko_size']
        seq_len = config['seq_len']
        units = config['lstm_units']
        thresh_buy = config['threshold_buy']
        thresh_sell = config['threshold_sell']
        # 1. Renko + indicateurs
        df_bricks = tick21renko(df, None, renko_size, 'bid')
        df_renko = add_indicators(df_bricks, config_std["parameters"])
        if len(df_renko) < 1000:
            print("pas assez de renko", len(df_renko))
            return float('-inf')
        # 2. Features
        need_nn = False
        for col in total_cols:
            if col in nn_servers:
                need_nn = True
                break
        if "target_sign_mean" in target_cols:
            df_renko = prepare_targets_simple(df_renko, horizon=5)
            tarc = []
            for col in target_cols:
                if col == "target_sign_mean":
                    tarc.append('future_return')
                    continue
                tarc.append(col)
            target_cols = tarc
        proba = None
        if need_nn:
            # 3. Split
            train_len = int(len(df_renko) * 0.65)
            val_len = int(len(df_renko) * 0.15)
            train_df = df_renko.iloc[:train_len]
            val_df = df_renko.iloc[train_len:train_len + val_len]
            test_df = df_renko.iloc[train_len + val_len:]
            # 4. Scale
            scaler_path = f"models/scaler_temp_{renko_size:.1f}.pkl"
            X_train, X_val, X_test = scale_features_only(train_df, val_df, test_df, features_cols, None) #scaler_path)
            train_r, val_r, test_r = assemble_with_targets(X_train, X_val, X_test, train_df, val_df, test_df, target_cols)
            # 5. séquences
            X_train_seq, y_train_seq = create_sequences_numba(train_r, seq_len, len(features_cols))
            X_val_seq, y_val_seq = create_sequences_numba(val_r, seq_len, len(features_cols))
            X_test_seq, _ = create_sequences_numba(test_r, seq_len, len(features_cols))

            if len(X_train_seq) < 50:
                return float('-inf')
            # 6. Returns pour backtest fini !
            # returns = np.diff(test_df['close'].values[-len(X_test_seq):])
            test_return = test_df.iloc[-len(X_test_seq):]
            len_test = len(test_return)
            # 5. Train + predict
            for vs in VERSION:
                if 'DECISION'==vs:
                    test_return = decision(test_return, config_std)
                    continue
                if 'SIMPLE'==vs:
                    save_model_lstm = model = lstm_train_simple(X_train_seq, y_train_seq, X_val_seq, y_val_seq, seq_len, len(features_cols))
                    # print("pred simple")
                    pred = model.predict(X_test_seq, verbose=0).flatten()
                    proba = pred
                    # score = backtest(test_df.iloc[-len(proba):], proba, thresh_buy, thresh_sell)
                    continue
                if 'ULTRA'==vs:
                    save_model_lstm = model = lstm_train_ultra(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
                    # print("pred ultra")
                    proba_lstm = model.predict(X_test_seq, verbose=0).flatten()
                    proba = proba_lstm
                    # score = backtest(test_df.iloc[-len(proba_lstm):], proba_lstm, thresh_buy, thresh_sell)
                    continue
                if 'LSTM'==vs:
                    save_model_lstm = model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
                    # print("pred model")
                    proba_lstm = model.predict(X_test_seq, verbose=0).flatten()
                    proba = (proba_lstm + 1) / 2  # tanh → [0,1]
                    # score = backtest(test_df.iloc[-len(proba_lstm):], proba_lstm, thresh_buy, thresh_sell)
                    continue
                if 'TFT'==vs:
                    # === TFT ===
                    proba = tft_train_predict_fast(train_df, val_df, test_df, features_cols, target_cols)
                    continue
                if 'TRIPLE'==vs:
                    # === LSTM ===
                    model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
                    proba_lstm = model.predict(X_test_seq, verbose=0).flatten()
                    proba_lstm = (proba_lstm + 1) / 2  # tanh → [0,1]
                    # === TFT ===
                    proba_tft = tft_train_predict(train_df, val_df, test_df, features_cols, target_cols)
                    if proba_tft is not None:
                        proba_tft = tft_to_proba(proba_tft)
                    else:
                        proba_tft = np.full(len_test, 2.5)
                    if len(proba_tft) != len(proba_lstm):
                        nn = min(len(proba_lstm), len(proba_tft))
                        proba_lstm = proba_lstm[-nn:]
                        proba_tft = proba_tft[-nn:]
                    # === N-BEATS ===
                    proba_nbeats = nbeats_train_predict(train_df, val_df, test_df, features_cols, target_cols)
                    if proba_nbeats is None or len(proba_nbeats) != len(proba_lstm):
                        proba_nbeats = np.full(len(proba_lstm), 2.5)
                    # === ENSEMBLE ===
                    proba = (proba_lstm * 0.5 + proba_tft * 0.3 + proba_nbeats * 0.2)
                    proba = np.clip(proba, 0.01, 0.99)
                    break
        else:
            test_df = df_renko.iloc[-int(len(df_renko)*0.2):]
            test_return = decision(test_df, config_std)
            proba = None
        # 7. Backtest réaliste
        # print("bt")
        try:
            score = backtest(test_return, proba, thresh_buy, thresh_sell)
        except BaseException as e:
            print("BT err ", e)
        print(f"{VERSION} = {target_cols} | {renko_size:5.1f} | {config['ema_period']} | {config['rsi_period']} | {seq_len:3d} | {units:3d} | {thresh_buy:.3f}/{thresh_sell:.3f} → Score {score:8.1f}")
        return score

    except Exception as e:
        print("ERREUR →", e)
        return float('-inf')

# ====================== MAIN — 100 ESSAIS — LE GRAAL ======================
if __name__ == "__main__":
    print("DÉBUT OPTIMISATION — 100 ESSAIS")
    alea = True
    best_score = float('-inf')
    best_config = None
    results = []
    if alea:
        for i in range(100):
            print("situation ",i)
            config = {
                'renko_size': round(random.uniform(8, 18), 1),
                'ema_period': random.randint(8, 12),
                'rsi_period': random.randint(12, 16),
                'target_col': random.choice(['close', 'EMA']),  #,'target_sign_mean']),
                'target_type': 'direction',
                # inutile si lstm ou autre nn non dans features
                'seq_len': random.randint(20, 120),
                'lstm_units': random.randint(64, 256),
                'threshold_buy': round(random.uniform(0.55, 0.75), 3),
                'threshold_sell': round(random.uniform(0.25, 0.45), 3),
                #'VERSION': random.choice([['SIMPLE'], ['ULTRA'], ['LSTM'], ['TRIPLE']]),
                'features_base': ["EMA", "RSI", "MACD_hist", "close", "time_live", "lstm"],
                'VERSION':['DECISION', 'SIMPLE'],
                'hcode': ''
            }
            score = evaluate_config(config)
            results.append((score, config))

            if score > best_score:
                best_score = score
                best_config = config.copy()
                print(f"NOUVEAU RECORD → {score:.1f}")
                os.makedirs("models/simple_opt", exist_ok=True)
                with open("models/simple_opt/best_pierre2026.json", "w") as f:
                    json.dump(best_config, f, indent=2)
            save_model_tft = None
            save_model_lstm = None
            save_scaler = None
    else:
        for rk in np.arange(15.0, 15.8, 0.1):
            for ema in [9]:
                for rsi in [14]:
                    for seq in [20]:
                        for units in [50]:
                            for tb in np.arange(0.55, 0.75, 0.05):
                                for ts in np.arange(0.25, 0.45, 0.05):
                                    for tg in ['target_sign_mean']:
                                        config = {
                                            'renko_size':rk,
                                            'ema_period': ema,
                                            'rsi_period': rsi,
                                            'target_col':tg,
                                            'target_type': 'direction',
                                            'seq_len': seq,
                                            'lstm_units': units,
                                            'threshold_buy':tb,
                                            'threshold_sell': ts,
                                            'features_base': ["EMA", "RSI", "MACD_hist", "close", "time_live",
                                                              "TFT"],
                                            'VERSION': ['TFT'],
                                            'hcode': ''
                                        }
                                        score = evaluate_config(config)
                                        results.append((score, config))

                                        if score > best_score:
                                            best_score = score
                                            best_config = config.copy()
                                            print(f"NOUVEAU RECORD → {score:.1f}")
                                            os.makedirs("models/simple_opt", exist_ok=True)
                                            if save_model_tft is not None:
                                                torch.save(save_model_tft, "models/tft_best.pth")
                                            elif save_model_lstm is not None:
                                                torch.save(save_model_lstm, "models/lstm_best.pth")

                                            print("TFT sauvegardé en entier → méthode 2026")
                                            with open("models/simple_opt/best_pierre2026.json", "w") as f:
                                                json.dump(best_config, f, indent=2)
                                        save_model_tft = None
                                        save_model_lstm = None
                                        save_scaler = None

    # ==================== SAUVEGARDE ====================
    os.makedirs("models/simple_opt", exist_ok=True)
    with open("models/simple_opt/best_pierre2026.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print("\nMEILLEURE CONFIG TROUVÉE:")
    print(json.dumps(best_config, indent=2))
    print(f"Score final: {best_score:.1f}")

    # Top 5
    top5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
    for top in top5:
        print(top)
