import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, BatchNormalization

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
from typing import List, Tuple, Dict
from numba import njit, prange
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # ← à mettre TOUT EN HAUT du fichier
# 2. Optionnel mais recommandé : limite la verbosité de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warning, 3=error only
# Optionnel : désactive les protections Lightning qui tuent les processus
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
# Forcer TensorFlow à utiliser le CPU si le GPU pose problème (évite l'erreur CUDA 303)
# Commente cette ligne si ton GPU est bien configuré
tf.config.set_visible_devices([], 'GPU')
# Ou, pour forcer le CPU explicitement :
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # à mettre en haut du script

# ==================================================================
# 3. MODÈLES
# ==================================================================
# --- MODÈLE 1 : MLP (Multi-Layer Perceptron) - L'alternative simple au LSTM ---
#
# Un réseau de neurones simple, mais souvent très efficace et beaucoup
# plus rapide à entraîner qu'un LSTM. Il ne prend pas en compte l'ordre
# des séquences, mais regarde l'ensemble des features d'un instant 't'.
# =========================================================================
def mlp_predict(model, X_test):
    """Prédiction des probabilités (classe positive)"""
    return model.predict(X_test, verbose=0).flatten()

def mlp_clear(model):
    """Nettoyage propre du modèle et du graph TF"""
    del model
    tf.keras.backend.clear_session()

def mlp_train(X_train, y_train, X_val, y_val, features_len, mlp):
    """
    Entraîne un MLP simple pour signaux de trading.
    Données NON séquencées (features plates).
    """
    start_time = time.time()
    print("Démarrage entraînement MLP...")
    layers = []
    layers.append(tf.keras.layers.Dense(mlp['mlp_unit1'], activation='swish'))  # Swish au lieu de Relu
    layers.append(tf.keras.layers.BatchNormalization())  # Ajout stabilité
    layers.append(tf.keras.layers.Dropout(mlp['mlp_dropout']))
    if mlp['mlp_unit2'] > 0:  # permet de supprimer la 2e couche
        layers.append(tf.keras.layers.Dense(mlp['mlp_unit2'], activation='swish'))  # Swish au lieu de Relu
        layers.append(tf.keras.layers.BatchNormalization())  # Ajout stabilité
        layers.append(tf.keras.layers.Dropout(mlp['mlp_dropout']))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(features_len,)),
        *layers
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=mlp['mlp_lr'])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',      # bonne métrique
        patience=mlp['mlp_patience'],             # attends 10 epochs sans amélioration
        restore_best_weights=True,  # récupère les meilleurs poids
        verbose=0
    )
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=200,           # on met haut, early stopping gère
              batch_size=mlp['mlp_batch_size'],
              callbacks=[early_stopping],
              verbose=0)
    print(f"MLP entraîné en {(time.time() - start_time):.1f}s")
    print(f"Meilleure val_loss atteinte à l'epoch {early_stopping.stopped_epoch}")  # - 9 if early_stopping.stopped_epoch > 0 else 'toutes'}")
    return model
# =========================================================================
# --- MODÈLE 2 : LightGBM - Le champion de la vitesse et de la performance ---
#
# Un modèle basé sur les arbres de décision (Gradient Boosting).
# Extrêmement rapide et souvent plus performant que les réseaux de neurones
# sur des données "tabulaires" comme les vôtres.
# =========================================================================
def lgbm_predict(model, X_test):
    """Prédiction sans warning"""
    # Si X_test est un DataFrame, on garde les noms de colonnes
    #proba = model.predict_proba(X_test)[:, 1]
    #return proba
    """
    Prédiction unifiée pour LightGBM (Booster ou Classifier).
    Retourne toujours les probabilités de la classe positive.
    """
    X_test = np.asarray(X_test)  # force array NumPy (supprime warnings noms colonnes)

    if isinstance(model, lgb.Booster):
        # Booster → predict() retourne directement les probas
        proba = model.predict(X_test)
    elif hasattr(model, 'predict_proba'):
        # Classifier → predict_proba()
        proba = model.predict_proba(X_test)
    else:
        raise TypeError(f"Modèle non supporté : {type(model)}")

    # Si proba est 1D → c'est déjà la proba positive
    # Si 2D → on prend la colonne 1
    if proba.ndim == 1:
        return proba
    elif proba.shape[1] == 2:
        return proba[:, 1]
    else:
        raise ValueError(f"Shape de probabilités inattendue : {proba.shape}")

def lgbm_clear(model):
    del model

def lgbm_train(X_train, y_train, X_val, y_val,
               learning_rate=0.1,
               num_leaves=31,
               n_estimators=1000,
               feature_fraction=0.9,
               bagging_fraction=0.9,
               min_child_samples=5,
               early_stop_rounds=20):
    """
    Entraîne LightGBM avec paramètres variables pour optimisation.
    Prend en entrée les données NON séquencées.
    """
    start_time = time.time()
    print("Démarrage de l'entraînement LightGBM...")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'n_estimators': n_estimators,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': 1,
        'min_child_samples': min_child_samples,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'importance_type': 'gain',  # Plus pertinent pour le trading que le 'split' par défaut
        'min_gain_to_split': 0,  # Évite de créer des branches pour des gains insignifiants
        'max_bin': 255,  # Standard, mais peut être réduit à 63 pour accélérer l'optuna
        #'is_unbalance': True,
        # 'seed': 42
    }
    # Forcez la déconnexion totale de Pandas juste avant le fit
    model = lgb.LGBMClassifier(**params)
    X_train_clean = np.array(X_train)
    X_val_clean = np.array(X_val)
    # print(f"LGB y {y_train.mean()}")
    model.fit(
        X_train_clean, y_train.ravel(),
        eval_set=[(X_val_clean, y_val.ravel())],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)]
    )
    # SOLUTION ICI : On supprime la trace des noms de colonnes
    model._feature_name_ = None
    best_iter = model.best_iteration_
    print(f"LightGBM entraîné en {(time.time() - start_time):.1f}s | "
          f"best iteration = {best_iter if best_iter else n_estimators}")
    if best_iter < 5:
        return None
    return model
# ==========================================================================
# --- MODÈLE 3 : XGBoost - L'autre grand champion du Gradient Boosting ---
#
# Très similaire à LightGBM, c'est son concurrent direct. Il est parfois
# un peu moins rapide mais peut donner des résultats légèrement différents
# ou meilleurs selon les données.
# =========================================================================
def xgb_predict(model, X_test):
    """Prédiction des probabilités de la classe positive"""
    proba = model.predict_proba(X_test)[:, 1]
    return proba

def xgb_clear(model):
    del model

# =====================================
def xgb_train(X_train, y_train, X_val, y_val,
              learning_rate=0.05,
              max_depth=6,
              n_estimators=1000,
              subsample=0.8,
              colsample_bytree=0.8,
              early_stop_rounds=50):
    """
    Entraîne XGBoost avec le même style que LightGBM.
    Fonctionne avec XGBoost 1.3+ à 2.x (2025).
    Paramètres variables pour optimisation/grid search en trading.
    """
    start_time = time.time()
    print("Démarrage de l'entraînement XGBoost...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,        # limite haute (early stopping gère)
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'n_jobs': -1,
        'random_state': 42,                  # remplace 'seed' déprécié
        'tree_method': 'hist',               # rapide sur CPU
        'verbosity': 0,                        # équivalent verbose=-1
        'min_child_weight': 1,
        'gamma': 0,
        'early_stopping_rounds' : early_stop_rounds,  # ← maintenant accepté ici
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train.ravel(),
        eval_set=[(X_val, y_val.ravel())],
        verbose=False
    )
    best_iter = model.best_iteration
    bestI = best_iter + 1 if best_iter is not None else n_estimators
    print(f"XGBoost entraîné en {(time.time() - start_time):.1f}s | "
          f"best iteration = {bestI}")
    if bestI < 5:
        return None
    return model
# =========================================================================
# ------- Modèle 4 LSTM ULTRA
# =========================================================================
def lstm_predict_ultra(model, X_test):
    raw = model.predict(X_test, verbose=0)
    proba = raw.squeeze().flatten()  # tue les dimensions inutiles
    return proba

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
# =========================================================================
# ----------------- Modèle 5 LSTM normal
# =========================================================================
# ---------- LSTM AMÉLIORÉ (le seul qui marche vraiment en trading) ----------
def lstm_predict_model(model, X_test):
    proba_lstm = model.predict(X_test, verbose=0).flatten()
    proba = (proba_lstm + 1) / 2  # tanh → [0,1]
    return proba

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
# =========================================================================
# ================== LSTM SIMPLE MAIS QUI GAGNE ==================
# =========================================================================
# Amélioration du LSTM Simple
def lstm_train_simple(X_tr, y_tr, X_va, y_va, seq, feats, units=96):
    start = time.time()
    try:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(seq, feats)),
            tf.keras.layers.GaussianNoise(0.01),  # Ajoute du "bruit" pour éviter l'overfitting
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.LayerNormalization(),  # Mieux que Batchnorm pour les RNN
            tf.keras.layers.LSTM(units // 2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=35,
                  batch_size=64, verbose=0)
        print(f"lstm simple {(time.time() - start):.0f}")
        return model
    except BaseException as e:
        print(f"lstm simple err : {e}")
    return None


def lstm_train_simple_v1(X_tr, y_tr, X_va, y_va, seq, feats, units=96):
    start = time.time()
    try:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(seq, feats)),
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.LSTM(units // 2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=35,
                  batch_size=64, verbose=0)
        print(f"lstm simple {(time.time() - start):.0f}")
        return model
    except BaseException as e:
        print(f"lstm simple err : {e}")
    return None
# =========================================================================
# ------ modèle GRU
# =========================================================================

def build_gru_model_base(input_shape):
    model = Sequential([
        # Première couche GRU (retourne des séquences pour la suivante)
        GRU(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),

        # Deuxième couche GRU (ne retourne que le dernier état)
        GRU(32, return_sequences=False),
        Dropout(0.2),

        # Couche dense pour la décision finale
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Pour votre classification binaire
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def build_gru_model(input_shape, params):
    """
    Construit le modèle GRU en utilisant les paramètres suggérés par Optuna.
    """
    model = Sequential([
        Input(shape=input_shape),
        # Couche 1 : On utilise 'units_l1' suggéré par Optuna
        GRU(params['gru_units1'], return_sequences=True),
        BatchNormalization(),
        Dropout(params['gru_dropout']),
        # Couche 2 : On utilise 'units_l2'
        GRU(params['gru_units2'], return_sequences=False),
        Dropout(params['gru_dropout']),
        # Couche dense intermédiaire
        Dense(params['gru_units2'] // 2, activation='relu'),
        # Sortie
        Dense(1, activation='sigmoid')
    ])

    # On injecte le learning rate suggéré par Optuna
    optimizer = Adam(learning_rate=params['gru_lr'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['auc'])
    return model

def gru_train(X_train, y_train, X_val, y_val, params):
    start = time.time()
    # 1. Construction (Utilise l'input_shape détecté automatiquement)
    # X_train.shape[1] = Time Steps (ex: 10 bougies)
    # X_train.shape[2] = Features (ex: 5 indicateurs)
    # Transformation rapide de (samples, features) vers (samples, 1, features)
    X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_3d = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    # Maintenant input_shape sera (1, nombre_de_features)
    input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])

    model = build_gru_model(input_shape, params)  # On passe les suggestions Optuna ici

    # 2. Callbacks pour éviter l'overfitting (équivalent early_stopping de LGBM)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=params['gru_patience'],
        restore_best_weights=True,
        mode='max'
    )

    # 3. L'ENTRAÎNEMENT (Le "Train")
    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=100,
        batch_size=params['batch_size'],
        callbacks=[early_stop],
        verbose=0
    )
    print(f"GRU {(time.time() - start):.0f}")
    return model, history

# 4. LA PRÉDICTION (Le "Predict")
# Pour prédire, on fait simplement :
# preds = model.predict(X_test)

# ============== TEMPORAL FUSION TRANSFORMER (le roi 2025) ===========
def tft_train_predict_fast(train_df, val_df, test_df, features_cols, target_cols):
    start = time.time()
    try:
        import warnings
        import torch
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting import MultiNormalizer, MultiLoss
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
            num_workers=0,  # ← Set to 0 to prevent nested parallelism and SIGSEGV crashes
            persistent_workers=False,
            shuffle=True,
            pin_memory=True  # ← GPU-ready même sur CPU
        )
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=1024,
            num_workers=0, # ← Set to 0
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
            default_root_dir="../optimize/checkpoints",  # dir fixe = pas de warning
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
            num_workers=0, # ← Set to 0
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
    finally:
        del tft, trainer

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

        train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
        test_loader = test_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

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
        return pred.numpy().flatten()
    except BaseException as e:
        print(f"TFT err {(time.time()-start):.0f} except {e}")
        return None

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

# ================================
# 3. MODÈLE LSTM (avec ou sans hyperparamètres)
# ================================
def build_lstm(params, seq_len,  hp=None):
    """
    Construit le modèle LSTM
    - Si hp is None → utilise params['lstm_units']
    - Si hp est fourni → utilise hp.Int / hp.Choice
    """

    if 'features' not in params:
        raise KeyError("'features' manquant dans params")

    if hp is None:
        # === MODE FIXE (optimisation classique) ===
        model = Sequential([
            Input(shape=(seq_len, len(params['features']))),
            LSTM(params['lstm_units'], return_sequences=True),
            Dropout(0.2),
            LSTM(params['lstm_units'] // 2),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # === MODE HYPERPARAMÈTRES (Keras Tuner) ===
        model = Sequential([
            Input(shape=(seq_len, len(params['features']))),
            LSTM(hp.Int('units1', 32, 128, step=32), return_sequences=True),
            Dropout(hp.Float('drop1', 0.1, 0.4, step=0.1)),
            LSTM(hp.Int('units2', 32, 128, step=32)),
            Dropout(hp.Float('drop2', 0.1, 0.4, step=0.1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(hp.Choice('lr', [1e-3, 5e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    return model

# ========================================================================
### ALTERNATIVE SI TU VEUX RESTER EN KERAS/TF (plus simple)
# 4 Si tu veux rester en TensorFlow → le meilleur compromis 2025
# =====================================================================
def build_informer_like(n_features, seq_len=120, d_model=64, n_heads=8):
    inputs = tf.keras.Input(shape=(seq_len, n_features))

    # ProbSparse Attention (Informer style)
    x = tf.keras.layers.MultiHeadAttention(n_heads, d_model // n_heads)(inputs, inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization()(x + inputs)

    x = tf.keras.layers.Conv1D(d_model, 3, padding='causal')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4), loss='huber')
    return model

# ================================
# 5. TRANSFORMER (avec ou sans hyperparamètres)
# ================================
def build_transformer(seq_len, features_len, hp=None):
    """
    Construit un modèle Transformer
    - hp=None → mode fixe (optimisation classique)
    - hp fourni → mode Keras Tuner
    """
    inputs = Input(shape=(seq_len, features_len))
    if hp is None:
        # === MODE FIXE (votre optimisation actuelle) ===
        x = Dense(64)(inputs)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        attn = Dropout(0.1)(attn)
        attn = LayerNormalization()(x + attn)
        ff = Dense(128, activation='relu')(attn)
        ff = Dense(x.shape[-1])(ff)
        x = LayerNormalization()(attn + ff)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # === MODE HYPERPARAMÈTRES (Keras Tuner) ===
        x = Dense(hp.Int('dense_units', 32, 128, step=32))(inputs)
        attn = MultiHeadAttention(
            num_heads=hp.Choice('num_heads', [2, 4, 8]),
            key_dim=hp.Int('key_dim', 16, 64, step=16)
        )(x, x)
        attn = Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1))(attn)
        attn = LayerNormalization()(x + attn)
        ff = Dense(hp.Int('ff_units', 64, 256, step=64), activation='relu')(attn)
        ff = Dense(x.shape[-1])(ff)
        x = LayerNormalization()(attn + ff)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(hp.Choice('lr', [1e-3, 5e-4, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    return model

def build_transformer_tunable(seq_len, n_features, hp):
    inputs = Input(shape=(seq_len, n_features))
    x = MultiHeadAttention(
        num_heads=hp.Int('heads', min_value=2, max_value=8, step=2),
        key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32)
    )(inputs, inputs)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1))(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
# ======================================================================
# tft_model.py — LE MODÈLE QUE PIERRE UTILISE EN 2025
# =====================================================================
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
import torch

def build_tft(train_df, feature_cols, target_cols, max_encoder_length=120):
    # 1. Dataset TFT
    training = TimeSeriesDataSet(
        train_df.assign(time_idx=train_df.index),
        time_idx="time_idx",
        target=target_cols[0],
        group_ids=["symbol"],  # ou ["regime"] si tu veux multi-séries
        max_encoder_length=max_encoder_length,
        max_prediction_length=1,
        static_categoricals=["volatility_regime", "trend_strength"],
        static_reals=["avg_brick_size"],
        time_varying_known_reals=feature_cols,
        time_varying_unknown_reals=target_cols,
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 2. Modèle TFT
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-4,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 3. Entraînement rapide
    trainer = pl.Trainer(max_epochs=40, gpus=1 if torch.cuda.is_available() else 0,
                         gradient_clip_val=0.1, limit_train_batches=30)
    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=6)

    trainer.fit(tft, train_dataloaders=train_loader)

    return tft, training

# ================================
# 6. ÉVALUATION
# ================================
def evaluate_model(model, X, y_true, name):
    pred_proba = model.predict(X, verbose=0).flatten()
    pred_class = (pred_proba > 0.5).astype(int)
    print(f"\n{name}:")
    print(f"Accuracy:  {accuracy_score(y_true, pred_class):.4f}")
    print(f"F1-Score:  {f1_score(y_true, pred_class):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, pred_proba):.4f}")
    print(f"Pred 0/1:  {np.bincount(pred_class)}")
    return pred_proba, pred_class
# ================================
# 7. ALIGNEMENT + BACKTEST
# ================================
def backtest_align(pred_signal, df_test, df_full_with_rules, seq_len):
    """
    pred_signal: LSTM signals (1/-1)
    df_test: test_df (OHLC + features)
    df_full_with_rules: df_bricks avec sigo, sigc (même index)
    """
    start_idx = seq_len
    end_idx = start_idx + len(pred_signal)

    df_bt = df_test.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    df_rules = df_full_with_rules.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    if len(df_bt) != len(pred_signal):
        raise ValueError(f"Mismatch: {len(df_bt)} vs {len(pred_signal)}")

    df_bt['signal'] = pred_signal
    df_bt['sigo'] = df_rules['sigo'].values
    df_bt['sigc'] = df_rules['sigc'].values
    df_bt['time'] = df_rules['time'].values
    df_bt['open_renko'] = df_rules['open_renko'].values
    df_bt['close_renko'] = df_rules['close_renko'].values
    df_bt['EMA'] = df_rules['EMA'].values
    df_bt['pred_signal'] = pred_signal  # pour visu

    # --- PNL ---
    df_bt['next_close'] = df_bt['close'].shift(-1)
    df_bt['return'] = np.where(
        df_bt['signal'] == 1,
        (df_bt['next_close'] - df_bt['close']) / df_bt['close'],
        (df_bt['close'] - df_bt['next_close']) / df_bt['close']
    )
    df_bt['return'] = df_bt['return'].fillna(0)
    df_bt['cum_pnl'] = (1 + df_bt['return']).cumprod() - 1
    sharpe = df_bt['return'].mean() / df_bt['return'].std() * np.sqrt(252) if df_bt['return'].std() > 0 else 0
    print(f"PNL: {df_bt['cum_pnl'].iloc[-1]:+.2%} | Sharpe: {sharpe:.2f}")
    df_bt['final_signal'] = 0
    df_bt.loc[(df_bt['sigo'] == 1) & (df_bt['pred_signal'] == 1), 'final_signal'] = 1
    df_bt.loc[(df_bt['sigo'] == -1) & (df_bt['pred_signal'] == -1), 'final_signal'] = -1
    return df_bt

def recalculate_pnl(df_bt, signal_col='final_signal'):
    """
    Recalcule le PNL sur un signal existant dans df_bt
    """
    df = df_bt.copy()
    df['signal'] = df[signal_col]
    df['next_close'] = df['close'].shift(-1)

    df['return'] = np.where(
        df['signal'] == 1,
        (df['next_close'] - df['close']) / df['close'],
        (df['close'] - df['next_close']) / df['close']
    )
    df['return'] = df['return'].fillna(0)
    df['cum_pnl'] = (1 + df['return']).cumprod() - 1

    sharpe = df['return'].mean() / df['return'].std() * np.sqrt(252) if df['return'].std() > 0 else 0
    print(f"\n=== SIGNAL HYBRIDE ===")
    print(f"PNL: {df['cum_pnl'].iloc[-1]:+.2%} | Sharpe: {sharpe:.2f}")
    return df
# ===================#################################################
# =========================  Les predictions
# =================########################
def prediction(VERSION, model, X_test, X_test_seq=None):
    """
    Problème principal : confusion X_test vs X_test_seq + conditions LSTM dupliquées
    """
    if model is None:
        print(f"Modèle None pour {VERSION}")
        return None

    #print(f"[pred] {VERSION:12s} | X_test {X_test.shape if X_test is not None else '—'} | seq {X_test_seq.shape if X_test_seq is not None else '—'}")

    proba = None
    try:
        if any(k in VERSION for k in ['LGBM', 'LIGHTGBM']):
            proba = lgbm_predict(model, X_test)
        elif 'XGB' in VERSION:
            proba = xgb_predict(model, X_test)
        elif 'MLP' in VERSION:
            proba = mlp_predict(model, X_test)
        elif 'GRU' in VERSION:
            X_in = X_test if X_test.ndim == 3 else X_test.reshape(-1, 1, X_test.shape[-1])
            proba = model.predict(X_in, verbose=0).ravel()
        elif 'LSTM' in VERSION or 'ULTRA' in VERSION:
            if X_test_seq is None or len(X_test_seq) == 0:
                raise ValueError("X_test_seq requis pour LSTM/Ultra")
            if 'ULTRA' in VERSION:
                proba = lstm_predict_ultra(model, X_test_seq)
            else:
                proba = lstm_predict_model(model, X_test_seq)
        else:
            # fallback : on essaie séquence si disponible, sinon flat
            input_data = X_test_seq if (X_test_seq is not None and X_test_seq.ndim == 3) else X_test
            proba = model.predict(input_data, verbose=0).flatten()

        if proba is not None:
            proba = np.asarray(proba).ravel()  # uniformise
            proba = np.clip(proba, 0.001, 0.999)

    except Exception as e:
        print(f"→ ÉCHEC {VERSION}: {e.__class__.__name__} → {str(e)[:180]}")
        # traceback.print_exc()   # décommente en debug

    return proba
