import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, BatchNormalization
import optuna
from optuna.integration import TFKerasPruningCallback

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
#from typing import List, Tuple, Dict
#from numba import njit, prange
#from tensorflow.python.profiler.profiler_client import monitor

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
    layers = []
    layers.append(tf.keras.layers.Dense(mlp['mlp_unit1'], activation='swish'))  # Swish au lieu de Relu
    layers.append(tf.keras.layers.BatchNormalization())  # Ajout stabilité
    layers.append(tf.keras.layers.Dropout(mlp['mlp_dropout']))
    if mlp['mlp_unit2'] > 0:  # permet de supprimer la 2e couche
        layers.append(tf.keras.layers.Dense(mlp['mlp_unit2'], activation='swish'))  # Swish au lieu de Relu
        layers.append(tf.keras.layers.BatchNormalization())  # Ajout stabilité
        layers.append(tf.keras.layers.Dropout(mlp['mlp_dropout']))
    layers.append(tf.keras.layers.Dense(1, activation='linear'))
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(features_len,)),
        *layers
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=mlp['mlp_lr'])
    model.compile(optimizer=optimizer,
                  loss='mae',
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
def lgbm_train(X_train, y_train, X_val, y_val,
               learning_rate=0.05, num_leaves=31, objective='regression', **kwargs):
    """
    Entraînement LightGBM - Supporte à la fois régression et classification
    """
    from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping

    # Détection du type de tâche
    if objective in ['regression', 'reg:squarederror', 'reg:absoluteerror']:
        model = LGBMRegressor(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            objective=objective,
            n_estimators=500,
            random_state=42,
            verbose=-1,
            **kwargs
        )
    else:
        model = LGBMClassifier(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            objective=objective,
            n_estimators=500,
            random_state=42,
            verbose=-1,
            **kwargs
        )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False)]
    )
    return model


def lgbm_trainx(X_train, y_train, X_val, y_val,
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
        'reg_alpha': 0.1,     # Régularisation L1 (Anti-overfitting sur bruit)
        'reg_lambda': 1.0,    # Régularisation L2 (Anti-overfitting sur bruit)
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'importance_type': 'gain',  # Plus pertinent pour le trading que le 'split' par défaut
        'min_gain_to_split': 0.01,  # Évite de créer des branches pour des gains insignifiants
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
def xgb_clear(model):
    del model

# =====================================
def xgb_train(X_train, y_train, X_val, y_val,
              learning_rate=0.05, max_depth=6, objective='reg:squarederror', **kwargs):
    from xgboost import XGBRegressor, XGBClassifier
    import numpy as np

    is_reg = objective in ['reg:squarederror', 'regression']
    ModelClass = XGBRegressor if is_reg else XGBClassifier

    # 1. On définit le modèle AVEC early_stopping_rounds ici
    model = ModelClass(
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective=objective,
        n_estimators=1000, # Augmentez un peu pour laisser l'early stopping agir
        early_stopping_rounds=50, # <--- C'EST ICI QU'IL DOIT ÊTRE
        random_state=42,
        verbosity=0,
        **kwargs
    )

    # 2. On retire early_stopping_rounds de .fit()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def xgb_trainx(X_train, y_train, X_val, y_val,
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
def lstm_train_ultra(X_train, y_train, X_val, y_val, units, seq_len, features_len, trial=None, batch_size=128):
    start = time.time()
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, features_len)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mae', jit_compile=True)
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)]
    if trial is not None:
        callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
        
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=callbacks, batch_size=batch_size, verbose=0)
    print(f"lstm ultra {(time.time()-start):.0f}s")
    return model
# =========================================================================
# ----------------- Modèle 5 LSTM normal
# =========================================================================
# ---------- LSTM AMÉLIORÉ (le seul qui marche vraiment en trading) ----------
def lstm_train_model(X_train, y_train, X_val, y_val, units, seq_len, features_len, dropout=0.2, trial=None, batch_size=128):
    start = time.time()
    l2_reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, features_len)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, kernel_regularizer=l2_reg)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units // 2, kernel_regularizer=l2_reg),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation='swish', kernel_regularizer=l2_reg),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

    model.compile(optimizer=optimizer, loss='mae', metrics=['AUC'], jit_compile=True)
    
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=0
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        lr_reducer
    ]
    if trial is not None:
        callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
        
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=batch_size, callbacks=callbacks, verbose=0)
    print(f"lstm model {(time.time()-start):.0f}s")
    return model
# =========================================================================
# ================== LSTM SIMPLE MAIS QUI GAGNE ==================
# =========================================================================
# Amélioration du LSTM Simple
def lstm_train_simple(X_tr, y_tr, X_va, y_va, seq, feats, units=96, trial=None, batch_size=128):
    start = time.time()
    try:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(seq, feats)),
            tf.keras.layers.GaussianNoise(0.01),  # Ajoute du "bruit" pour éviter l'overfitting
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.LayerNormalization(),  # Mieux que Batchnorm pour les RNN
            tf.keras.layers.LSTM(units // 2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mae', jit_compile=True)
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)]
        if trial is not None:
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
            
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=100, callbacks=callbacks,
                  batch_size=batch_size, verbose=0)
        print(f"lstm simple {(time.time() - start):.0f}s")
        return model
    except BaseException as e:
        print(f"lstm simple err : {e}")
    return None

# =========================================================================
# ------ modèle GRU
# =========================================================================
def gru_train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, params):
    start = time.time()
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    l2_reg = tf.keras.regularizers.l2(1e-4)

    model = Sequential([
        Input(shape=input_shape),

        # GRU Bidirectionnel
        tf.keras.layers.Bidirectional(
            GRU(params['gru_units1'], return_sequences=True, kernel_regularizer=l2_reg)
        ),
        # BatchNormalization est souvent préférable pour stabiliser les activations
        tf.keras.layers.BatchNormalization(),
        Dropout(params['gru_dropout']),

        GRU(params['gru_units2'], return_sequences=False, kernel_regularizer=l2_reg),
        tf.keras.layers.BatchNormalization(),
        Dropout(params['gru_dropout']),

        # Dense avec Swish
        Dense(params['gru_units1'], activation='swish', kernel_regularizer=l2_reg),
        # Sortie sigmoïde classique sans lissage forcé
        #Dense(1, activation='sigmoid')
        Dense(1, activation='linear', bias_initializer=tf.keras.initializers.Constant(-2.0))
    ])

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=params['gru_lr'],
        weight_decay=1e-4
    )
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    # Suppression du label_smoothing pour permettre des prédictions fortes (proches de 0 ou 1)
    #loss_fn = tf.keras.losses.BinaryCrossentropy()
    #model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'], jit_compile=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params.get('gru_patience', 10),
        restore_best_weights=True
    )

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=0
    )
    # Calculez le poids des classes avant le fit
    from sklearn.utils import class_weight

    # Supposons que y_train_seq soit votre vecteur cible
    # Si c'est un array 3D, aplatissez-le juste pour le calcul
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_seq),
        y=y_train_seq.flatten()
    )
    class_weight_dict = {0: weights[0], 1: weights[1]}

    # Dans model.fit :
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=params['batch_size'],
        callbacks=[early_stop, lr_reducer],
        class_weight=class_weight_dict,
        verbose=0
    )

    print(f"GRU entraîné en {(time.time() - start):.0f}s")
    return model

# 4. LA PRÉDICTION (Le "Predict")
# Pour prédire, on fait simplement :
# preds = model.predict(X_test)

# =========================================================================
# --- MODÈLE TabICL : Tabular In-Context Learning ---
# =========================================================================
from tabicl import TabICLRegressor

def tabicl_train(X_train, y_train):
    """
    Entraîne le modèle TabICLRegressor sur les données normalisées non séquencées.
    y_train est aplati avec ravel() pour correspondre aux attentes scikit-learn.
    """
    #print("Démarrage entraînement TabICL...")
    model = TabICLRegressor()
    model.fit(X_train, y_train.ravel())
    #print("TabICL entraîné avec succès.")
    return model

import catboost as cb

def catboost_train(X_train, y_train, X_val, y_val,
                   iterations=500, depth=6, learning_rate=0.05, objective='Logloss'):
    """
    Entraînement CatBoost robuste.def catboost_train(X_train, y_train, X_val, y_val, iterations=500, depth=6, learning_rate=0.05, objective='RMSE'):
    from catboost import CatBoostRegressor, CatBoostClassifier

    # Si l'objectif est RMSE, on force le Regressor
    if objective == 'RMSE':
        model = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                  loss_function='RMSE', verbose=False)
    else:
        model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                   loss_function=objective, verbose=False)

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    return model

    'objective' : 'Logloss' pour classification (0/1), 'RMSE' pour régression.
    """
    model = cb.CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function=objective,
        verbose=False,
        allow_writing_files=False,
        random_seed=42
    ) if objective == 'Logloss' else cb.CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function=objective,
        verbose=False,
        allow_writing_files=False,
        random_seed=42
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    return model

from tabicl import FinetunedTabICLRegressor

def tabicl_fine_train(X_train, y_train, X_val, y_val, save_dir="./ckpts"):
    """
    1. Entraîne et affine (fine-tune) le modèle TabICL sur les briques historiques.
    2. Sauvegarde les poids et le modèle.
    """
    print("--- Démarrage du Fine-Tuning TabICL Fine ---")

    # Initialisation du régresseur avec fine-tuning activé
    model = FinetunedTabICLRegressor(
        device="cpu",  # Utilisation du GPU si disponible
        epochs=30,  # Nombre d'époques d'apprentissage
        learning_rate=1e-4,  # Taux d'apprentissage adapté
        eval_metric="mse",  # Métrique d'évaluation de la perte
        early_stopping=True,  # Arrêt précoce pour éviter l'overfitting
        patience=5,  # Époques sans amélioration avant l'arrêt
        verbose=True
    )

    # Lancement de l'apprentissage avec validation séquentielle (chronologique)
    model.fit(
        X_train, y_train.ravel(),
        X_val=X_val,
        y_val=y_val.ravel(),
        output_dir=save_dir
    )

    print(f"Fine-tuning terminé. Checkpoints sauvegardés dans : {save_dir}")
    return model


import torch
import torch.nn as nn
import optuna  # Assurez-vous d'importer optuna
torch.set_num_threads(4)

class JepaPredictor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.proj_latent = nn.Linear(hidden_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, h = self.encoder(x)
        z = self.proj_latent(h[-1])
        pred_brute = self.head(z).squeeze(-1)
        return pred_brute, z


# 1. On ajoute trial=None ici
def jepa_train(train_loader, test_loader, jepa_cfg, trial=None):
    def train_model(model, train_loader, test_loader, num_epochs=15, lr=1e-3, trial=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        print("=" * 80)
        print(f"Entraînement JEPA - Prédiction Brute (Sans Activation) sur {device}")
        print(f"Échantillons Train : {len(train_loader.dataset)} | Échantillons Test : {len(test_loader.dataset)}")
        print(f"Époques : {num_epochs} | Taux d'apprentissage : {lr}")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                visible_seq = batch[0].to(device)
                target_brute = batch[1].to(device)
                optimizer.zero_grad()
                pred_brute, _ = model(visible_seq)
                loss = loss_fn(pred_brute, target_brute)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # Évaluation à chaque époque (ou tous les 3)
            if epoch % 3 == 0 or epoch == num_epochs:
                model.eval()
                test_loss = 0.0
                total_test = 0
                with torch.no_grad():
                    for batch in test_loader:
                        visible_seq = batch[0].to(device)
                        target_brute = batch[1].to(device)
                        pred_brute, _ = model(visible_seq)
                        batch_loss = loss_fn(pred_brute, target_brute)
                        test_loss += batch_loss.item() * len(target_brute)
                        total_test += len(target_brute)

                mse_brute = test_loss / total_test

                print(
                    f"Époque [{epoch:02d}/{num_epochs:02d}] "
                    f"| Loss Train: {train_loss:.4f} "
                    f"| Test Loss (MSE): {mse_brute:.4f}"
                )

                # ==========================================================
                # INTÉGRATION DU PRUNER OPTUNA ICI
                # ==========================================================
                if trial is not None:
                    # On rapporte la MSE (ou loss de test) à Optuna
                    trial.report(mse_brute, epoch)

                    # On vérifie si l'essai doit être stoppé net
                    try:
                        if trial.should_prune():
                            print(f"🛑 Essai stoppé net par le Pruner à l'époque {epoch} (MSE: {mse_brute:.4f})")
                            raise optuna.TrialPruned()
                    except optuna.exceptions.TrialPruned:
                        raise
                # ==========================================================

    HIDDEN_DIM = jepa_cfg["HIDDEN_DIM"]
    LATENT_DIM = jepa_cfg["LATENT_DIM"]
    NUM_LAYERS = jepa_cfg["NUM_LAYERS"]
    NUM_EPOCHS = jepa_cfg["NUM_EPOCHS"]
    INPUT_DIM = jepa_cfg["INPUT_DIM"]
    LR = jepa_cfg["LR"]

    model = JepaPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS
    )

    # Inutile sur CPU, mais vous pouvez le laisser ou l'enlever
    if hasattr(torch, 'compile') and torch.cuda.is_available():  model = torch.compile(model)

    # 2. On transmet le trial à train_model
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        trial=trial  # <--- Transmis ici
    )
    return model

# ===================================================== version fin-jepa
import copy

class FinJepaModel(nn.Module):
    def __init__(self, input_dim=5, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        # 1. PriceEncoder (MLP avec GELU et LayerNorm, inspiré de Fin-JEPA)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Le Target Encoder a la même structure, mais ses poids sont mis à jour par EMA
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False  # Pas de rétropropagation directe

        # 2. Prédicteur Causal (Transformer léger pour anticiper le futur latent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=128
        )
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embed_dim = embed_dim

    @torch.no_grad()
    def update_target_encoder(self, m=0.996):
        """Mise à jour EMA (Exponential Moving Average) des poids de la cible"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

    def forward(self, context_x, target_x):
        """
        context_x : Historique passé (ex: [Batch, Seq_Len_Context, Features])
        target_x  : Futur à prédire (ex: [Batch, Seq_Len_Target, Features])
        """
        # Obtention des représentations latentes du passé
        h_context = self.context_encoder(context_x)  # [B, T_ctx, Dim]

        # Obtention des représentations latentes du futur (via Target Encoder sans gradients)
        with torch.no_grad():
            h_target = self.target_encoder(target_x)  # [B, T_tgt, Dim]

        # Le prédicteur essaie de deviner le futur à partir du contexte
        # (On peut ajouter des tokens de masquage ou concaténer selon le besoin)
        prediction = self.predictor(h_context)

        return prediction, h_target


class FinJepaPredictor:
    def __init__(self, input_dim=5, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FinJepaModel(input_dim=input_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.context_encoder.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # Ou une perte combinée MSE + Cosine Similarity

    def train_step(self, context_batch, target_batch):
        self.model.train()
        context_batch = context_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        self.optimizer.zero_grad()

        pred, target_emb = self.model(context_batch, target_batch)

        # Calcul de la perte dans l'espace latent (on aligne les dimensions si nécessaire)
        loss = self.criterion(pred[:, -target_emb.size(1):, :], target_emb)

        loss.backward()
        self.optimizer.step()

        # Mettre à jour l'encodeur cible par EMA
        self.model.update_target_encoder()

        return loss.item()
