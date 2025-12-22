# Auteur : Pierre — Version finale — Prêt à lancer
# Objectif : 100 essais → trouve le Graal ou meurt
import os
import pickle

import optuna

from train.trainer import (lstm_train_simple, lstm_train_ultra, lstm_predict_ultra,
                           lstm_train_model, lstm_predict_model, tft_train_predict_fast,
                           tft_train_predict, tft_to_proba, nbeats_train_predict,
                           mlp_train, mlp_predict,
                           lgbm_train, lgbm_predict, xgb_train, xgb_predict)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # ← à mettre TOUT EN HAUT du fichier
# 2. Optionnel mais recommandé : limite la verbosité de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warning, 3=error only
# Optionnel : désactive les protections Lightning qui tuent les processus
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import glob
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import json
from datetime import datetime
import time
from joblib import dump

# ====================== TES FONCTIONS EXISTANTES (à garder) ======================
from utils.config_utils import config_to_hash, prepare_to_hashcode, params_base, features_base, to_config_std
from utils.renko_utils import tick21renko
from utils.utils import reload_ticks_from_pickle
from utils.model_utils import (scale_cols_only, assemble_with_targets,
                               create_sequences_numba,
                               config_to_features, nn_servers)
from decision.candle_decision import add_indicators, choix_features
from decision.trading_decision import trading_decision  # ← ton add_indicators Numba parfait


# =========================================================================
# --- utilities
# =========================================================================
# ====================== TARGETS SIMPLES ======================
def prepare_target_column(df, target_col, target_type, horizon=1):
    """
    Prépare la colonne cible en fonction du type demandé.
    Retourne le DataFrame avec une nouvelle colonne 'target'.
    """
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

def decision(df, config):
    print("decision")
    features = config.get("features", None)
    if features is None:
        raise "inutile de poursuivre features inconnues"
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
# 4. BACKTEST RÉALISTE
# ==================================================================
def backtest(df_test, proba, sl=60, tp=30, buy_thr=0.6, sell_thr=0.4):
    from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE
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
    pnl = []
    pos = 0
    entry_price = 0
    spread_dollar = 2.5  # spread en $
    for i in range(3, len(df)):
        close = df['close'].iloc[i]
        if close == 0:
            continue
        if proba is not None:
            signal = trading_decision(pos, entry_price, close, df[i-3:i], None,
                                    proba[i-3:i], sl, tp, buy_thr, sell_thr)
        else:
            signal = trading_decision(pos, entry_price, close, df[i-3:i], None,
                                      None, sl, tp, buy_thr, sell_thr)
        # === GESTION POSITION ===
        if pos == 0 and signal != NONE:
            pos = signal
            entry_price = close
            continue
        elif pos != 0 and signal >= CLOSE:
            # sortie forcée si signal neutre ou de sens opposé
            pnl.append(pos * (close - entry_price) - spread_dollar)
            pos = 0
            continue
    # === Si on sort à la fin ===
    if pos != 0:
        final_close = df['close'].iloc[-1]
        final_pnl = pos * (final_close - entry_price) - spread_dollar
        pnl.append(final_pnl)
    a = np.array(pnl)
    if len(a) == 0:
        # print("aucun trade")
        return -999999, {}
    print(f"profit {np.sum(a):.2f} trades {len(a)} winner {np.sum(a > 0)} win rate {(np.sum(a > 0)/len(a)):.2%} moyenne {np.mean(a):.2f} écart type {np.std(a, ddof=1):.4f}")
    result = {'profit': np.sum(a), 'trades': len(a), 'winner': np.sum(a > 0), 'win_rate': (np.sum(a > 0) / len(a)), 'mean': np.mean(a), 'std': np.std(a, ddof=1)}
    sharpe = a.mean() / a.std() * np.sqrt(365 * 390)   # calcul pour 390 renko/j. estimés
    return sharpe * 1000, result

# ==================================================================
# 5. ÉVALUATION FINALE — LA VÉRITÉ
# ==================================================================
def run_backtest(config, save_artifacts=False):
    score = float('-inf')
    try:
        # ← On charge les données ici, une fois par worker (ou même une fois globalement)
        """
        start_date = config['start_date']
        end_date = config['end_date']
        symbol = config['symbol']
        start = datetime(*start_date)
        end = datetime(*end_date)
        base_name = f"../data/{symbol}_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
        df = reload_ticks_from_pickle(base_name, symbol, None, start, end)
        if df is None or df.empty:
            print("Pas de données → exit")
            exit()
        df['time'] = pd.to_datetime(df['time'])
        """
        if config is None:
            raise Exception("no config no optim")
        try:
            df_renko = config['data']
        except BaseException as e:
            print("No data no optim", e)
            return score, {}
        del config['data']
        # 0. standardiser la config par exemple si on doit utiliser le hcode
        VERSION = config["VERSION"]
        config_std = to_config_std(config)
        # les colonnes data, cible...
        features_cols, target_cols, total_cols = config_to_features(config_std)
        config_std["features"] = features_cols
        renko_size = config['renko_size']
        # 1. Renko + indicateurs
        """
        try:
            df_bricks = tick21renko(df, None, renko_size, 'bid')
        except BaseException as e:
            print("err create renko", e)
            return score, {}
        """
        try:
            df_renko = add_indicators(df_renko, config_std["parameters"])
            if df_renko is None:
                return score, {}
        except BaseException as e:
            print("err add indicators ", e)
            return score, {}
        # 2. Features
        need_nn = any(vs in nn_servers for vs in VERSION)
        proba = None
        # inutile si decision seule
        seq_len = config.get('seq_len', 24)
        units = config.get('lstm_units', 48)
        thresh_buy = config.get('threshold_buy', 0.6)
        thresh_sell = config.get('threshold_sell', 0.4)
        hcode = "default_hcode"
        if need_nn:
            target_type = config.get('target_type', 'direction')
            df_renko = prepare_target_column(df_renko, target_cols[0], target_type)
            if 'target' in df_renko.columns:
                target_cols = ['target']
            else:
                if "target_sign_mean" in target_cols:
                    df_renko = prepare_targets_simple(df_renko, horizon=5)
                    tarc = []
                    for col in target_cols:
                        if col == "target_sign_mean":
                            tarc.append('target')
                            continue
                        tarc.append(col)
                    target_cols = tarc
            if len(df_renko) < seq_len*4:
                print("pas assez de renko", len(df_renko))
                return score, {}
            config_std['target']['target_col'] = [target_cols[0]]     # on se limite à une seule cible
            try:
                hcode = config_to_hash(prepare_to_hashcode(config_std))
                # print("hcode", hcode)
                config["hcode"] = hcode
            except BaseException as e:
                print("err hcode", e)
            # 3. Split
            train_len = int(len(df_renko) * 0.65)
            val_len = int(len(df_renko) * 0.15)
            train_df = df_renko.iloc[:train_len]
            val_df = df_renko.iloc[train_len:train_len + val_len]
            test_df = df_renko.iloc[train_len + val_len:]

            # 4. Scale
            X_scaler, X_train, X_val, X_test = scale_cols_only(train_df, val_df, test_df, features_cols)
            if save_artifacts:
                scaler_path = f"../models/scaler_{hcode}.pkl"
                with (open(scaler_path, 'wb')) as f:
                    pickle.dump(X_scaler, f)
                print(f"Scaler saved to {scaler_path}")

            if target_type == 'value':
                y_scaler, y_train, y_val, y_test = scale_cols_only(train_df, val_df, test_df, target_cols)
                if save_artifacts and y_scaler:
                    scaler_y_path = f"../models/scaler_y_{hcode}.pkl"
                    with (open(scaler_y_path, 'wb')) as f:
                        pickle.dump(y_scaler, f)
                    print(f"Y-Scaler saved to {scaler_y_path}")
            else:
                y_train = train_df[target_cols].to_numpy(dtype=np.float32)
                y_val = val_df[target_cols].to_numpy(dtype=np.float32)
                y_test = test_df[target_cols].to_numpy(dtype=np.float32)

            train_r, val_r, test_r = assemble_with_targets(X_train, X_val, X_test, y_train, y_val, y_test)
            X_train_seq, y_train_seq = create_sequences_numba(train_r, seq_len, len(features_cols))
            X_val_seq, y_val_seq = create_sequences_numba(val_r, seq_len, len(features_cols))
            X_test_seq, _ = create_sequences_numba(test_r, seq_len, len(features_cols))

            if len(X_train_seq) < 50:
                return float('-inf'), {}

            test_return = test_df.iloc[-len(X_test):]

            model = None
            if test_return is not None:
                test_return = decision(test_return, config_std)

            for vs in VERSION:
                # --- Model Training ---
                if 'SIMPLE'==vs:
                    model = lstm_train_simple(X_train_seq, y_train_seq, X_val_seq, y_val_seq, seq_len, len(features_cols), units)
                elif 'ULTRA'==vs:
                    model = lstm_train_ultra(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
                elif 'LSTM'==vs:
                    model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
                elif 'MLP' == vs:
                    mlp = config_std['mlp']
                    model = mlp_train(X_train, y_train, X_val, y_val, len(features_cols),
                                      units1=mlp['mlp_unit1'], units2=mlp['mlp_unit2'],
                                      dropout=mlp['mlp_dropout'], lr=mlp['mlp_lr'],
                                      batch_size=mlp['mlp_batch_size'], patience=mlp['mlp_patience'])
                elif 'LGBM' == vs:
                    try:
                        lgbm = config_std['lgbm']
                    except BaseException as e:
                        print("No LGBM")
                        break
                    model = lgbm_train(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32),
                                       np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.float32),
                                       learning_rate=lgbm['lgbm_learning_rate'], n_estimators=lgbm['lgbm_n_estimators'],
                                       num_leaves=lgbm['lgbm_num_leaves'],feature_fraction=lgbm['lgbm_feature_fraction'],
                                       min_child_samples=lgbm['lgbm_min_child_samples'],early_stop_rounds=lgbm['lgbm_early_stop_rounds'],
                                       bagging_fraction=lgbm['lgbm_bagging_fraction'])
                elif 'XGB' == vs:
                    xgb = config_std['xgb']
                    model = xgb_train(X_train, y_train, X_val, y_val, learning_rate=xgb['xgb_learning_rate'],
                                      max_depth=xgb['xgb_max_depth'],n_estimators=xgb['xgb_n_estimators'],
                                      subsample=xgb['xgb_subsample'], colsample_bytree=xgb['xgb_colsample_bytree'],
                                      early_stop_rounds=xgb['xgb_early_stop_rounds'])

                # --- Prediction & Saving ---
                if model:
                    if save_artifacts:
                        path = f"../models/model_{hcode}"
                        directory = os.path.dirname(path)
                        if directory:
                            os.makedirs(directory, exist_ok=True)
                        if "XGB" in VERSION:
                            path = f"{path}.json"
                            model.save_model(path)
                        elif "LGBM" in VERSION:
                            path = f"{path}.txt"
                            model.save_model(path)
                        else:
                            path = f"{path}.keras"
                            model.save(path)
                        print(f"Model saved to {path}")
                        # If we are just saving, we don't need to predict and backtest
                        return 0, {'hcode': hcode}

                    # Prediction logic
                    try:
                        if 'LGBM' in VERSION:
                            proba = lgbm_predict(model, X_test)
                        elif 'XGB' in VERSION:
                            proba = xgb_predict(model, X_test)
                        elif 'LSTM' in VERSION:
                            proba = lstm_predict_model(model, X_test_seq)
                        elif 'MLP' in VERSION:
                            proba = mlp_predict(model, X_test)
                        else:
                            proba = model.predict(X_test_seq, verbose=0).flatten()
                    except BaseException as e:
                        print(f"err prediction {VERSION} ", e)
                        proba = None
                    del model
                    break # Exit after first valid model is trained and used
        else:
            if not save_artifacts:
                test_df = df_renko.iloc[-min(240, int(len(df_renko)*0.2)):]
                test_return = decision(test_df, config_std)
                proba = None

        if save_artifacts:
            print("Artifacts saved. Skipping backtest.")
            return 0, {'hcode': hcode}

        # 7. Backtest réaliste
        try:
            score, result = backtest(test_return, proba, config_std['live']['sl'], config_std['live']['tp'],thresh_buy, thresh_sell)
        except BaseException as e:
            print("BT err ", e)
            result = {}
        print(f"{VERSION} = {target_cols} | {renko_size:5.1f} | {config['ema_period']} | {config['rsi_period']} | {seq_len:3d} | {units:3d} | {thresh_buy:.3f}/{thresh_sell:.3f} → Score {score:8.1f}")
        return score, result

    except Exception as e:
        print("ERREUR →", e)
        return score, {}
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
