import gc
import json
import os
import sys
import time

import pandas as pd
import optuna
from joblib import Parallel, delayed

from backtest.backtest_module import run_backtest
from decision.candle_decision import calculate_japonais
from utils.config_utils import to_config_std
from utils.utils import clean_numpy_types

## **3. DASHBOARD — `optuna-dashboard`**
"""
---bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db

→ Ouvre `http://localhost:8080` → **visualisation en temps réel**
"""
# =========================================================================
RENKO_CACHE_DIR = "../data/renko_cache"
#SYMBOL = 'NAS100'
SYMBOL = "ETHUSD"
# ======================================================================
# GESTION DU CACHE ET DES RÉSULTATS
# ======================================================================
# On utilise des variables globales pour le cache et pour stocker les résultats
# C'est une approche simple et efficace pour un script mono-processus.
renko_cache_global = {}
last_renko_size = None
# all_trials_results = []  # Pour stocker les résultats détaillés de chaque essai
df = None
"""
def evaluate_config_for_optuna(config):
    # Fonction qui gère le cache, lance le backtest et stocke les résultats.
    global renko_cache_global, last_renko_size, all_trials_results

    try:
        current_renko_size = config['renko_size']
        # Logique de vidage du cache si la taille change
        if current_renko_size != last_renko_size:
            print(f"Changement de taille de {last_renko_size} à {current_renko_size}. Vidage du cache.")
            renko_cache_global.clear()
            last_renko_size = current_renko_size
        # Logique de chargement depuis le cache
        if current_renko_size in renko_cache_global:
            df_renko_original = renko_cache_global[current_renko_size]
        else:
            print(f"Cache MISS pour renko_size={current_renko_size}. Chargement...")
            file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{current_renko_size:.2f}.pkl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier Renko manquant: {file_path}")

            df_renko_original = pd.read_pickle(file_path)
            renko_cache_global[current_renko_size] = df_renko_original
        # On travaille sur une copie
        config['data'] = df_renko_original.copy()
        # On exécute le backtest qui retourne (score, result_dict)
        score, result_dict = run_backtest(config)
        # On stocke les résultats détaillés pour analyse ultérieure
        full_result = {
            'score': score,
            'params': config,
            'details': result_dict
        }
        all_trials_results.append(full_result)
        # On supprime la donnée pour ne pas la garder en mémoire inutilement
        return float(score)

    except Exception as e:
        print(f"ERREUR sur config {config.get('renko_size')}: {e}")
        return -999.0
    finally:
        gc.collect()
"""
# ======================================================================
# FONCTION D'ÉVALUATION POUR UN SEUL ESSAI (EXÉCUTÉE PAR CHAQUE WORKER)
# ======================================================================
def evaluate_trial(config):
    """
    Évalue une seule configuration. C'est cette fonction qui sera parallélisée.
    Elle a son propre cache interne (qui sera détruit avec le processus).
    """
    # Le cache est local à cette exécution
    #renko_cache_local = {}
    global df
    try:
        current_renko_size = config['parameters']['renko_size']
    except Exception as e:
        print(f"ERREUR sur config: {e}")
        return {
            'score': -999.0,
            'params': {'error': str(e), 'confid': config}
        }
    try:
        # Pas besoin de vider le cache, car il est recréé à chaque fois
        # que Joblib démarre un nouveau processus pour cette tâche.
        # Logique de chargement depuis le cache (ici, il sera toujours vide)
        # ou depuis le disque.
        file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{current_renko_size:.1f}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier Renko manquant: {file_path}")
        df_renko_original = pd.read_pickle(file_path)
        config['data'] = df_renko_original.copy()
        """
        try:
            config['df'] = df.copy()
        except BaseException as e:
            print("japon err", e)
        """
        score, result_dict = run_backtest(config)
        # On retourne tout pour l'analyse finale
        return {
            'score': float(score),
            'params': result_dict
        }
    except Exception as e:
        print(f"ERREUR renko: {e}")
        return {
            'score': -999.0,
            'params': {'error': str(e), 'confid': config}
        }
    finally:
        gc.collect()

# ======================================================================
# SCRIPT PRINCIPAL
# ======================================================================
def optimize_start():
    global df
    if not os.path.exists(RENKO_CACHE_DIR):
        print(f"ERREUR: Le dossier '{RENKO_CACHE_DIR}' n'existe pas.")
        exit()
    start_time = time.time()
    reprise = False
    vers = 0
    if not reprise:
        # 1. Création de l'étude Optuna
        study = optuna.create_study(direction='maximize')
        # 2. Définition du nombre d'essais et de workers
        n_trials = 1048
        n_jobs = 4  # Nombre de processus à utiliser en parallèle
        print(f"Démarrage de l'optimisation Optuna pour {n_trials} essais sur {n_jobs} coeurs...")
        # 3. Boucle d'optimisation manuelle pour la parallélisation
        total_results = []
        for i in range(n_trials // n_jobs):
            # On génère un "batch" de configurations suggérées par Optuna
            trials = [study.ask() for _ in range(n_jobs)]
            configs = []
            for trial in trials:
                if vers == 1:
                    config = {
                    "parameters":
                    {
                        "renko_size": trial.suggest_float("renko_size", 8.5, 9.5, step=0.1),
                        "ema_period": 9,
                        "rsi_period": 14,
                        "rsi_high": 70,
                        "rsi_low": 30,
                        "macd":
                        {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
                        "threshold_buy": 0.65,
                        "threshold_sell": 0.4,
                        "close_buy": 0.53,
                        "close_sell": 0.45
                    },
                    "features": ["EMA", "RSI", "MACD_hist", "close"],
                    "target": {
                        "target_col": trial.suggest_categorical('target_col', ['close', 'EMA']),
                        "target_type": "direction",
                        "target_include": False
                    },
                    "open_rules":
                    {"rule_ema": True,
                    "rule_rsi": True,
                    "rule_macd": True},
                    "close_rules":
                    {"close_sens": True},
                    "lstm":{
                        "lstm_seq_len": trial.suggest_int('lstm_seq_len', 24, 144, step=24),
                        "lstm_units": trial.suggest_int('lstm_units', 48, 256, step=48)},
                    "mlp":
                    {"mlp_unit1": 128, "mlp_unit2": 0, "mlp_dropout": 0.30000000000000004, "mlp_lr": 0.001,
                    "mlp_batch_size": 256, "mlp_patience": 10},
                    "live":
                    {"symbol": "ETHUSD",
                     "version": [trial.suggest_categorical('version', ['SIMPLE', 'ULTRA', 'MLP', 'LGBM', 'XGB'])],
                     "hcode": "9983c07c8ba0",
                    "timeframe": "1m", "volume": 1.0,
                     "sl": trial.suggest_categorical("sl", [0, 60, 70]),
                     "tp": trial.suggest_categorical("tp", [0, 60, 70]),
                     "magic": 125788, "mt5_login": 203690,
                    "mt5_password": "axa77Garp&", "mt5_server": "FusionMarkets-Demo",
                    "name": "pmxRKO", "lot_size": 1.0},
                    "xgb":
                    {"xgb_learning_rate": 0.03, "xgb_max_depth": 6, "xgb_n_estimators": 1000, "xgb_subsample": 0.9,
                    "xgb_colsample_bytree": 0.8, "xgb_early_stop_rounds": 50},
                    "lgbm": {"lgbm_learning_rate": 0.01, "lgbm_num_leaves": 31, "lgbm_n_estimators": 1000,
                    "lgbm_feature_fraction": 0.9, "lgbm_bagging_fraction": 0.8, "lgbm_min_child_samples": 50,
                    "lgbm_early_stop_rounds": 50}
                    }
                    configs.append(config)
                if vers == 0:
                    config = {
                        "parameters": {
                            "renko_size": trial.suggest_float('renko_size', 6, 39.0, step=0.1),
                            "ema_period": trial.suggest_int('ema_period', 8, 10),
                            "rsi_period": trial.suggest_int('rsi_period', 12, 16),
                            "rsi_high": 70,
                            "rsi_low": 30,
                            "macd": {
                              "macd_fast": 12,
                              "macd_slow": 26,
                              "macd_signal": 9
                            },
                            "threshold_buy": trial.suggest_float('threshold_buy', 0.55, 0.8, step=0.05),
                            "threshold_sell": trial.suggest_float('threshold_sell', 0.2, 0.45, step=0.05),
                            "close_buy": trial.suggest_float('close_buy', 0.51, 0.57, step=0.02),
                            "close_sell": trial.suggest_float('close_sell', 0.43, 0.49, step=0.02)
                            },
                        "features": ["EMA","RSI","MACD_hist", "close"],
                        "target": {
                            "target_col": trial.suggest_categorical('target_col', ['close', 'EMA']),
                            "target_type": "direction",
                            "target_include": False
                            },
                        "open_rules": {
                            "rule_ema": True,
                            "rule_rsi": True,
                            "rule_macd": True
                            },
                        "close_rules": {
                            "close_sens": True
                            },
                        "lstm": {
                            "lstm_seq_len": trial.suggest_int('lstm_seq_len', 24, 144, step=24),
                            "lstm_units": trial.suggest_int('lstm_units', 48, 256, step=48),
                            },
                        "mlp": {
                            "mlp_unit1": trial.suggest_int('mlp_unit1', 128, 256, step=128),
                            "mlp_unit2": 0,
                            "mlp_dropout": trial.suggest_float('mlp_dropout', 0.2, 0.5, step=0.1),
                            "mlp_lr": 0.001,
                            "mlp_batch_size": 256,
                            "mlp_patience": 10
                            },
                        "xgb": {
                            "xgb_learning_rate": trial.suggest_categorical('xgb_learning_rate',[0.01, 0.03, 0.05]),
                            "xgb_max_depth": trial.suggest_categorical('xgb_max_depth',[4, 6, 8]),
                            "xgb_n_estimators": 1000,
                            "xgb_subsample": trial.suggest_categorical('xgb_subsample',[0.7, 0.8, 0.9]),
                            "xgb_colsample_bytree": trial.suggest_categorical('xgb_colsample_bytree',[0.7, 0.8, 0.9]),
                            "xgb_early_stop_rounds": 50
                            },
                        "lgbm": {
                            "lgbm_learning_rate": trial.suggest_categorical('lgbm_learning_rate',[0.01, 0.03, 0.05]),
                            "lgbm_num_leaves": trial.suggest_categorical('lgbm_num_leaves',[31, 63, 127]),
                            "lgbm_n_estimators": 1000,
                            "lgbm_feature_fraction": trial.suggest_categorical('lgbm_feature_fraction',[0.7, 0.8, 0.9]),
                            "lgbm_bagging_fraction": trial.suggest_categorical('lgbm_bagging_fraction',[0.7, 0.8, 0.9]),
                            "lgbm_min_child_samples": trial.suggest_categorical('lgbm_min_child_samples',[20, 50]),
                            "lgbm_early_stop_rounds": trial.suggest_categorical('lgbm_early_stop_rounds',[20, 50]),
                            },
                        "live": {
                            "symbol": SYMBOL,
                            "version": [
                                trial.suggest_categorical('version', ['SIMPLE', 'ULTRA', 'MLP', 'LGBM', 'XGB'])],
                            "hcode": "32ad03392bd7",
                            "timeframe": "1m",
                            "volume": 1.0,
                            "sl": trial.suggest_categorical('sl', [0, 60, 70]),
                            "tp": trial.suggest_categorical('tp', [0, 60, 70]),
                            "magic": 125788,
                            "mt5_login": 203690,
                            "mt5_password": "axa77Garp&",
                            "mt5_server": "FusionMarkets-Demo",
                            "name": "pmxRKO",
                            "lot_size": 1.0
                        },
                    }
                    configs.append(config)
            """
            filename =  "../data/df_ETHUSD.pkl"
            df = pd.read_pickle(filename)
            """
            # On exécute le batch de tâches en parallèle
            print(f"\nLancement du batch {i + 1}/{n_trials // n_jobs}...")
            results = Parallel(n_jobs=n_jobs)(delayed(evaluate_trial)(cfg) for cfg in configs)

            # On informe Optuna des résultats obtenus pour ce batch
            for trial, result in zip(trials, results):
                # 'tell' prend l'essai et le score, et met à jour l'étude
                study.tell(trial, result['score'])

            # On peut sauvegarder les résultats détaillés au fur et à mesure
            with open("all_trials_results.json", "a") as f:
                for res in results:
                    # Nettoyage des types NumPy avant sauvegarde
                    f.write(json.dumps(clean_numpy_types(res)) + "\n")
                    total_results.append([res['score'], res])

        # --- Affichage et sauvegarde finale ---
        print("\nOptimisation terminée !")
        print(f"Meilleur score : {study.best_value}")
        print(f"Meilleurs paramètres : {study.best_params}")

        best_params = study.best_params
        with open("best_params_optuna.json", "w") as f:
            json.dump(best_params, f, indent=2)
        tops = sorted(total_results, key=lambda x: x[0], reverse=True)
        for i in range(20):
            print(tops[i])
        print(f"optimisation en {(time.time() - start_time):.0f}")
        resultat = tops[0]
        # Créer la configuration complète avec les meilleurs paramètres
        final_config = resultat[1]['params']['config']
    else:
        final_config = {
                                  'renko_size': 33.7, 'ema_period': 8, 'rsi_period': 16, 'target_col': 'EMA',
                                             'threshold_buy': 0.65, 'threshold_sell': 0.25, 'close_buy': 0.57,
                                             'close_sell': 0.47, 'lstm_seq_len': 96, 'lstm_units': 48,
                                             'lgbm_learning_rate': 0.03, 'lgbm_num_leaves': 31,
                                             'lgbm_feature_fraction': 0.8, 'lgbm_bagging_fraction': 0.8,
                                             'lgbm_min_child_samples': 50, 'lgbm_early_stop_rounds': 20,
                                             'mlp_unit1': 128, 'mlp_unit2': 0, 'mlp_dropout': 0.30000000000000004,
                                             'xgb_learning_rate': 0.05, 'xgb_max_depth': 6, 'xgb_subsample': 0.9,
                                             'xgb_colsample_bytree': 0.9, 'xgb_early_stop_rounds': 50,
                                             'features': ['EMA', 'RSI', 'MACD_hist', 'close'], 'VERSION': ['LGBM'],
                                             'target_type': 'direction', 'target_include': False, 'symbol': 'ETHUSD',
                                             'sl': 40, 'tp': 0, 'hcode': '32ad03392bd7'}
    # Charger les données renko correspondantes
    renko_size = final_config['parameters']['renko_size']
    file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{renko_size:.1f}.pkl")
    if os.path.exists(file_path):
        df_renko = pd.read_pickle(file_path)
        # Appeler run_backtest avec save_artifacts=True
        final_config['data'] = df_renko.copy() # Ajouter les données à la config pour run_backtest
        print("\nSauvegarde du modèle et des scalers optimaux...")
        score, result = run_backtest(final_config, save_artifacts=True)
        print("Sauvegarde terminée.", score, result)
    else:
        print(f"Fichier Renko manquant pour la meilleure configuration: {file_path}")
        sys.exit(2)
    hcode = final_config['live']['hcode']
    config_path = f"../config_live_{hcode}.json"
    with open(config_path, 'w') as f:
        json.dump(result['config'], f, indent=2)
    print(f"+ sauvegarde en {(time.time() - start_time):.0f}")

if __name__ == "__main__":
    optimize_start()
