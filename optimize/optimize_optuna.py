import gc
import json
import os
import sys
import time

import pandas as pd
import optuna
from joblib import Parallel, delayed

from backtest.backtest_module import run_backtest
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

# ======================================================================
# GESTION DU CACHE ET DES RÉSULTATS
# ======================================================================
# On utilise des variables globales pour le cache et pour stocker les résultats
# C'est une approche simple et efficace pour un script mono-processus.
renko_cache_global = {}
last_renko_size = None
all_trials_results = []  # Pour stocker les résultats détaillés de chaque essai


def evaluate_config_for_optuna(config):
    """
    Fonction qui gère le cache, lance le backtest et stocke les résultats.
    """
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

# 1. On définit une fonction "objectif" pour Optuna
def objective(trial):
    """
    Cette fonction est appelée par Optuna à chaque essai.
    'trial' est un objet spécial qui suggère des paramètres.
    """
    # On définit les plages de recherche pour chaque paramètre
    config = {
        'renko_size': trial.suggest_float('renko_size', 30.0, 38.0),
        'ema_period': trial.suggest_int('ema_period', 8, 15),
        'rsi_period': trial.suggest_int('rsi_period', 10, 20),
        'target_col': trial.suggest_categorical('target_col', ['close', 'EMA']),
        # ... etc. pour tous vos autres paramètres
        'threshold_buy': trial.suggest_float('threshold_buy', 0.55, 0.8),
        'threshold_sell': trial.suggest_float('threshold_sell', 0.2, 0.45),
        # Exemple avec un choix de modèle
        'VERSION': [trial.suggest_categorical('VERSION', ['LGBM', 'XGB'])],
        # Exemple avec un type de cible à optimiser
        #'target_type': trial.suggest_categorical('target_type', ['direction', 'pct_change']),

        # Paramètres fixes
        # 'target_col': 'close',
        'target_type': 'direction',
        'features_base': ["EMA", "RSI", "MACD_hist", "close", "lstm"],
        'params_base': {"renko_size": 17.1, "ema_period": 9, "rsi_period": 14, "rsi_high": 70, "rsi_low": 30,
                        "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}},
        # 'VERSION': ['DECISION'],
        # ...
    }

    # On lance l'évaluation comme avant
    # Note : Cette version simple est mono-processus.
    # Optuna peut être parallélisé, mais commençons simplement.
    score, result = evaluate_config_for_optuna(config)

    # On retourne le score. Optuna cherchera à maximiser cette valeur.
    return score


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
    try:
        current_renko_size = config['renko_size']
        # Pas besoin de vider le cache, car il est recréé à chaque fois
        # que Joblib démarre un nouveau processus pour cette tâche.
        # Logique de chargement depuis le cache (ici, il sera toujours vide)
        # ou depuis le disque.
        file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{current_renko_size:.1f}0.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier Renko manquant: {file_path}")
        df_renko_original = pd.read_pickle(file_path)
        config['data'] = df_renko_original.copy()
        score, result_dict = run_backtest(config)
        # On retourne tout pour l'analyse finale
        return {
            'score': float(score),
            'params': config,
            'details': result_dict
        }
    except Exception as e:
        print(f"ERREUR sur config {config.get('renko_size')}: {e}")
        return {
            'score': -999.0,
            'params': config,
            'details': {'error': str(e)}
        }
    finally:
        gc.collect()

# ======================================================================
# SCRIPT PRINCIPAL
# ======================================================================
def optimize_start():
    if not os.path.exists(RENKO_CACHE_DIR):
        print(f"ERREUR: Le dossier '{RENKO_CACHE_DIR}' n'existe pas.")
        exit()
    start_time = time.time()
    reprise = False
    if not reprise:
        # 1. Création de l'étude Optuna
        study = optuna.create_study(direction='maximize')
        # 2. Définition du nombre d'essais et de workers
        n_trials = 1800
        n_jobs = 9  # Nombre de processus à utiliser en parallèle
        print(f"Démarrage de l'optimisation Optuna pour {n_trials} essais sur {n_jobs} coeurs...")
        # 3. Boucle d'optimisation manuelle pour la parallélisation
        total_results = []
        for i in range(n_trials // n_jobs):
            # On génère un "batch" de configurations suggérées par Optuna
            trials = [study.ask() for _ in range(n_jobs)]
            configs = []
            for trial in trials:
                config = {
                    'renko_size': trial.suggest_float('renko_size', 20.0, 38.0, step=0.1),
                    'ema_period': trial.suggest_int('ema_period', 8, 15),
                    'rsi_period': trial.suggest_int('rsi_period', 10, 20),
                    'target_col': trial.suggest_categorical('target_col', ['close', 'EMA']),
                    'threshold_buy': trial.suggest_float('threshold_buy', 0.55, 0.8, step=0.05),
                    'threshold_sell': trial.suggest_float('threshold_sell', 0.2, 0.45, step=0.05),
                    'lstm_seq_len': trial.suggest_int('lstm_seq_len', 24, 144, step=24),
                    'lstm_units': trial.suggest_int('lstm_units', 48, 256, step=48),
                    'lgbm_learning_rate': trial.suggest_categorical('lgbm_learning_rate',[0.01, 0.03, 0.05]),
                    'lgbm_num_leaves': trial.suggest_categorical('lgbm_num_leaves',[31, 63, 127]),
                    'lgbm_feature_fraction': trial.suggest_categorical('lgbm_feature_fraction',[0.7, 0.8, 0.9]),
                    'lgbm_bagging_fraction': trial.suggest_categorical('lgbm_bagging_fraction',[0.7, 0.8, 0.9]),
                    'lgbm_min_child_samples': trial.suggest_categorical('lgbm_min_child_samples',[20, 50]),
                    'lgbm_early_stop_rounds': trial.suggest_categorical('lgbm_early_stop_rounds',[20, 50]),
                    'mlp_unit1': trial.suggest_int('mlp_unit1', 128, 256, step=128),
                    'mlp_unit2': 0,
                    'mlp_dropout': trial.suggest_float('mlp_dropout', 0.2, 0.5, step=0.1),
                    'xgb_learning_rate': trial.suggest_categorical('xgb_learning_rate',[0.01, 0.03, 0.05]),
                    'xgb_max_depth': trial.suggest_categorical('xgb_max_depth',[4, 6, 8]),
                    'xgb_subsample': trial.suggest_categorical('xgb_subsample',[0.7, 0.8, 0.9]),
                    'xgb_colsample_bytree': trial.suggest_categorical('xgb_colsample_bytree',[0.7, 0.8, 0.9]),
                    'xgb_early_stop_rounds': 50,
                    'features': ['EMA', 'RSI', 'MACD_hist', 'close'],
                    'VERSION': [trial.suggest_categorical('VERSION', ['SIMPLE', 'ULTRA', 'MLP', 'LGBM', 'XGB'])],
                    'target_type': 'direction',
                    'symbol':'ETHUSD',
                }
                configs.append(config)

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

        # Créer la configuration complète avec les meilleurs paramètres
        final_config = {
            'renko_size': best_params['renko_size'],
            'ema_period': best_params['ema_period'],
            'rsi_period': best_params['rsi_period'],
            'target_col': best_params['target_col'],
            'threshold_buy': best_params['threshold_buy'],
            'threshold_sell': best_params['threshold_sell'],
            'lstm_seq_len': best_params['lstm_seq_len'],
            'lstm_units': best_params['lstm_units'],
            'lgbm_learning_rate': best_params['lgbm_learning_rate'],
            'lgbm_num_leaves': best_params['lgbm_num_leaves'],
            'lgbm_feature_fraction': best_params['lgbm_feature_fraction'],
            'lgbm_bagging_fraction': best_params['lgbm_bagging_fraction'],
            'lgbm_min_child_samples': best_params['lgbm_min_child_samples'],
            'lgbm_early_stop_rounds': best_params['lgbm_early_stop_rounds'],
            'mlp_unit1': best_params['mlp_unit1'],
            'mlp_unit2': 0,
            'mlp_dropout': best_params['mlp_dropout'],
            'xgb_learning_rate': best_params['xgb_learning_rate'],
            'xgb_max_depth': best_params['xgb_max_depth'],
            'xgb_subsample': best_params['xgb_subsample'],
            'xgb_colsample_bytree': best_params['xgb_colsample_bytree'],
            'xgb_early_stop_rounds': 50,
            'VERSION': [best_params['VERSION']],
            'target_type': 'direction',
            'features': ["EMA", "RSI", "MACD_hist", "close"],
        }
    else:
        final_config = {
            'renko_size': 35.1,
            'ema_period': 13,
            'rsi_period': 15,
            'target_col': 'close',
            'threshold_buy': 0.6,
            'threshold_sell': 0.35,
            'seq_len': 53,
            'units': 106,
            'VERSION': ['ULTRA'],
            'target_type': 'direction',
            'features': ["EMA", "RSI", "MACD_hist", "close"],
        }
    # Charger les données renko correspondantes
    renko_size = final_config['renko_size']
    file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{renko_size:.1f}0.pkl")
    if os.path.exists(file_path):
        df_renko = pd.read_pickle(file_path)
        # Appeler run_backtest avec save_artifacts=True
        final_config['data'] = df_renko.copy() # Ajouter les données à la config pour run_backtest
        print("\nSauvegarde du modèle et des scalers optimaux...")
        score, result = run_backtest(final_config, save_artifacts=True)
        print("Sauvegarde terminée.")
    else:
        print(f"Fichier Renko manquant pour la meilleure configuration: {file_path}")
        sys.exit(2)
    confid_std = to_config_std(final_config)
    confid_std['live']['hcode'] = result['hcode']
    config_path = '../config_live.json'
    with open(config_path, 'w') as f:
        json.dump(confid_std, f, indent=2)
    print(f"+ sauvegarde en {(time.time() - start_time):.0f}")

if __name__ == "__main__":
    optimize_start()
