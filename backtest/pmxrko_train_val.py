import gc
import os
import sys
import json
import pickle
import time

import joblib
import pandas as pd
import optuna
from optuna.storages import fail_stale_trials

from utils.renko_utils import tick21renko
from utils.model_utils import  config_to_features, prepare_target_column, save_model
# Dans pmxrko_train_val.py
from pipeline_manager import train_all_models
from config import SQLALCHEMY_DATABASE_URL
from tensorflow.keras import backend as K

# Désactivation des logs TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# 1. Désactiver les logs inutiles
tf.get_logger().setLevel('ERROR')

# 2. Configurer la croissance de la mémoire (indispensable pour Optuna)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Ajout du chemin racine pour les imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)
"""
    config_path = os.path.join(ROOT_DIR, "config_live.json")
    if not os.path.exists(config_path):
        print("Erreur: config_live.json introuvable.")
        sys.exit(1)
        
    # pour ne garder que la config en dehors du nom de stratégie
    with open(config_path, 'r') as f:
        config = json.load(f)
        if "live" not in config:
            for k, v in config.items():
                if isinstance(v, dict) and "live" in v:
                    config = v
                    break
"""
RENKO_CACHE_DIR = "/media/pierre/datad/data/renko_cache"
TOTAL_MAX_TRIALS = 256
BATCH_TRIALS = 30  # Nombre maximal de trials exécutés avant de recycler le processus (RAM)

CONFIG = {}

def objective_wrapper(trial):
    # 1. On vérifie combien de trials locaux ont déjà été exécutés dans CE process pour recycler la RAM
    study = trial.study
    local_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    # 2. Si on a fini l'objectif global
    if local_completed_trials >= TOTAL_MAX_TRIALS:
        study.stop()
        import sys
        sys.exit(0)  # Code 0 : Optimisation globale terminée, on s'arrête définitivement

    # 3. Calcul du quota de trials exécutés par CE process en cours
    # On isole les trials démarrés par l'instance actuelle (RUNNING et COMPLETE très récents)
    # Pour faire simple, on utilise une variable globale ou on inspecte les derniers trials pour éviter l'explosion RAM
    trial_counter = getattr(objective_wrapper, 'counter', 0)
    if trial_counter >= BATCH_TRIALS:
        print(f"♻️ Limite locale par lot atteinte ({BATCH_TRIALS} trials). Fermeture propre pour recyclage RAM...")
        study.stop()
        import sys
        sys.exit(10)  # Code 10 spécifique : demande de relance propre au superviseur

    setattr(objective_wrapper, 'counter', trial_counter + 1)

    # 4. Sinon, on lance votre calcul habituel
    return objective(trial)


def objective(trial):
    # Reset au début de chaque essai
    K.clear_session()
    # Reset impératif pour éviter l'erreur 'NoneType' / pop
    tf.keras.backend.clear_session()
    """
    Fonction objectif unique pour Optuna.
    Chaque worker exécute cette fonction de manière indépendante.
    """
    # 1. Définition de l'espace de recherche (Paramètres)
    # 1. Définition de l'espace de recherche (Paramètres)
    # optionSL = [0] + list(range(20, 61, 4))
    optionSL = list(range(30, 121, 5))
    optionTP = list(range(40, 201, 5))  # On autorise des TP plus ambitieux
    close_buy = trial.suggest_float("close_buy", 0.42, 0.48, step=0.01)
    threshold_buy= trial.suggest_float("threshold_buy", 0.6, 0.8, step=0.01)
    close_sell = trial.suggest_float("close_sell", 0.52, 0.58, step=0.01)
    threshold_sell = trial.suggest_float("threshold_sell", 0.2, 0.4, step=0.01)
    renko_size = trial.suggest_float('renko_size', 12, 24.0, step=0.1)
    ema_period = trial.suggest_int('ema_period', 6, 12)
    rsi_period = trial.suggest_int('rsi_period', 8, 16)
    macd_fast = trial.suggest_int('macd_fast', 4, 13)
    macd_slow = trial.suggest_int('macd_slow', 10, 30)
    macd_signal = trial.suggest_int('macd_signal', 3, 11)
    target_col = trial.suggest_categorical('target_col', ['diff_close', 'diff_ema', 'diff_rsi'])

    config["parameters"]["renko_size"] = renko_size
    config["parameters"]["ema_period"] = ema_period
    config["parameters"]["rsi_period"] = rsi_period
    config["parameters"]["macd"]["macd_fast"] = macd_fast
    config["parameters"]["macd"]["macd_slow"] = macd_slow
    config["parameters"]["macd"]["macd_signal"] = macd_signal
    config["parameters"]["close_buy"] = close_buy
    config["parameters"]["threshold_buy"] = threshold_buy
    config["parameters"]["close_sell"] = close_sell
    config["parameters"]["threshold_sell"] = threshold_sell

    config["features"] = ["time_live", "diff_close", "diff_ema", "diff_rsi", "diff_macd"]
    config["target"]["target_col"] = target_col

    config["lstm"]["lstm_seq_len"] = trial.suggest_int('lstm_seq_len', 24, 64, step=8)
    config["lstm"]["lstm_units"] = trial.suggest_int('lstm_units', 48, 240, step=48)

    config["mlp"]["mlp_unit1"] = trial.suggest_int('mlp_unit1', 128, 256, step=128)
    config["mlp"]["mlp_dropout"] = trial.suggest_float('mlp_dropout', 0.2, 0.5, step=0.1)
    config["mlp"]["mlp_patience"] = trial.suggest_int('mlp_patience', 10, 20)

    config["xgb"]["xgb_learning_rate"] = trial.suggest_categorical('xgb_learning_rate', [0.01, 0.03, 0.05])
    config["xgb"]["xgb_max_depth"] = trial.suggest_categorical('xgb_max_depth', [4, 6, 8])
    config["xgb"]["xgb_subsample"] = trial.suggest_categorical('xgb_subsample', [0.7, 0.8, 0.9])
    config["xgb"]["xgb_colsample_bytree"] = trial.suggest_categorical('xgb_colsample_bytree', [0.7, 0.8, 0.9])

    config["lgbm"]["lgbm_learning_rate"] = trial.suggest_categorical('lgbm_learning_rate', [0.01, 0.03, 0.05])
    config["lgbm"]["lgbm_num_leaves"] = trial.suggest_categorical('lgbm_num_leaves', [15, 31, 63])
    config["lgbm"]["lgbm_feature_fraction"] = trial.suggest_categorical('lgbm_feature_fraction', [0.7, 0.8, 0.9])
    config["lgbm"]["lgbm_bagging_fraction"] = trial.suggest_categorical('lgbm_bagging_fraction', [0.7, 0.8, 0.9])
    config["lgbm"]["lgbm_min_child_samples"] = trial.suggest_categorical('lgbm_min_child_samples', [20, 50])
    config["lgbm"]["lgbm_early_stop_rounds"] = trial.suggest_categorical('lgbm_early_stop_rounds', [20, 50])

    config["gru"]["gru_seq_len"] = trial.suggest_int('gru_seq_len', 24, 64, step=8)
    config["gru"]["gru_units1"] = trial.suggest_int("gru_units1", 32, 128, step=32)  # Puissance de la 1ère couche
    config["gru"]["gru_units2"] = trial.suggest_int("gru_units2", 16, 64, step=16)  # Puissance de la 2ème couche
    config["gru"]["gru_lr"] = trial.suggest_float("gru_lr", 0.0001, 0.01, log=True)  # Vitesse d'apprentissage (log)
    config["gru"]["gru_dropout"] = trial.suggest_float("gru_dropout", 0.1, 0.4, step=0.1)  # Anti - overfitting
    config["gru"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 128])  # Taille du paquet de données
    config["gru"]["gru_patience"] = 10  # fixez une patience pour l'early stopping

    config['live']["version"] = [trial.suggest_categorical("version", ["SIMPLE", "LSTM", "GRU", "ULTRA", "LGBM", "XGB"])]
    config['live']["sl"] = trial.suggest_categorical('sl', optionSL)
    config['live']["tp"]= trial.suggest_categorical('tp', optionTP)
    # 2. Chargement des données (Spécifique au Trial actuel)
    try:
        current_renko_size = config['parameters']['renko_size']
        file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{current_renko_size:.1f}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier manquant: {file_path}")

        full_df = pd.read_pickle(file_path)
        # 1. Conversion forcée en Datetime
        # Si c'est des entiers (nanosecondes), on précise l'unité
        if not pd.api.types.is_datetime64_any_dtype(full_df['time']):
            # On teste d'abord les nanosecondes (format classique MT5/Pandas)
            full_df['time'] = pd.to_datetime(full_df['time'], unit='ms', errors='coerce')
        # 1. On récupère la date la plus récente réellement présente dans CE fichier
        # (indispensable car chaque taille de brique finit à un instant différent)
        last_date = full_df['time'].max()

        if pd.isna(last_date):
            print(f"⚠️ Fichier corrompu ou vide pour taille {current_renko_size}")
            return -9999.0

        # 2. On calcule le début (Date de fin du fichier - 120 jours)
        # On ajoute 5 jours de marge pour le "warm-up" des RNN (seq_len)
        date_debut = last_date - pd.Timedelta(days=125)

        # 3. Filtrage
        full_df[full_df['time'] >= date_debut].copy()

        # 4. Vérification de densité
        if len(full_df) < 100:  # Si on a moins de 100 briques sur 100 jours
            print(f"⚠️ Pas assez de briques ({len(full_df)}) pour taille {current_renko_size}")
            return -9998.0

        print(
            f"✅ Trial {trial.number} | Début {date_debut.strftime('%Y-%m-%d')} | Fin: {last_date.strftime('%Y-%m-%d')} | rSize {current_renko_size:.1f} ")

        # # 3. Exécution du Backtest
        result = train_all_models(config, full_df)

        if len(result) == 0:
            return -9999.0
        model_rnn = result['model_rnn']
        if model_rnn is None:
            return -9999.0
        model_tabicl = result['model_tabicl']
        config_std = result['config_std']
        X_scaler = result['scaler']
        VERSION = config_std['live']['version']

        # sauvegarde des modèles
        lock_dir = os.path.join(ROOT_DIR, "save_model.lock")
        while True:
            try:
                # os.mkdir est atomique : si le dossier existe, il lève une erreur direct
                os.mkdir(lock_dir)
                break  # On a le verrou !
            except FileExistsError:
                time.sleep(0.1)  # On attend un peu et on réessaie
        try:
            # --- Zone sécurisée ---
            # 1. Sauvegarde du modèle
            path = os.path.join(ROOT_DIR, "data", f"model_{trial.number}")
            if "XGB" in VERSION:
                path = f"{path}.json"
                model_rnn.save_model(path)
            elif "LGBM" in VERSION:
                # path = f"{path}.txt"
                save_model(model_rnn, path, 'lgbm')
            else:
                path = f"{path}.keras"
                model_rnn.save(path)
            print(f"Model saved to {path}")
            temp_bundle_path = os.path.join(ROOT_DIR, "data", f"tabicl_{trial.number}.joblib")
            bundle = {
                "model": model_tabicl,
                "scaler": X_scaler,
            }
            joblib.dump(bundle, temp_bundle_path)
            # 2. Sauvegarde du Scaler  _{hcode}
            scaler_path = os.path.join(ROOT_DIR, "data", f"scaler_{trial.number}.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(X_scaler, f)
            print(f"Scaler saved to {scaler_path}")
            # 3. Sauvegarde de la Config
            # Multi-strategy wrap
            strategy_name = config_std['live'].get('name', 'unknown_strategy')
            nested_config = {strategy_name: config_std}
            config_path = os.path.join(ROOT_DIR, "data", f"config_{trial.number}.json")
            with open(config_path, 'w') as f:
                json.dump(nested_config, f, indent=4)
            print(f"✅ Modèle trial # {trial.number} sauvegardé par le processus {os.getpid()}")
        finally:
            # On libère toujours le verrou, même si la sauvegarde plante
            if os.path.exists(lock_dir):
                os.rmdir(lock_dir)
        return float(trial.number)
        # ===============
    except optuna.TrialPruned:
        raise  # On laisse remonter l'exception de pruning pour Optuna
    except Exception as e:
        print(f"Erreur Trial {trial.number}: {e}")
        return -999.0
    finally:
        # Reset à la fin pour libérer la RAM/GPU
        K.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()
        # Force glibc à libérer et restituer la mémoire C++ inutilisée au système Linux (efficace à 100% contre les fuites de TF)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

def optimize_start():
    start = time.time()
    if not os.path.exists(RENKO_CACHE_DIR):
        print(f"ERREUR: Dossier cache introuvable.")
        return

    # 1. Création de l'étude avec SQLite et MedianPruner
    # Dans optimize_start()
    # L'URL de connexion à ta nouvelle usine à gaz
    storage_url = SQLALCHEMY_DATABASE_URL

    study_name = "trading_model_postg"
    study = optuna.create_study(
        study_name=study_name,  # Change le nom si tu veux repartir à zéro
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=20,  # Attend 20 tests avant de commencer à élaguer
            n_warmup_steps=5  # Laisse au moins 5 étapes de backtest avant de couper
        )
    )
    # Nettoyage des trials fantômes (crashs précédents)
    fail_stale_trials(study)

    # Calcul du bilan
    total_faits = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    restants = TOTAL_MAX_TRIALS - total_faits

    print("-" * 50)
    print(f"📊 BILAN DE L'ÉTUDE : {study_name}")
    print(f"✅ Tests déjà validés dans Postgres : {total_faits}")
    print(f"🎯 Objectif total : {TOTAL_MAX_TRIALS}")

    if restants > 0:
        print(f"🚀 Cette instance va participer à l'exécution des {restants} tests restants.")
        print("-" * 50)
        # 2. Lancement de l'optimisation parallélisée native
        n_trials = TOTAL_MAX_TRIALS
        n_jobs = 1

        print(f"Optimisation : {n_trials} essais sur {n_jobs} coeurs.")
        study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs)
    else:
        print("🛑 Objectif déjà atteint. Le script va s'arrêter ou passer au renommage final.")
    print(f"durée : {(time.time() - start):.0f}")

if __name__ == "__main__":
    # Script d'auto-test pour valider le chargement et la simulation
    config_path = os.path.join(ROOT_DIR, "config_live.json")
    if not os.path.exists(config_path):
        print(f"Fichier config_live.json manquant. Création d'une config simulée...")
        exit(111)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Gérer le multi-strategy wrap si nécessaire
            if "live" not in config:
                for k, v in config.items():
                    if isinstance(v, dict) and "live" in v:
                        config = v
                        break
    optimize_start()
