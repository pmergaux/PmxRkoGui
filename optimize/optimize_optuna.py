import os
# Désactive les optimisations oneDNN qui causent souvent des erreurs de pointeurs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Désactive les logs excessifs de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import time
import pandas as pd
import optuna
from optuna.storages import fail_stale_trials
from backtest.backtest_module import run_backtest
import tensorflow as tf
import gc
# ====================== CONFIGURATION ======================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RENKO_CACHE_DIR = os.path.abspath(os.path.join(CURRENT_DIR,"../data/renko_cache"))
SYMBOL = "ETHUSD"
TOTAL_MAX_TRIALS = 2096


def objective_wrapper(trial):
    # 1. On vérifie combien de trials sont déjà terminés dans la base
    study = trial.study
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    # 2. Si on a atteint le quota, on arrête cette instance
    if completed_trials >= TOTAL_MAX_TRIALS:
        study.stop()  # Arrête proprement l'instance actuelle
        return 0.0  # Ou une valeur bidon, le stop() va couper court

    # 3. Sinon, on lance votre calcul habituel
    return objective(trial)

def objective(trial):
    # Reset impératif pour éviter l'erreur 'NoneType' / pop
    tf.keras.backend.clear_session()
    """
    Fonction objectif unique pour Optuna.
    Chaque worker exécute cette fonction de manière indépendante.
    """
    # 1. Définition de l'espace de recherche (Paramètres)
    optionSL = [0] + list(range(20, 61, 4))

    config = {
        "parameters": {
            "renko_size": trial.suggest_float('renko_size', 12, 39.0, step=0.1),
            "ema_period": trial.suggest_int('ema_period', 6, 12),
            "rsi_period": trial.suggest_int('rsi_period', 8, 16),
            "rsi_high": 70,
            "rsi_low": 30,
            "macd": {
                "macd_fast": trial.suggest_int('macd_fast', 4, 13),
                "macd_slow": trial.suggest_int('macd_slow', 10, 30),
                "macd_signal": trial.suggest_int('macd_signal', 3, 11)
            },
            "threshold_buy": trial.suggest_float('threshold_buy', 0.55, 0.8, step=0.05),
            "threshold_sell": trial.suggest_float('threshold_sell', 0.2, 0.45, step=0.05),
            "close_buy": trial.suggest_float('close_buy', 0.51, 0.57, step=0.02),
            "close_sell": trial.suggest_float('close_sell', 0.43, 0.49, step=0.02)
        },
        #"features": ["EMA", "RSI", "MACD_hist", "close"],
        "features": ["time_live", "diff_close", "diff_ema", "diff_rsi", "diff_macd"],
        "target": {
            "target_col": trial.suggest_categorical('target_col', ['diff_close', 'diff_ema', 'diff_rsi']),
            "target_type": "direction",
            "target_include": False
        },
        "open_rules": {"rule_ema": True, "rule_rsi": True, "rule_macd": True},
        "close_rules": {"close_sens": True},
        "lstm": {
            "lstm_seq_len": trial.suggest_int('lstm_seq_len', 24, 144, step=24),
            "lstm_units": trial.suggest_int('lstm_units', 48, 240, step=48),
        },
        "mlp": {
            "mlp_unit1": trial.suggest_int('mlp_unit1', 128, 256, step=128),
            "mlp_unit2": 0,
            "mlp_dropout": trial.suggest_float('mlp_dropout', 0.2, 0.5, step=0.1),
            "mlp_lr": 0.001,
            "mlp_batch_size": 256,
            "mlp_patience": trial.suggest_int('mlp_patience', 10, 20)
        },
        "xgb": {
            "xgb_learning_rate": trial.suggest_categorical('xgb_learning_rate', [0.01, 0.03, 0.05]),
            "xgb_max_depth": trial.suggest_categorical('xgb_max_depth', [4, 6, 8]),
            "xgb_n_estimators": 1000,
            "xgb_subsample": trial.suggest_categorical('xgb_subsample', [0.7, 0.8, 0.9]),
            "xgb_colsample_bytree": trial.suggest_categorical('xgb_colsample_bytree', [0.7, 0.8, 0.9]),
            "xgb_early_stop_rounds": 50
        },
        "lgbm": {
            "lgbm_learning_rate": trial.suggest_categorical('lgbm_learning_rate', [0.01, 0.03, 0.05]),
            "lgbm_num_leaves": trial.suggest_categorical('lgbm_num_leaves', [31, 63, 127]),
            "lgbm_n_estimators": 1000,
            "lgbm_feature_fraction": trial.suggest_categorical('lgbm_feature_fraction', [0.7, 0.8, 0.9]),
            "lgbm_bagging_fraction": trial.suggest_categorical('lgbm_bagging_fraction', [0.7, 0.8, 0.9]),
            "lgbm_min_child_samples": trial.suggest_categorical('lgbm_min_child_samples', [20, 50]),
            "lgbm_early_stop_rounds": trial.suggest_categorical('lgbm_early_stop_rounds', [20, 50]),
        },
        "gru":{
            "gru_units1": trial.suggest_int("gru_units1", 32, 128, step=32),    # Puissance de la 1ère couche
            "gru_units2": trial.suggest_int("gru_units2", 16, 64, step=16),    # Puissance de la 2ème couche
            "gru_lr": trial.suggest_float("gru_lr", 0.0001, 0.0096, step=0.0005),      # Vitesse d'apprentissage
            "gru_dropout": trial.suggest_float("gru_dropout", 0.1, 0.4, step=0.1),       # Anti - overfitting
            "batch_size": trial.suggest_categorical("batch_size", [32, 128]),    #Taille du paquet de données
            "gru_patience": 10      #fixez une patience pour l'early stopping
        },
        "live": {
            "symbol": SYMBOL,
            "version": [trial.suggest_categorical('version', ['SIMPLE', 'LSTM', 'GRU', 'ULTRA', 'MLP', 'LGBM', 'XGB'])],
            "sl": trial.suggest_categorical('sl', optionSL),
            "tp": trial.suggest_categorical('tp', [60, 70]),
            "lot_size": 1.0,
            "timeframe": "1m",
            "volume": 1.0,
            "magic": 125788,
            "mt5_login": 203690,
            "mt5_password": "axa77Garp&",
            "mt5_server": "FusionMarkets-Demo",
            "name": "pmxRKO"
        }
    }

    # 2. Chargement des données (Spécifique au Trial actuel)
    try:
        current_renko_size = config['parameters']['renko_size']
        file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{current_renko_size:.1f}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier manquant: {file_path}")

        config['data'] = pd.read_pickle(file_path)

        # 3. Exécution du Backtest (Le pruning se passe à l'intérieur de run_backtest)
        score, result_dict = run_backtest(config, trial=trial)

        tf.keras.backend.clear_session()
        return float(score)

    except optuna.TrialPruned:
        raise  # On laisse remonter l'exception de pruning pour Optuna
    except Exception as e:
        print(f"Erreur Trial {trial.number}: {e}")
        return -999.0
    finally:
        if 'data' in config:
            del config['data']
        gc.collect()


def optimize_start():
    start = time.time()
    if not os.path.exists(RENKO_CACHE_DIR):
        print(f"ERREUR: Dossier cache introuvable.")
        return

    # Nettoyage du fichier de contrôle avant lancement
    control_path = os.path.abspath(os.path.join(CURRENT_DIR,"../best_score.json"))
    # if os.path.exists(control_path):        os.remove(control_path)

    # 1. Création de l'étude avec SQLite et MedianPruner
    # Dans optimize_start()
    # L'URL de connexion à ta nouvelle usine à gaz
    storage_url = "postgresql+pg8000://pierre:axa8Garp@localhost/optuna_db"

    study_name = "trading_diff_postg"
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
    else:
        print("🛑 Objectif déjà atteint. Le script va s'arrêter ou passer au renommage final.")
    print("-" * 50)
    # 2. Lancement de l'optimisation parallélisée native
    n_trials = TOTAL_MAX_TRIALS
    n_jobs = 1

    print(f"Optimisation : {n_trials} essais sur {n_jobs} coeurs.")
    study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs)

    # 3. Post-Optimisation : Renommage des fichiers du champion
    lock_dir = os.path.abspath(os.path.join(CURRENT_DIR,"save_model.lock"))
    while True:
        try:
            # os.mkdir est atomique : si le dossier existe, il lève une erreur direct
            os.mkdir(lock_dir)
            break  # On a le verrou !
        except FileExistsError:
            time.sleep(0.1)  # On attend un peu et on réessaie
    try:
        if os.path.exists(control_path):
            with open(control_path, 'r') as f:
                best_meta = json.load(f)

            hcode = best_meta.get('hcode')
            VERSION = best_meta.get('version', [])
            # 1. Sauvegarde du modèle      _{hcode}
            path = "../models/model_"
            if "XGB" in VERSION:
                path1 = f"{path}hcode.json"
                path2 = f"{path}{hcode}.json"
                os.rename(path1, path2)
            elif "LGBM" in VERSION:
                path1 = f"{path}hcode.txt"
                path2 = f"{path}{hcode}.txt"
                os.rename(path1, path2)
                path1 = f"{path1}.pkl"
                path2 = f"{path2}.pkl"
                os.rename(path1, path2)
            else:
                path1 = f"{path}hcode.keras"
                path2 = f"{path}{hcode}.keras"
                os.rename(path1, path2)

            # 2. Sauvegarde du Scaler  _{hcode}
            scaler_path = "../models/scaler_"
            path1 = f"{scaler_path}hcode.pkl"
            path2 = f'{scaler_path}{hcode}.pkl'
            os.rename(path1, path2)

            # 3. Sauvegarde de la Config   _{hcode}
            config_path = "../config_"
            path1 = f"{config_path}hcode.json"
            path2 = f'{config_path}{hcode}.json'
            os.rename(path1, path2)
            print(f"✅ Champion sauvegardé (HCODE: {hcode})")

            print(f"\n--- OPTIMISATION TERMINÉE ---")
            ncontrol = f"../best_{hcode}_{study_name}.json"
            os.rename(control_path, ncontrol)
            print(f"Meilleur score : {study.best_value:.2f} (HCODE: {hcode} study: {study_name})")

            # Ici vous pouvez ajouter votre logique de renommage final basée sur hcode
    except Exception as e:
        print(f"Erreur lors du traitement final : {e}")
    finally:
        if os.path.exists(lock_dir):
            os.rmdir(lock_dir)
        print(f"durée : {(time.time() - start):.0f}")

if __name__ == "__main__":
    optimize_start()
