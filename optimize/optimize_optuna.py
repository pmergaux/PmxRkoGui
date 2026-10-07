from mt5linux import MetaTrader5
from datetime import datetime, timezone, timedelta
import multiprocessing as mp

from sympy.core import parameters

from train.pipeline_manager import prepare_renko
from utils.renko_utils import tick21renko
from utils.utils import JAUNE, RESET

path_fus="/home/pierre/.wine/drive_c/Program Files/Fusion Markets MetaTrader 5/terminal64.exe"
path_adm='/home/pierre/.wine/drive_c/Program Files/Admiral Markets MT5/terminal64.exe'
path_meta='/home/pierre/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"

mt5 = MetaTrader5(port=18812)  # Valeurs par défaut

import copy
import os
import sys

from backtest.backtest_module import run_backtest

# Désactive les optimisations oneDNN qui causent souvent des erreurs de pointeurs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Désactive les logs excessifs de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import pandas as pd
import numpy as np
import optuna
from optuna.storages import fail_stale_trials
import tensorflow as tf
import gc
from config import SQLALCHEMY_DATABASE_URL
from tensorflow.keras import backend as K
import warnings

# Masquer les avertissements expérimentaux d'Optuna
# warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

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

# ====================== CONFIGURATION ======================
# Ajout du chemin racine pour les imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

RENKO_CACHE_DIR = "/media/pierre/datad/data/renko_cache"
TOTAL_MAX_TRIALS = 1024
BATCH_TRIALS = 2560  # Nombre maximal de trials exécutés avant de recycler le processus (RAM)
OPTION = ['F', 'T', 'D']
df_ticks = None
df_renko = None
config_base = {}
batch_data = []

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
    trial_counter = getattr(objective_wrapper, 'counter', 0)
    if trial_counter >= BATCH_TRIALS:
        print(f"♻️ Limite locale par lot atteinte ({BATCH_TRIALS} trials). Fermeture propre pour recyclage RAM...")
        #save_batch_log()
        #print(f"{JAUNE}----------------------------------- +++ Log saugardés{RESET}")
        study.stop()
        import sys
        sys.exit(10)  # Code 10 spécifique : demande de relancement propre au superviseur

    setattr(objective_wrapper, 'counter', trial_counter + 1)

    # 4. Sinon, on lance votre calcul habituel
    return objective(trial)

def save_batch_log(logs_dir="./optimization_logs"):
    """
    Sauvegarde les résultats d'un batch de 30 tests dans un fichier unique.
    batch_data : liste de dictionnaires contenant les scores, profits, et hcode de chaque test.
    """
    os.makedirs(logs_dir, exist_ok=True)
    pid = os.getpid()
    timestamp_ns = time.time_ns()  # Horodatage ultra-précis en nanosecondes
    # Combinaison unique : PID + Numéro de trial + Temps nanoseconde
    filename = f"batch_{pid}_{timestamp_ns}.json"
    filepath = os.path.join(logs_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(batch_data, f, indent=2)

def objective(trial):
    # Reset au début de chaque essai
    K.clear_session()
    tf.keras.backend.clear_session()

    # 1. Définition de l'espace de recherche (Paramètres)
    """
    optionSL = list(range(30, 121, 5))
    optionTP = list(range(40, 201, 5))

    window_monitor = trial.suggest_int('window_monitor', 45, 75, step=5)
    close_level_rnn = trial.suggest_float("close_level_rnn", 0.0, 0.5, step=0.1)
    open_level_rnn = trial.suggest_float("open_level_rnn", 1.3, 2.0, step=0.1)
    close_level_tabicl = trial.suggest_float("close_level_tabicl", 0.0, 0.5, step=0.1)
    open_level_tabicl = trial.suggest_float("open_level_tabicl", 1.3, 2.0, step=0.1)

    """
    renko_size = round(trial.suggest_float('renko_size', 9.0, 22.1, step=0.1), 1)
    ema_period = trial.suggest_int('ema_period', 6, 15)
    rsi_period = trial.suggest_int('rsi_period', 8, 16)
    macd_fast = trial.suggest_int('macd_fast', 4, 13)
    macd_slow = trial.suggest_int('macd_slow', 10, 30)
    macd_signal = trial.suggest_int('macd_signal', 3, 11)

    # Optimisation des bornes dynamiques
    threshold_sell = round(trial.suggest_float("threshold_sell", 0.15, 0.35, step=0.01), 2)
    close_buy = round(trial.suggest_float('close_buy', 0.3, 0.5, step=0.01), 2)
    close_sell = round(trial.suggest_float("close_sell", 0.5, 0.7, step=0.01), 2)
    threshold_buy = round(trial.suggest_float("threshold_buy", 0.65, 0.85, step=0.01), 2)
    
    # Paramètres pour le filtrage temporel
    min_stability_time = trial.suggest_int('min_stability_time', 180, 600, step=60)  # 3 à 10 minutes
    
    # Poids des modèles pour la combinaison IA + Rules
    weight_lstm = trial.suggest_int("weight_lstm", 1, 10, step=1)
    weight_jepa = trial.suggest_int("weight_jepa", 1, 10, step=1)
    weight_tab = trial.suggest_int("weight_tab", 1, 10, step=1)
    weight_cat = trial.suggest_int("weight_cat", 1, 10, step=1)
    weight_xgb = trial.suggest_int("weight_xgb", 1, 10, step=1)
    target_col = trial.suggest_categorical('target_col', ['diff_close', 'diff_ema', 'diff_rsi'])
    # 1. Utiliser un tuple incluant JEPA dans les choix de modèles

    version_choices = (
        "CAT, TAB",
        "JEPA",
        "CAT, JEPA",
        "TAB, JEPA",
        "TAB",
        "FINJEPA, TABFIN",
        "TAB, FINJEPA, TABFIN, JEPA",
        "CAT, TAB, FINJEPA, TABFIN, JEPA"
    )
    """
    version_choices = {
        "TAB, JEPA",
    }
    """
    config = copy.deepcopy(config_base)  # Utilisez une config "vierge"

    config["parameters"]["rsi_period"] = rsi_period
    config["parameters"]["macd"]["macd_fast"] = macd_fast
    config["parameters"]["macd"]["macd_slow"] = macd_slow
    config["parameters"]["macd"]["macd_signal"] = macd_signal
    """
    config["parameters"]["window_monitor"] = window_monitor
    config["parameters"]["close_level_rnn"] = close_level_rnn
    config["parameters"]["open_level_rnn"] = open_level_rnn
    config["parameters"]["close_level_tabicl"] = close_level_tabicl
    config["parameters"]["open_level_tabicl"] = open_level_tabicl
    """
    config["parameters"]["renko_size"] = renko_size
    config["parameters"]["ema_period"] = ema_period
    config["parameters"]["threshold_buy"] = threshold_buy
    config["parameters"]["threshold_sell"] = threshold_sell
    config["parameters"]["close_buy"] = close_buy
    config["parameters"]["close_sell"] = close_sell
    
    # Configuration des poids pour la combinaison IA + Rules
    config["parameters"]["weights"] = {
        "LSTM": weight_lstm,
        "JEPA": weight_jepa,
        "TAB": weight_tab,
        "TABFIN": weight_tab,
        "FINJEPA": weight_cat,
        "CAT": weight_cat,
        "XGB": weight_xgb,
        "LGBM": trial.suggest_int("weight_lgbm", 1, 10, step=1)
    }
    
    # Paramètres pour le filtrage temporel (à utiliser dans enhanced_decision)
    config["parameters"]["min_stability_time"] = min_stability_time

    # config["features"] = ["time_live", "close", "diff_close", "diff_ema", "RSI", "diff_macd"]
    chosen_string = trial.suggest_categorical("version", version_choices)
    config['live']["version"] = [v.strip() for v in chosen_string.split(",")]
    # config['live']['version'] = ['CAT']
    # config["live"]["version"] = ['TAB']
    # config["live"]["version"] = ['LGBM']
    # config["live"]["version"] = ["JEPA"]
    for vs in config['live']['version']:
        if 'CAT' == vs:
            config["catboost"] = {
                "iterations": trial.suggest_int('iterations', 200, 2000, step=200),
                "depth": trial.suggest_int('depth', 4, 10, step=1),
                "learning_rate":  trial.suggest_float('learning_rate', 0.01, 0.31, step=0.05),
                "l2_leaf_reg": trial.suggest_int('l2_leaf_reg', 1, 10, step=1)
            }
        if 'LGBM' == vs:
            config["lgbm"] = {
                "lgbm_learning_rate" : trial.suggest_categorical('lgbm_learning_rate', [0.01, 0.03, 0.05]),
                "lgbm_num_leaves": trial.suggest_categorical('lgbm_num_leaves', [15, 31, 63]),
                "lgbm_feature_fraction": trial.suggest_categorical('lgbm_feature_fraction', [0.7, 0.8, 0.9]),
                "lgbm_bagging_fraction": trial.suggest_categorical('lgbm_bagging_fraction', [0.7, 0.8, 0.9]),
                "lgbm_min_child_samples": trial.suggest_categorical('lgbm_min_child_samples', [20, 50]),
                "lgbm_early_stop_rounds": trial.suggest_categorical('lgbm_early_stop_rounds', [20, 50]),
            }
        if 'FIMJEPA' == vs:
            config["finjepa"] = {
                # Longueur du contexte historique analysé par la Fin-JEPA
                "context_len": trial.suggest_categorical('fin_context_len', [30, 45, 60, 90]),
                # Horizon de prédiction (cible)
                "target_len": trial.suggest_categorical('fin_target_len', [10, 15, 20, 30]),
                # Taille de batch fixée pour la stabilité sur CPU
                "batch_size": 32,
                # Nombre d'époques resserré pour que l'entraînement reste instantané
                "epochs": trial.suggest_int('fin_epochs', 2, 6),
                # Taux d'apprentissage exploré sur une échelle logarithmique
                "lr": trial.suggest_float('fin_lr', 5e-5, 1e-3, log=True),
            }
        if 'JEPA' == vs:
            config["jepa"] = {
                "SEQ_LEN": trial.suggest_categorical('SEQ_LEN', [32, 64, 96, 128]),
                # Passage en discret pour cibler les tailles idéales
                "BATCH_SIZE": 32,  # Fixé, pas besoin de le faire varier si 32 stabilise bien vos epochs
                "HIDDEN_DIM": trial.suggest_categorical('HIDDEN_DIM', [64, 128, 256]),
                # Élargissement léger si le modèle a besoin de capacité
                "LATENT_DIM": trial.suggest_categorical('LATENT_DIM', [32, 64, 128]),  # Proportionnel au hidden_dim
                "NUM_LAYERS": trial.suggest_int('NUM_LAYERS', 1, 2),  # Validé par vos tests précédents
                "LR": trial.suggest_float('LR', 5e-4, 5e-3, log=True),
                # Plage resserrée autour des zones de convergence stables
                "NUM_EPOCHS": trial.suggest_int('NUM_EPOCHS', 1, 3),  # Un peu plus de temps de convergence
                "INPUT_DIM": 6,
        }
        if 'LSTM' == vs:
            config["lstm"]["lstm_seq_len"] = trial.suggest_int('lstm_seq_len', 24, 64, step=8)
            config["lstm"]["lstm_units"] = trial.suggest_int('lstm_units', 48, 240, step=48)
        if 'MLP' == vs:
            config["mlp"]["mlp_unit1"] = trial.suggest_int('mlp_unit1', 128, 256, step=128)
            config["mlp"]["mlp_dropout"] = trial.suggest_float('mlp_dropout', 0.2, 0.5, step=0.1)
            config["mlp"]["mlp_patience"] = trial.suggest_int('mlp_patience', 10, 20)
        if 'XGB' == vs:
            config["xgb"]["xgb_learning_rate"] = trial.suggest_categorical('xgb_learning_rate', [0.01, 0.03, 0.05])
            config["xgb"]["xgb_max_depth"] = trial.suggest_categorical('xgb_max_depth', [4, 6, 8])
            config["xgb"]["xgb_subsample"] = trial.suggest_categorical('xgb_subsample', [0.7, 0.8, 0.9])
            config["xgb"]["xgb_colsample_bytree"] = trial.suggest_categorical('xgb_colsample_bytree', [0.7, 0.8, 0.9])
        if 'GRU' == vs:
            config["gru"]["gru_seq_len"] = trial.suggest_int('gru_seq_len', 24, 64, step=8)
            config["gru"]["gru_units1"] = trial.suggest_int("gru_units1", 32, 128, step=32)
            config["gru"]["gru_units2"] = trial.suggest_int("gru_units2", 16, 64, step=16)
            config["gru"]["gru_lr"] = trial.suggest_float("gru_lr", 0.0001, 0.01, log=True)
            config["gru"]["gru_dropout"] = trial.suggest_float("gru_dropout", 0.1, 0.4, step=0.1)
            config["gru"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 128])
            config["gru"]["gru_patience"] = 10

    config["target"]["target_col"] = [target_col]

    # 3. On redécoupe la chaîne pour retrouver votre liste de modèles
    # config['live']["sl"] = trial.suggest_categorical('sl', optionSL)
    # config['live']["tp"] = trial.suggest_categorical('tp', optionTP)

    # Optimisation de l'option VSIMPLE/VTOTALE/VDIRECT
    option_str = trial.suggest_categorical("option", ["TTT", "TTF", "TFT", "TFF", "FTT", "FTF", "FFT", "FFF"])
    config["live"]["option"] = option_str
    # config['live']['version'] = ['CAT']
    # config["live"]["version"] = ['TAB']
    # config["live"]["version"] = ['LGBM']
    # config["live"]["version"] = ["JEPA"]
    # config['live']["sl"] = trial.suggest_categorical('sl', optionSL)
    # config['live']["tp"] = trial.suggest_categorical('tp', optionTP)

    # 2. Chargement des données
    try:
        current_renko_size = config['parameters']['renko_size']
        file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{current_renko_size:.1f}.pkl")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier manquant: {file_path}")

        full_df = pd.read_pickle(file_path)
        if not pd.api.types.is_datetime64_any_dtype(full_df['time']):
            full_df['time'] = pd.to_datetime(full_df['time'], unit='ms', errors='coerce')

        last_date = full_df['time'].max()
        if pd.isna(last_date):
            print(f"⚠️ Fichier corrompu ou vide pour taille {current_renko_size}")
            return -9999.0

        if len(full_df) < 1450:
            print(f"⚠️ Pas assez de briques ({len(config['data'])}) pour taille {current_renko_size}")
            return -9998.0
        config['data'] = full_df.copy()
        print(
            f"✅ Trial {trial.number} | Fin: {last_date.strftime('%Y-%m-%d')} | rSize {current_renko_size:.1f} | Segment: {len(config['data'])} briques")
        score, result_dict = run_backtest(config, trial=trial)
        # ------------------------------------------------------> récupération de chaque test ici
        # On s'assure que le résultat existe et n'est pas un rejet/erreur
        if result_dict is not None and isinstance(result_dict, dict) and len(result_dict) > 0:
            hcode = config.get('live', {}).get('hcode', f"trial_{trial.number}")
            test_result_item = {
                "trial_number": trial.number,
                "score": float(score),
                "profit": float(result_dict.get('profit', 0)),
                "trades": int(result_dict.get('trades', 0)),
                "hcode": hcode
            }
            batch_data.append(test_result_item)
        else:
            # Optionnel : loguer un rejet si besoin
            print(f"⚠️ Trial {trial.number} ignoré pour les logs (résultat invalide ou rejeté).")

        return float(score)
        return float(score)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Erreur Trial {trial.number}: {e}")
        return -9999.0
    finally:
        K.clear_session()
        tf.keras.backend.clear_session()
        if 'data' in config:
            del config['data']
        gc.collect()
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

    storage_url = SQLALCHEMY_DATABASE_URL
#    study_name = "trading_ONE"
    study_name = "robust_trading_optimization"
    # study_name = "trading_diff_postg"
    pruning_step = 15  # au lieu de 5 ou 10 par exemple
    # À l'initialisation de votre étude Optuna :
    # 1. Configuration avancée pour privilégier la robustesse et les interactions
    sampler = optuna.samplers.TPESampler(
        multivariate=True,  # Active l'analyse croisée des paramètres (liaison des variables)
        group=True,  # Groupe logiquement les paramètres par type/modèle
        n_startup_trials=60,  # Augmente les essais aléatoires initiaux pour bien cartographier
        constant_liar=True  # Optimise l'exécution en parallèle (si vous utilisez du multiprocessing)
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(  # ou Remplace le pruning par un HyperbandPruner ou augmente fortement les seuils.
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=pruning_step        )
    )

    #fail_stale_trials(study)
    #total_faits = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    total_faits = len(study.trials)
    restants = TOTAL_MAX_TRIALS - total_faits

    print("-" * 50)
    print(f"📊 BILAN DE L'ÉTUDE : {study_name}")
    print(f"✅ Tests déjà validés dans Postgres : {total_faits}")
    print(f"🎯 Objectif total : {TOTAL_MAX_TRIALS}")

    if restants > 0:
        print(f"🚀 Cette instance va participer à l'exécution des {restants} tests restants.")
        print("-" * 50)
        n_trials = TOTAL_MAX_TRIALS
        n_jobs = 4
        print(f"Optimisation : {n_trials} essais sur {n_jobs} coeurs.")
        study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs)
    else:
        print("🛑 Objectif déjà atteint.")

    print(f"\n🏆 Les 10 meilleurs modèles (RNN + TabICL) ont été sauvegardés automatiquement dans backtest_module.py")
    save_batch_log()
    print(f"{JAUNE}----------------------------------- +++ Log saugardés{RESET}")
    print(f"Durée totale : {(time.time() - start):.0f} secondes")

def load_ticks_incremental_forward(symbol, start_date_dt, chunk_days=2):
    all_chunks = []
    # On commence avec la date demandée
    current_start_dt = start_date_dt
    final_end_dt = datetime.now()

    last_msc_processed = 0

    print(f"🚀 Récupération précise : {symbol} depuis {current_start_dt}")

    while current_start_dt < final_end_dt:
        segment_end_dt = current_start_dt + timedelta(days=chunk_days)
        if segment_end_dt > final_end_dt:
            segment_end_dt = final_end_dt

        # Requête MT5
        ticks = mt5.copy_ticks_range(symbol, current_start_dt, segment_end_dt, mt5.COPY_TICKS_ALL)

        if ticks is not None and len(ticks) > 0:
            df_chunk = pd.DataFrame(ticks)

            # GESTION DES DOUBLONS DE JONCTION
            # Si le premier tick de ce segment est le même que le dernier du précédent, on l'enlève
            if last_msc_processed > 0:
                df_chunk = df_chunk[df_chunk['time_msc'] > last_msc_processed]

            if not df_chunk.empty:
                all_chunks.append(df_chunk)
                last_msc_processed = int(df_chunk['time_msc'].iloc[-1])

                # Mise à jour du curseur sur le dernier temps reçu
                current_start_dt = segment_end_dt
                print(f"  ✅ {len(df_chunk)} nouveaux ticks (Total cumulé: {sum(len(c) for c in all_chunks)})")
            else:
                # Si après filtrage c'est vide, on doit quand même avancer le temps
                # pour ne pas redemander le même micro-segment
                current_start_dt = segment_end_dt
        else:
            print(f"  ⚠️ Vide sur ce segment. On avance à {segment_end_dt}")
            current_start_dt = segment_end_dt

        # SÉCURITÉ ANTI-STAGNATION
        # Si le curseur n'avance pas (ex: moins d'une ms d'écart), on force l'avance au segment_end
        if 'old_start' in locals() and current_start_dt <= old_start:
            current_start_dt = segment_end_dt

        old_start = current_start_dt
        time.sleep(0.05)  # Pause minimale pour le pont RPC

    if not all_chunks: return pd.DataFrame()

    final_df = pd.concat(all_chunks).drop_duplicates(subset=['time_msc']).sort_values('time_msc')
    return final_df

RENKO_SIZES_TO_PREPARE = np.arange(16.0, 28.0, 0.1)  # Assurez-vous que cette liste est à jour

def create_renko_file(renko_size):
    global df_ticks
    """
    Fonction exécutée par chaque worker :
    1. Charge les ticks bruts.
    2. Calcule les bougies Renko pour UNE taille.
    3. Sauvegarde le résultat dans un fichier .pkl dédié.
    """
    renko_size = round(renko_size, 1)
    file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{renko_size:.1f}.pkl")

    # Si le fichier existe déjà, on ne fait rien pour gagner du temps
    """
    if os.path.exists(file_path):
        print(f"Cache HIT: Le fichier pour renko_size={renko_size} existe déjà. Skip.")
        return
    """
    print(f"Cache MISS: Création du fichier pour renko_size={renko_size}...")
    try:
        # On calcule les bougies Renko
        df_renko = tick21renko(df_ticks, None, renko_size, 'bid')

        # On sauvegarde en format pickle, très rapide à lire
        df_renko.to_pickle(file_path)
        print(f"SUCCÈS: Fichier créé pour renko_size={renko_size}")
    except Exception as e:
        print(f"ERREUR lors de la création pour renko_size={renko_size}: {e}")

def load_partial():
    global df_ticks, df_renko
    filename = f"/media/pierre/datad/data/ETHUSD_150.csv"
    if os.path.exists(filename):
        df_ticks = pd.read_csv(filename, sep=";")
        df_ticks.set_index('time_msc', inplace=True)
        df_ticks.index = pd.to_datetime(df_ticks.index, unit='ms')
        if "renko_volatility_ratio" in config_base["features"]:
            prepare_renko(config_base, df_ticks)
    elif mt5.initialize(path_meta=path_meta, portable=True):
        account_info = mt5.account_info()
        print("name   = {}".format(account_info.name))
        print("login  = {}".format(account_info.login))
        print("server = {}".format(account_info.server))
        symbol = "ETHUSD"  # Remplacez par votre symbole exact
        # On demande les ticks depuis l'an 2000
        date_test = datetime(2025, 8, 1, tzinfo=timezone.utc)

        # On essaie de récupérer 1 seul tick à partir de cette date
        ticks = mt5.copy_ticks_from(symbol, date_test, 1, mt5.COPY_TICKS_ALL)

        if ticks is not None and len(ticks) > 0:
            first_tick_time = pd.to_datetime(ticks[0]['time_msc'], unit='ms')
            print(f"✅ Le premier tick disponible sur ce serveur est le : {first_tick_time}")
        else:
            print("❌ Aucune donnée trouvée sur ce serveur.")
        df_ticks = load_ticks_incremental_forward("ETHUSD", datetime.now() - timedelta(days=180), chunk_days=2)
        print("fund fini")
        mt5.shutdown()
        pd.DataFrame(df_ticks).to_csv(filename, sep=";", index=False)
        df_ticks.set_index('time_msc', inplace=True)
        df_ticks.index = pd.to_datetime(df_ticks.index, unit='ms')
        if "renko_volatility_ratio" not in config_base["features"]:
            prepare_renko(config_base, df_ticks)
        else:
            import multiprocessing as mp
            os.makedirs(RENKO_CACHE_DIR, exist_ok=True)
            print(f"Démarrage du pré-calcul pour {len(RENKO_SIZES_TO_PREPARE)} tailles de Renko...")
            print(f"Les fichiers seront sauvegardés dans le dossier: '{RENKO_CACHE_DIR}'")
            # Utilise un Pool de processus pour paralléliser la création des fichiers
            # Prend tous les coeurs disponibles moins un pour garder le système réactif
            num_cpus = max(1, mp.cpu_count() - 6)
            with mp.Pool(processes=num_cpus) as pool:
                pool.map(create_renko_file, RENKO_SIZES_TO_PREPARE)

            print("\nPré-calcul de toutes les bougies Renko terminé !")
            gc.collect()
    else:
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas")

    df_renko = prepare_renko(config_base, df_ticks)
    del df_ticks

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    config_path = os.path.join(ROOT_DIR, "config_test.json")
    if not os.path.exists(config_path):
        print(f"Fichier config_live.json manquant.")
        exit(111)
    else:
        with open(config_path, 'r') as f:
            config_base = json.load(f)
            if "live" not in config_base:
                for k, v in config_base.items():
                    if isinstance(v, dict) and "live" in v:
                        config_base = v
                        break
    # load_partial()
    optimize_start()
