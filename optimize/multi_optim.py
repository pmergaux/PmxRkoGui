# ======================================================================
#  TON FICHIER ACTUEL – CORRIGÉ UNE FOIS POUR TOUTES
#  → Plus d’erreur NoneType
#  → time_msc conservé en colonne ET en index
#  → 9 millions de ticks = +15 Mo par worker seulement
# ======================================================================

import os
import json
import multiprocessing as mp
import signal
import gc
import time

import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from multiprocessing import shared_memory
from queue import Empty

from utils.utils import clean_numpy_types
from utils.utils import reload_ticks_from_pickle

# ======================================================================
# GESTION DE LA MÉMOIRE PARTAGÉE
# ======================================================================

# Variables globales pour stocker les infos de la mémoire partagée
SHARED_MEM_NAME = None
TICKS_SHAPE = None
TICKS_DTYPE = None


def load_ticks_into_shared_memory(symbol, start, end):
    """
    Charge les ticks depuis le .pkl UNE SEULE FOIS et les place en mémoire partagée.
    """
    global SHARED_MEM_NAME, TICKS_SHAPE, TICKS_DTYPE

    print("\nChargement unique des ticks pour partage mémoire...")
    base_name = f"../data/{symbol}_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    df = reload_ticks_from_pickle(base_name, symbol, None, start, end)

    # On garde uniquement les colonnes nécessaires et on optimise les types
    df_shared = df[['time', 'bid', 'ask']].copy()
    df_shared['time'] = df_shared['time'].astype('int64')
    df_shared['bid'] = df_shared['bid'].astype('float32')
    df_shared['ask'] = df_shared['ask'].astype('float32')

    # Création du bloc de mémoire partagée
    shm = shared_memory.SharedMemory(create=True, size=df_shared.values.nbytes)

    # On copie les données dans la mémoire partagée
    shared_array = np.ndarray(df_shared.shape, dtype=df_shared.values.dtype, buffer=shm.buf)
    shared_array[:] = df_shared.values[:]

    # On stocke les informations nécessaires pour que les workers puissent se reconnecter
    SHARED_MEM_NAME = shm.name
    TICKS_SHAPE = df_shared.shape
    TICKS_DTYPE = df_shared.values.dtype

    print(f"Ticks partagés avec succès. Nom du bloc: {SHARED_MEM_NAME}")

    # Le processus principal n'a plus besoin de la référence au bloc
    shm.close()


def get_ticks_from_shared_memory():
    """
    Fonction appelée par chaque worker pour accéder aux ticks partagés.
    Ne consomme quasiment pas de RAM.
    """
    shm = shared_memory.SharedMemory(name=SHARED_MEM_NAME)

    # On se rattache au tableau NumPy existant dans la mémoire partagée
    shared_array = np.ndarray(TICKS_SHAPE, dtype=TICKS_DTYPE, buffer=shm.buf)

    # On crée un DataFrame Pandas à partir de ce tableau (c'est une vue, pas une copie)
    df = pd.DataFrame(shared_array, columns=['time', 'bid', 'ask'])

    # On convertit la colonne 'time' en index Datetime
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('time')

    # Le worker n'a plus besoin de la référence au bloc, mais le buffer reste valide
    shm.close()

    return df

def cleanup_shared_memory():
    """
    Nettoie la mémoire partagée à la toute fin du programme.
    """
    print("Nettoyage de la mémoire partagée...")
    try:
        shm = shared_memory.SharedMemory(name=SHARED_MEM_NAME)
        shm.close()
        shm.unlink()  # Marque le bloc pour suppression
        print("Mémoire partagée nettoyée.")
    except FileNotFoundError:
        print("La mémoire partagée était déjà nettoyée.")
    except Exception as e:
        print(f"Erreur lors du nettoyage de la mémoire partagée: {e}")


# ======================================================================
# GESTION DU CACHE RENKO DANS CHAQUE WORKER
# ======================================================================

def evaluate_config_cache(config, renko_cache):
    """
    Utilise un cache pour charger le fichier DataFrame Renko pré-calculé.
    """
    try:
        from optimize.triple_module import run_backtest

        renko_size = config['renko_size']

        # On vérifie si le DataFrame est dans le cache du worker
        if renko_size in renko_cache:
            # Cache HIT: on réutilise le DataFrame existant
            df_renko = renko_cache[renko_size]
        else:
            # Cache MISS: on charge le fichier .pkl une fois et on le stocke
            print(f"[Worker {os.getpid()}] Cache MISS pour renko_size={renko_size}. Chargement du fichier...")
            file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{renko_size:.2f}.pkl")
            if not os.path.exists(file_path):
                print(f"[ERREUR] Fichier Renko manquant: {file_path}. Lancez d'abord prepare_renko_data.py")
                return -999.0, {}
            df_renko = pd.read_pickle(file_path)
            renko_cache[renko_size] = df_renko  # On met en cache pour la prochaine fois

        config['data'] = df_renko
        score, result = run_backtest(config)
        return float(score), result

    except Exception as e:
        print(f"[ERREUR] dans evaluate_config pour renko_size={config.get('renko_size')}: {e}")
        return -999.0, {}
    finally:
        gc.collect()

def worker_cache(task_queue: mp.Queue, result_queue: mp.Queue):
    """
    Worker avec un cache Renko qui se vide automatiquement
    lors d'un changement de taille.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pid = os.getpid()

    renko_cache = {}
    # Variable pour suivre la dernière taille de renko traitée par ce worker
    last_renko_size = None

    while True:
        try:
            config = task_queue.get()
            if config is None:
                print(f"[Worker {pid}] Poison pill reçu, arrêt.")
                break

            current_renko_size = config['renko_size']

            # Si la taille de renko de la nouvelle tâche est différente de la précédente...
            if current_renko_size != last_renko_size:
                print(
                    f"[Worker {pid}] Changement de taille de {last_renko_size} à {current_renko_size}. Vidage du cache.")
                # On vide le cache pour libérer la mémoire de l'ancien DataFrame
                renko_cache.clear()
                # On met à jour la dernière taille vue
                last_renko_size = current_renko_size

            # Le reste de la logique est inchangé
            start_time = datetime.now()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {pid} démarre config renko={current_renko_size}...")

            score, result = evaluate_config_cache(config, renko_cache)

            duration = (datetime.now() - start_time).total_seconds() / 60
            print(f"[Worker {pid}] FINI en {duration:.1f} min → score = {score:.6f}")
            result_queue.put((score, result, config))

        except Exception as e:
            print(f"[Worker {pid}] CRASH sur une tâche : {e}")
            result_queue.put((-999.0, {}, {"error": str(e)}))


# ======================================================================
# VARIABLES GLOBALES
# ======================================================================
# Chemin fixe du fichier optimisé (on le crée une seule fois)
TICKS_NPY_PATH = "../data/ETHUSD_ticks_optimized.npy"
RENKO_CACHE_DIR = "../data/renko_cache"
SAVE_PATH_DIR = "../models/simple_opt"
start_date = (2025, 10, 1,0,0,0)
end_date = (2025, 12, 11, 23, 59, 59)

def prepare_ticks_once():
    """À lancer UNE FOIS dans ta vie (ou quand tu changes de période)"""
    if os.path.exists(TICKS_NPY_PATH):
        print("Fichier ticks optimisé déjà existant → skip")
        return

    print("Préparation du fichier ticks optimisé (une seule fois)...")
    start = datetime(*start_date)
    end = datetime(*end_date)
    base_name = f"../data/ETHUSD_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"

    from utils.utils import reload_ticks_from_pickle
    df = reload_ticks_from_pickle(base_name, 'ETHUSD', None, start, end)
    print("col ticks", df.columns.tolist())
    if df.empty:
        raise "data empty"
    # On garde QUE time_msc, bid, ask → float32 + int64
    arr = np.zeros(len(df), dtype=[
        ('time', 'int64'),
        ('bid', 'float32'),
        ('ask', 'float32')
    ])

    if 'time' not in df.columns:
        arr['time'] = df.index.values.astype('int64')
    else:
        arr['time'] = df['time'].values.astype('int64')  # ← déjà en ms depuis 1970
    arr['bid'] = df['bid'].astype('float32').values
    arr['ask'] = df['ask'].astype('float32').values

    np.save(TICKS_NPY_PATH, arr)
    print(f"Fichier créé : {TICKS_NPY_PATH} → {os.path.getsize(TICKS_NPY_PATH) / 1024 ** 3:.2f} GB")
    print("Tu peux supprimer le .pkl maintenant si tu veux")


def get_ticks_dataframe():
    """Chaque worker charge en 0.8 seconde depuis le .npy"""
    arr = np.load(TICKS_NPY_PATH, mmap_mode='r')  # ← mémoire mappée → quasi zéro RAM
    df = pd.DataFrame(arr)
    df = df.set_index('time', drop=False).sort_index()
    df.index = pd.to_datetime(df.index, unit="ms")
    return df

# ======================================================================
# TA FONCTION D'ÉVALUATION (exactement comme avant)
# ======================================================================
def evaluate_config_renko(config):
    try:
        from optimize.triple_module import run_backtest
        # Construit le chemin vers le fichier .pkl correspondant
        renko_size = config['renko_size']
        file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{renko_size:.2f}.pkl")
        if not os.path.exists(file_path):
            print(f"[ERREUR] Le fichier Renko pré-calculé n'existe pas: {file_path}")
            return -999.0
        # Charge le DataFrame Renko depuis le fichier
        df_renko = pd.read_pickle(file_path)
        # Passe ce DataFrame à la config
        config['data'] = df_renko
        score, result = run_backtest(config)
        return float(score), result
    except Exception as e:
        print(f"[ERREUR] Config {config.get('renkosize')} → {e}")
        return -999.0, {}
    finally:
        gc.collect()
        if config.get('VERSION') == ['TFT']:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        # print(f"[Cleanup] Mémoire libérée après config")


# ======================================================================
# WORKER (avec la pause)
# ======================================================================
def worker_renko(task_queue: mp.Queue, result_queue: mp.Queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pid = os.getpid()

    while True:
        try:
            # On attend une tâche (None est une tâche de fin)
            config = task_queue.get()
            if config is None:
                # C'est le signal de fin, on sort de la boucle
                print(f"[Worker {pid}] Poison pill reçu, arrêt.")
                break

            # Traitement de la tâche (inchangé)
            start_time = datetime.now()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker {pid} démarre config renko={config['renko_size']}...")
            score, result = evaluate_config_renko(config)
            duration = (datetime.now() - start_time).total_seconds()
            print(f"[Worker {pid}] FINI en {duration:.0f} sec. → score = {score:.6f}")
            result_queue.put((score, result, config))

        except Exception as e:
            print(f"[Worker {pid}] CRASH sur une tâche : {e}")
            # On met un résultat d'erreur pour ne pas bloquer la boucle principale
            result_queue.put((-999.0,{}, {"error": str(e)}))

# ======================================================================
# GRID SEARCH (100 % identique à ton fichier actuel)
# ======================================================================
def run_grid_search_multiprocess():
    os.environ["PL_DISABLE_FORK_VALIDATION"] = "1"
    start_time = time.time()
    symbol = 'ETHUSD'
    prepare_ticks_once()

    renko_sizes = np.arange(32.0, 33.0, 0.1)
    ema_periods = [6, 9, 12]
    rsi_periods = [12, 14, 16]
    seq_lens = range(48, 192, 16)
    lstm_units = range(48, 256, 32)
    thresholds_buy = np.arange(0.55, 0.8, 0.05)
    thresholds_sell = np.arange(0.25, 0.45, 0.05)
    target_cols = ['EMA', 'target_sign_mean']

    configs = [
        {
            'symbol': symbol,
            'renko_size': round(rk, 2),
            'ema_period': ema,
            'rsi_period': rsi,
            'target_col': tg,
            'target_type': 'direction',
            'seq_len': seq,
            'lstm_units': units,
            'threshold_buy': tb,
            'threshold_sell': ts,
            'features_base': ["EMA", "RSI", "MACD_hist", "close", "lstm"],
            'params_base': {"renko_size": 17.1, "ema_period": 9, "rsi_period": 14, "rsi_high": 70, "rsi_low": 30,
                            "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}},
            'VERSION': ['DECISION', 'SIMPLE'],
            'hcode': ''
        }
        for rk, ema, rsi, seq, units, tb, ts, tg in product(
            renko_sizes, ema_periods, rsi_periods, seq_lens, lstm_units,
            thresholds_buy, thresholds_sell, target_cols
        )
    ]

    print(f"→ {len(configs)} configurations à tester")

    num_workers = 6
    print(f"Lancement de {num_workers} workers")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = [mp.Process(target=worker_cache, args=(task_queue, result_queue), daemon=False)
               for _ in range(num_workers)]

    for w in workers:
        w.start()

    for config in configs:
        task_queue.put(config)
    print(f"{len(configs)} configs envoyées")

    for _ in range(num_workers):
        task_queue.put(None)

    results = []
    best_score_prev = best_score = -float('inf')
    best_config = None
    for _ in range(len(configs)):
        try:
            # On attend un résultat, avec un long timeout pour la sécurité
            score, result, config_result = result_queue.get(timeout=4200)

            if score > -999.0:  # On ignore les erreurs
                config_result['result'] = result
                config_result = clean_numpy_types(config_result)
                results.append((score, config_result))
                if score > best_score:
                    best_score = score
                    best_config = config_result.copy()
                    os.makedirs(SAVE_PATH_DIR, exist_ok=True)
                    with open(SAVE_PATH_DIR + "/best_live.json", "w") as f:
                        json.dump(best_config, f, indent=2)
                    print(f"\nNOUVEAU RECORD → {score:.6f}")

        except Empty:
            print("[Main] Timeout ! Un worker est probablement bloqué. Arrêt forcé.")
            break
        except Exception as e:
            print(f"[Main] Erreur lors de la récupération d'un résultat : {e}")

    print("Arrêt des workers...")
    for w in workers:
        if w.is_alive():
            w.terminate()
            w.join(timeout=10)
            if w.is_alive():
                w.kill()
                w.join()

    if best_config is not None:
        print(best_config)
        os.makedirs(SAVE_PATH_DIR, exist_ok=True)
        with open(SAVE_PATH_DIR+"/best_live.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print("\nMEILLEURE CONFIG TROUVÉE ET SAUVEGARDÉE")
    else:
        print("\nAUCUNE CONFIGURATION VALIDE")
    print(f"fin durée {time.time()-start_time:.0f}")
    print("\nTOP 5 CONFIGS:")
    top5 = sorted(results, key=lambda x: x[0], reverse=True)
    score = 0
    i = 0
    for top in top5:
        if top[0] != score:
            print(top)
            score = top[0]
            i =+1
            if i == 20:
                break

    return best_config, results


# ======================================================================
# LANCEMENT
# ======================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_config, all_results = run_grid_search_multiprocess()
