import os
import json
import multiprocessing as mp
import signal
from datetime import datetime
from itertools import product

import pandas as pd
from tqdm import tqdm
import numpy as np

from utils.utils import reload_ticks_from_pickle
from queue import Empty  # ← crucial !


# ==================================================================
# TA FONCTION D'ÉVALUATION (doit être picklable et autonome)
# ==================================================================
def evaluate_config(config):
    """Évalue une configuration et retourne uniquement le score (float)"""
    try:
        # Import heavy libraries only inside the worker process
        from optimize.triple_module import run_backtest
        score = run_backtest(config)
        return float(score)
    except Exception as e:
        print(f"[ERREUR] Config {config.get('renko_size')} / {config.get('target_col')} → {e}")
        return -999.0
    finally:  # ← TOUJOURS exécuté, même en crash
        import gc
        import torch
        gc.collect()  # Garbage collect CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Libère GPU
            torch.cuda.ipc_collect()  # Nettoie IPC si shared tensors
        print(f"[Cleanup] Mémoire libérée après config")

# ==================================================================
# WORKER ULTRA-ROBUSTE pour tâches très longues
# ==================================================================
def worker(task_queue: mp.Queue, result_queue: mp.Queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    pid = os.getpid()

    while True:
        try:
            # Attente très longue : une tâche peut durer 40 min
            config = task_queue.get(timeout=4200)  # 1 heure max
        except Empty:
            print(f"[Worker {pid}] Timeout 1h sur task_queue → arrêt")
            break

        if config is None:  # poison pill unique
            print(f"[Worker {pid}] Poison pill → arrêt propre")
            break

        start_time = datetime.now()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Worker {pid} démarre config renko={config['renko_size']} "
              f"target={config['target_col']} buy={config['threshold_buy']:.3f} sell={config['threshold_sell']:.3f}")

        try:
            score = evaluate_config(config)
            duration = (datetime.now() - start_time).total_seconds() / 60
            print(f"[Worker {pid}] FINI en {duration:.1f} min → score = {score:.6f}")
            result_queue.put((score, config))

        except Exception as e:
            print(f"[Worker {pid}] CRASH après {(datetime.now() - start_time).total_seconds() / 60:.1f} min → {e}")
            result_queue.put((-999.0, {"error": str(e), "config": config}))

    # Signal de fin
    try:
        result_queue.put("WORKER_FINISHED")
    except:
        pass

# ==================================================================
# GRID SEARCH MULTIPROCESS ROBUSTE (sans Pool → contrôle total)
# ==================================================================
def run_grid_search_multiprocess():
    os.environ["PL_DISABLE_FORK_VALIDATION"] = "1"
    # ================ load datas ====================
    start = datetime(2025, 9, 1)
    end = datetime(2025, 11, 20, 23, 59, 59)
    base_name = f"../data/ETHUSD_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    df = reload_ticks_from_pickle(base_name, 'ETHUSD', None, start, end)
    if df is None or df.empty:
        print("Pas de données → exit")
        exit()
    df['time'] = pd.to_datetime(df['time'])
    # --- Paramètres du grid (identique à ton code) ---
    renko_sizes      = np.arange(15.0, 15.8, 0.1)
    ema_periods      = [9]
    rsi_periods      = [14]
    seq_lens         = [20]
    lstm_units       = [50]
    thresholds_buy   = np.arange(0.55, 0.75, 0.05)
    thresholds_sell  = np.arange(0.25, 0.45, 0.05)
    target_cols      = ['target_sign_mean']

    configs = [
        {
            'renko_size': round(rk, 2),
            'ema_period': ema,
            'rsi_period': rsi,
            'target_col': tg,
            'target_type': 'direction',
            'seq_len': seq,
            'lstm_units': units,
            'threshold_buy': round(tb, 3),
            'threshold_sell': round(ts, 3),
            'features_base': ["EMA", "RSI", "MACD_hist", "close", "time_live", "TFT"],
            'VERSION': ['TFT'],
            'hcode': '',
            'data': df
        }
        for rk, ema, rsi, seq, units, tb, ts, tg in product(
            renko_sizes, ema_periods, rsi_periods, seq_lens, lstm_units,
            thresholds_buy, thresholds_sell, target_cols
        )
    ]

    print(f"→ {len(configs)} configurations à tester")
    
    # FIX: Ensure at least 1 worker is used, leaving 2 cores free to avoid freezing the system.
    num_workers = 1  # max(1, mp.cpu_count() - 16)
    print(f"Lancement de {num_workers} workers pour tâches longues (10–40 min)")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = [mp.Process(target=worker, args=(task_queue, result_queue), daemon=False)
               for _ in range(num_workers)]

    for w in workers: w.start()

    # Envoi de toutes les tâches
    for config in configs:
        task_queue.put(config)
    print(f"{len(configs)} configs envoyées → attente des résultats...")

    # Poison pills CORRIGÉES : un par worker !
    for _ in range(num_workers):
        task_queue.put(None)

    # Collecte avec sauvegarde progressive
    results = []
    best_score_prev = best_score = -float('inf')
    best_config = None
    finished_workers = 0

    with tqdm(total=len(configs), desc="Grid TFT (long)", smoothing=0.05) as pbar:
        while finished_workers < num_workers:
            try:
                msg = result_queue.get(timeout=4200)  # 1h max d'attente
            except Empty:
                print(f"[Main] Plus de nouvelles depuis 1h → forçage arrêt")
                break

            if msg == "WORKER_FINISHED":
                finished_workers += 1
                continue

            score, config = msg
            results.append((score, config))
            pbar.update(1)

            if score > best_score:
                best_score = score
                best_config = config.copy()
                print(f"\nNEW BEST → {score:.6f} (+{score - best_score_prev:+.6f})")
                best_score_prev = score

                # SAUVEGARDE IMMÉDIATE du nouveau meilleur
                with open("models/simple_opt/best_live.json", "w") as f:
                    json.dump(best_config, f, indent=2)

    # Nettoyage agressif
    print("Arrêt des workers...")
    for w in workers:
        if w.is_alive():
            w.terminate()
            w.join(timeout=10)
            if w.is_alive():
                w.kill()
                w.join()

    # ==================================================================
    # SAUVEGARDE
    # ==================================================================
    if best_config:
        os.makedirs("models/simple_opt", exist_ok=True)
        with open("models/simple_opt/best_pierre2026.json", "w") as f:
            json.dump(best_config, f, indent=2)

        print("\n" + "="*60)
        print("MEILLEURE CONFIG TROUVÉE:")
        print(json.dumps(best_config, indent=2))
        print(f"Score final: {best_score:.4f}")
    else:
        print("\n" + "="*60)
        print("AUCUNE CONFIGURATION VALIDE N'A ÉTÉ TROUVÉE.")

    print("\nTOP 5 CONFIGS:")
    top5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
    for top in top5:
        print(top)
    return best_config, results


# ==================================================================
# LANCEMENT (indispensable sous Windows et recommandé partout)
# ==================================================================
if __name__ == "__main__":
    # Set the start method inside the main guard to prevent re-initialization
    mp.set_start_method('fork', force=True)
    best_config, all_results = run_grid_search_multiprocess()
