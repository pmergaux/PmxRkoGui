import gc
import os
import multiprocessing as mp
import pickle

import numpy as np
from mt5linux import MetaTrader5
from datetime import datetime, timezone, timedelta
import pandas as pd
import time

from decision.candle_decision import calculate_japonais
# 1. Définir le dossier de cache où seront stockés les fichiers
from optimize.optimize_optuna import optimize_start, RENKO_CACHE_DIR
from utils.rate_utils import ticks2rates
from utils.renko_utils import tick21renko

path_fus="/home/pierre/.wine/drive_c/Program Files/Fusion Markets MetaTrader 5/terminal64.exe"
path_adm='/home/pierre/.wine/drive_c/Program Files/Admiral Markets MT5/terminal64.exe'
path_meta='/home/pierre/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"

mt5 = MetaTrader5(port=18812)  # Valeurs par défaut
if not mt5.initialize(path=path_meta, portable=True):
    raise Exception("Erreur d'initialisation MT5 (Linux) : " + str(mt5.last_error()))
else:
    account_info = mt5.account_info()
    print("name   = {}".format(account_info.name))
    print("login  = {}".format(account_info.login))
    print("server = {}".format(account_info.server))

# ======================================================================
# VARIABLES GLOBALES
# ======================================================================
df_ticks = pd.DataFrame()

def ajout_japonaises(df_ticks):
    # ajout d'une DT rates'
    df = ticks2rates(df_ticks, '1m', 'bid')
    df = calculate_japonais((df))
    filename = f"/media/pierre/datad/data/df_ETHUSD.pkl"
    with (open(filename, 'wb'))as file:
        pickle.dump(df, file)

def prepare_ticks_once():
    global df_ticks
    print("Préparation du fichier ticks optimisé (une seule fois)...")
    df_ticks = load_partial()
    df_ticks['time_msc'] = pd.to_datetime(df_ticks['time_msc'], unit="ms")
    df_ticks = df_ticks.set_index('time_msc', drop=False).sort_index()
    print(f"deb {df_ticks.index[0]} fin {df_ticks.index[-1]}")
    return True

def get_ticks_dataframe():
    return df_ticks

# ======================================================================
# SCRIPT DE PRÉ-CALCUL DES BOUGIES RENKO
# ======================================================================
# 2. Définir ici EXACTEMENT les mêmes tailles de renko que dans votre script d'optimisation
#RENKO_SIZES_TO_PREPARE = np.arange(6.0, 40.0, 0.1)  # Assurez-vous que cette liste est à jour
RENKO_SIZES_TO_PREPARE = np.arange(8.0, 28.0, 0.1)  # Assurez-vous que cette liste est à jour

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

def load_partial():
    filename = f"/media/pierre/datad/data/ETHUSD_150.csv"
    if os.path.exists(filename):
        df_150 = pd.read_csv(filename, sep=";")
        return df_150
    else:
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
        df_ticks = load_ticks_incremental_forward("ETHUSD", datetime.now() - timedelta(days=270), chunk_days=2)
        print("fund fini")
        mt5.shutdown()
        return df_ticks

if __name__ == "__main__":
    if prepare_ticks_once():
        # Crée le dossier de cache s'il n'existe pas
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
    # optimize_start()
