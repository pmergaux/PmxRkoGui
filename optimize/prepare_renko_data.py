import gc
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 1. Définir le dossier de cache où seront stockés les fichiers
from optimize.optimize_optuna import optimize_start, RENKO_CACHE_DIR
from utils.renko_utils import tick21renko

# ======================================================================
# VARIABLES GLOBALES
# ======================================================================
# Chemin fixe du fichier bn/,-* optimisé (on le crée une seule fois)
start_date = None
end_date = None
symbol = "ETHUSD"
TICKS_NPY_PATH = f"../data/{symbol}_ticks_optimized.npy"

def prepare_ticks_once():
    print("Préparation du fichier ticks optimisé (une seule fois)...")
    start = start_date
    end = end_date
    base_name = f"../data/{symbol}_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    """À lancer UNE FOIS quand tu changes de période)"""
    if os.path.exists(base_name):
        print("Fichier ticks optimisé déjà existant → skip")
        return False

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
    del df
    return True

def get_ticks_dataframe():
    """Chaque worker charge en 0.8 seconde depuis le .npy"""
    arr = np.load(TICKS_NPY_PATH, mmap_mode='r')  # ← mémoire mappée → quasi zéro RAM
    df = pd.DataFrame(arr)
    df = df.set_index('time', drop=False).sort_index()
    df.index = pd.to_datetime(df.index, unit="ms")
    return df


# ======================================================================
# SCRIPT DE PRÉ-CALCUL DES BOUGIES RENKO
# ======================================================================

# 2. Définir ici EXACTEMENT les mêmes tailles de renko que dans votre script d'optimisation
RENKO_SIZES_TO_PREPARE = np.arange(10.0, 40.0, 0.1)  # Assurez-vous que cette liste est à jour

def create_renko_file(renko_size):
    """
    Fonction exécutée par chaque worker :
    1. Charge les ticks bruts.
    2. Calcule les bougies Renko pour UNE taille.
    3. Sauvegarde le résultat dans un fichier .pkl dédié.
    """
    renko_size = round(renko_size, 2)
    file_path = os.path.join(RENKO_CACHE_DIR, f"renko_{renko_size:.2f}.pkl")

    # Si le fichier existe déjà, on ne fait rien pour gagner du temps
    """
    if os.path.exists(file_path):
        print(f"Cache HIT: Le fichier pour renko_size={renko_size} existe déjà. Skip.")
        return
    """
    print(f"Cache MISS: Création du fichier pour renko_size={renko_size}...")
    try:
        # On charge les ticks bruts
        df_ticks = get_ticks_dataframe()

        # On calcule les bougies Renko
        df_renko = tick21renko(df_ticks, None, renko_size, 'bid')

        # On sauvegarde en format pickle, très rapide à lire
        df_renko.to_pickle(file_path)
        print(f"SUCCÈS: Fichier créé pour renko_size={renko_size}")
    except Exception as e:
        print(f"ERREUR lors de la création pour renko_size={renko_size}: {e}")


if __name__ == "__main__":
    #global end_date, start_date
    end_date = datetime.now().replace(hour=23, minute=30, second=0, microsecond=0)
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=75)
    # prepare les ticks
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
    optimize_start()
