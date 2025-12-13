import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime

# 1. Définir le dossier de cache où seront stockés les fichiers
from optimize.multi_optim import get_ticks_dataframe, prepare_ticks_once, RENKO_CACHE_DIR
from utils.renko_utils import tick21renko

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
    if os.path.exists(file_path):
        print(f"Cache HIT: Le fichier pour renko_size={renko_size} existe déjà. Skip.")
        return

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
    # prepare les ticks
    prepare_ticks_once()
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
