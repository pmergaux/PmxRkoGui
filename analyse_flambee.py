import os
import pandas as pd
import numpy as np
from utils.renko_utils import tick21renko

# Afficher toutes les colonnes (pas de troncature au milieu)
pd.set_option('display.max_columns', None)

# Définir une largeur maximale d'affichage très large
pd.set_option('display.width', 1000)


def analyser_flambee_renko(csv_path, renko_size=21.0, output_csv="renko_flambee_19_20_août.csv"):
    """
    Extrait les ticks du 19 et 20 août 2026, génère les briques Renko,
    et calcule les colonnes duration, direction et variation.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Fichier introuvable : {csv_path}")
        return

    print(f"📂 Chargement des ticks depuis {csv_path}...")
    # Chargement du CSV
    df = pd.read_csv(csv_path, sep=";")

    if 'time_msc' not in df.columns and df.index.name == 'time_msc':
        df.reset_index(inplace=True)

    # Conversion du temps millisecondes en datetime pour le filtrage
    df['datetime'] = pd.to_datetime(df['time_msc'], unit='ms')

    # Filtrer strictement pour les 19 et 20 août 2026
    print("🔍 Filtrage des dates du 19 et 20 août 2026...")
    mask = (df['datetime'] >= '2026-08-19 00:00:00') & (df['datetime'] < '2026-08-21 00:00:00')
    df_filtered = df.loc[mask].copy()

    if len(df_filtered) == 0:
        print("⚠️ Aucun tick trouvé pour cette période exacte dans le fichier.")
        return

    df_filtered.set_index('datetime', inplace=True)
    print(f"✅ {len(df_filtered)} ticks retenus pour l'analyse.")

    print(f"🧱 Génération des briques Renko (taille = {renko_size}$) sur la période...")
    df_bricks = tick21renko(df_filtered, None, step=renko_size, value='bid')

    if len(df_bricks) < 2:
        print("❌ Pas assez de briques générées sur ces deux jours.")
        return

    # =========================================================
    # CALCUL DES COLONNES DEMANDÉES
    # =========================================================
    print("⚙️ Calcul de la durée, de la direction et de la variation...")

    # Conversion des timestamps de la colonne 'time' (ou de l'index) en millisecondes numériques
    times_ms = df_bricks['time'].to_numpy().astype('int64') / 1_000_000

    # Écarts entre la brique suivante et la courante (diff inverse)
    # np.diff donne time[i+1] - time[i] pour chaque indice i
    durations_sec = np.diff(times_ms) / 1000.0

    # On assigne la durée à la brique i, et NaN pour la toute dernière brique en cours
    df_bricks['duration'] = np.append(durations_sec, np.nan)

    # Direction : 1 si close_renko > open_renko, sinon -1
    df_bricks['direction'] = np.where(df_bricks['close_renko'] > df_bricks['open_renko'], 1, -1)

    # Variation basée sur les vraies valeurs open et close (bid)
    df_bricks['variation'] = df_bricks['close'] - df_bricks['open']

    # Réinitialisation de l'index pour un export CSV propre
    df_bricks_export = df_bricks.reset_index(drop=True)

    # Sauvegarde en CSV
    df_bricks_export.to_csv(output_csv, index=False)
    print(f"✨ Fichier exporté avec succès : {output_csv} ({len(df_bricks_export)} briques)")

    # Aperçu rapide des briques les plus rapides
    print("\n📊 Top 5 des briques les plus rapides sur la période :")
    print(df_bricks_export.sort_values(by='duration').head(5)[
              ['time', 'duration', 'direction', 'open', 'close', 'open_renko', 'close_renko', 'variation']])

if __name__ == "__main__":
    CSV_SOURCE = "/media/pierre/datad/data/ETHUSD_120.csv"
    analyser_flambee_renko(CSV_SOURCE, renko_size=21.0)
