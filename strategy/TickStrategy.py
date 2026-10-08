import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from mt5linux import MetaTrader5

# ==========================================
# CONFIGURATION DE LA STRATÉGIE ET DES DONNÉES
# ==========================================
VERSION = "SPREAD"  # Options possibles : "SPREAD", "MID", "PRESSURE"
CSV_PATH = "/media/pierre/datad/data/ticks_ETHUSD_60.csv"
SYMBOL = "ETHUSD"

LSIZE = 2000
CSIZE = 100
SELL, NONE, BUY = -1, 0, 1
SENS, OPEN_TIME, CLOSE_TIME, OPEN_PRICE, CURRENT_PRICE, PROFIT = range(6)
ASK, BID, TIME = range(3)

def charger_ou_telecharger_ticks():
    dossier_csv = os.path.dirname(CSV_PATH)
    if dossier_csv and not os.path.exists(dossier_csv):
        os.makedirs(dossier_csv, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"Chargement optimisé des ticks depuis : {CSV_PATH}...")
        # Chargement en forçant les types en float32 pour économiser massivement la RAM
        # On ne charge que les colonnes utiles si le CSV en contient d'autres
        df = pd.read_csv(CSV_PATH, dtype={'ask': 'float32', 'bid': 'float32', 'time_msc': 'int64'})
        print(f"{len(df)} ticks chargés avec succès.")
        return df

    print("Fichier local introuvable. Connexion à MT5...")
    mt5 = MetaTrader5()
    if not mt5.initialize():
        print("Erreur initialisation MT5")
        return None

    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=60)

    print(f"Récupération des ticks pour {SYMBOL}...")
    ticks_raw = mt5.copy_ticks_range(SYMBOL, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    mt5.shutdown()

    if ticks_raw is None or len(ticks_raw) == 0:
        print("Aucun tick récupéré.")
        return None

    print(f"Sauvegarde de {len(ticks_raw)} ticks dans {CSV_PATH}...")
    df = pd.DataFrame(ticks_raw)
    df.to_csv(CSV_PATH, index=False)
    return df

def run_test_program():
    df_ticks = charger_ou_telecharger_ticks()
    if df_ticks is None or len(df_ticks) < LSIZE:
        print("Pas assez de ticks.")
        return

    print(f"Traitement de {len(df_ticks)} ticks avec la VERSION : {VERSION}...")

    # Extraction directe des colonnes sous forme de tableaux NumPy (ultra rapide)
    all_asks = df_ticks['ask'].to_numpy(dtype=np.float32)
    all_bids = df_ticks['bid'].to_numpy(dtype=np.float32)
    # all_times = df_ticks['time_msc'].to_numpy()

    positions = []
    position = [NONE, 0, 0, 0.0, 0.0, 0.0]
    nombre = 0

    for i in range(LSIZE, len(df_ticks)):
        window_asks = all_asks[i - LSIZE: i]
        window_bids = all_bids[i - LSIZE: i]

        # Récupération directe des valeurs du tick actuel via le DataFrame
        current_tick = df_ticks.iloc[i]
        ask = current_tick['ask']
        bid = current_tick['bid']
        time_msc = current_tick['time_msc']

        # =========================================================
        # BRANCHEMENT SELON LA VERSION CHOISIE
        # =========================================================
        ls = NONE
        lc = NONE

        if VERSION == "SPREAD":
            spread_1000 = window_asks - window_bids
            spread_30 = spread_1000[-CSIZE:]

            spread_moyenne_30 = float(np.mean(spread_30))
            spread_moyenne_1000 = float(np.mean(spread_1000))
            spread_std_1000 = float(np.std(spread_1000))

            spread_Z_30 = (spread_moyenne_30 - spread_moyenne_1000) / spread_std_1000 if spread_std_1000 != 0 else 0
            ls = BUY if spread_Z_30 > 1.96 else SELL if spread_Z_30 < -1.96 else NONE
            lc = BUY if spread_Z_30 > 1.0 else SELL if spread_Z_30 < -1.0 else NONE

        elif VERSION == "MID":
            window_mids = (window_asks + window_bids) / 2.0

            diff_mid_1000 = np.diff(window_mids)
            diff_mid_30 = diff_mid_1000[-CSIZE:]

            mid_moyenne_30 = float(np.mean(diff_mid_30))
            mid_moyenne_1000 = float(np.mean(diff_mid_1000))
            mid_std_1000 = float(np.std(diff_mid_1000))

            mid_Z_30 = (mid_moyenne_30 - mid_moyenne_1000) / mid_std_1000 if mid_std_1000 != 0 else 0
            ls = BUY if mid_Z_30 > 1.96 else SELL if mid_Z_30 < -1.96 else NONE
            lc = BUY if mid_Z_30 > 1.0 else SELL if mid_Z_30 < -1.0 else NONE

        elif VERSION == "PRESSURE":
            spread_1000 = window_asks - window_bids
            spread_30 = spread_1000[-CSIZE:]

            spread_moyenne_30 = float(np.mean(spread_30))
            spread_moyenne_1000 = float(np.mean(spread_1000))
            spread_std_1000 = float(np.std(spread_1000))
            spread_Z_30 = (spread_moyenne_30 - spread_moyenne_1000) / spread_std_1000 if spread_std_1000 != 0 else 0

            diff_ask = np.diff(window_asks[-CSIZE:])
            diff_bid = np.diff(window_bids[-CSIZE:])

            ask_up = np.sum(diff_ask >= 0)
            bid_up = np.sum(diff_bid > 0)

            pressure_diff = ask_up - bid_up
            if pressure_diff > int(CSIZE/6) and spread_Z_30 > 0:
                ls = BUY
            elif pressure_diff < -int(CSIZE/6) and spread_Z_30 < 0:
                ls = SELL
            lc = ls

        # =========================================================
        # GESTION DES POSITIONS
        # =========================================================
        if position[SENS] == NONE:
            if ls == BUY:
                position[SENS] = ls
                position[OPEN_TIME] = time_msc
                position[OPEN_PRICE] = ask
            elif ls == SELL:
                position[SENS] = ls
                position[OPEN_TIME] = time_msc
                position[OPEN_PRICE] = bid
        else:
            if position[SENS] == BUY:
                position[CURRENT_PRICE] = bid
            else:
                position[CURRENT_PRICE] = ask
            if lc != position[SENS]:
                position[CLOSE_TIME] = time_msc
                position[PROFIT] = (position[CURRENT_PRICE] - position[OPEN_PRICE]) * position[SENS]
                positions.append(list(position))
                position = [NONE, 0, 0, 0.0, 0.0, 0.0]

    if position[SENS] != NONE:
        position[PROFIT] = (position[CURRENT_PRICE] - position[OPEN_PRICE]) * position[SENS]
        positions.append(list(position))

    total_profit = sum(p[PROFIT] for p in positions) if positions else 0.0
    num_trades = len(positions)
    print(f"[{VERSION}] Total profit : {total_profit:.5f} pour {num_trades} trades")

if __name__ == "__main__":
    run_test_program()
