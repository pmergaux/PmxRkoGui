import numpy as np
from collections import deque

SIZE = 2000

class TickMonitor:
    def __init__(self):
        # Collections pour la conservation des 1000 derniers ask et bid bruts
        self.asks = deque(maxlen=SIZE)
        self.bids = deque(maxlen=SIZE)

    def update(self, ask: float, bid: float):
        """Ajoute un nouveau tick et met à jour les collections."""
        self.asks.append(ask)
        self.bids.append(bid)

    def get_collections_1000(self):
        """Extrait les tableaux numpy pour les fenêtres de 1000."""
        if len(self.asks) < 2:
            return None, None, None

        arr_ask = np.array(self.asks)
        arr_bid = np.array(self.bids)

        # ask[n-1] - ask[n] (Attention à l'ordre, ici variation par rapport au précédent)
        # Utilisation de np.diff : arr[1:] - arr[:-1] ou l'inverse selon la convention désirée
        diff_ask = np.diff(arr_ask)  # ask[n] - ask[n-1]
        diff_bid = np.diff(arr_bid)  # bid[n] - bid[n-1]
        spread_1000 = arr_ask - arr_bid  # ask[n] - bid[n]

        return diff_ask[-SIZE+1:], diff_bid[-SIZE+1:], spread_1000[-SIZE:]

    def get_collections_30(self):
        """Extrait les tableaux numpy pour les fenêtres de 30 (issues des 1000 communes)."""
        if len(self.asks) < 31:
            return None, None, None

        diff_ask_1000, diff_bid_1000, spread_1000 = self.get_collections_1000()

        if diff_ask_1000 is None or len(diff_ask_1000) < 30:
            return None, None, None

        # Extraction des 30 dernières valeurs à partir des collections 1000 communes
        diff_ask_30 = diff_ask_1000[-30:]
        diff_bid_30 = diff_bid_1000[-30:]
        spread_30 = spread_1000[-30:]

        return diff_ask_30, diff_bid_30, spread_30

    @staticmethod
    def stats(data):
        """Calcule la moyenne et l'écart-type d'une collection."""
        if data is None or len(data) == 0:
            return 0.0, 0.0
        return float(np.mean(data)), float(np.std(data))

    def get_indicators(self):
        """Fournit un résumé statistique prêt à l'emploi pour la prise de décision."""
        d_ask_1000, d_bid_1000, spread_1000 = self.get_collections_1000()
        if d_ask_1000 is None or len(d_ask_1000) < 30:
            return None
        d_ask_30, d_bid_30, spread_30 = d_ask_1000[-30:], d_bid_1000[-30:], spread_1000[-30:]

        return {
            "ask_diff_1000": self.stats(d_ask_1000),
            "ask_diff_30": self.stats(d_ask_30),
            "bid_diff_1000": self.stats(d_bid_1000),
            "bid_diff_30": self.stats(d_bid_30),
            "spread_1000": self.stats(spread_1000),
            "spread_30": self.stats(spread_30),
        }


import time
from datetime import datetime, timedelta

SELL, NONE, BUY = -1, 0, 1
SENS, OPEN_TIME, CLOSE_TIME, OPEN_PRICE, CURRENT_PRICE,  PROFIT = range(6)

# Importation fictive ou réelle selon votre installation de mt5linux
from mt5linux import MetaTrader5

def run_test_program():
    # --- 1. Initialisation mt5linux ---
    mt5 = MetaTrader5()
    if not mt5.initialize():
        print("Erreur initialisation MT5")
        return

    symbol = "ETHUSD"

    # Récupération de 60 jours de ticks
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=30)

    print(f"Récupération des ticks pour {symbol} du {utc_from} au {utc_to}...")
    ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    mt5.shutdown()
    # print(f"len ticks : {len(ticks)}, {ticks[:5]}")
    positions = []
    position = [NONE, 0, 0, 0.0,  0.0, 0.0]
    # Simulation de ticks pour l'exemple si mt5linux n'est pas instancié dans l'environnement immédiat
    # ticks = []  # Remplacer par vos données réelles issues de mt5linux

    monitor = TickMonitor()

    # --- 2. Initialisation du monitor avec les 1000 premiers ticks ---
    print(f"Initialisation du TickMonitor avec les {SIZE} premiers ticks...")
    for tick in ticks[:SIZE]:
        monitor.update(ask=tick['ask'], bid=tick['bid'])

    # Position actuelle simulée : 0 (aucune), 1 (achat), -1 (vente)
    current_position = 0

    # --- 3. Boucle de Trading / Test ---
    # pour commencer affichage de 10 valeurs
    def affichage():
        # Multiplicateur pour une meilleure lisibilité (conversion en micro-unités / pips)
        timestamp_sec = tick['time_msc'] / 1000.0
        dt_readable = datetime.fromtimestamp(timestamp_sec)
        print(f"time : {dt_readable} bid {tick['bid']} ask {tick['ask']}")
        mult = 1_000
        # Application du formatage .3f avec multiplication
        print(f"ask   - Moy(30/1000): {ask_moyenne_30 * mult:.3f} / {ask_moyenne_1000 * mult:.3f} "
              f"| Std(30/1000): {ask_std_30 * mult:.3f} / {ask_std_1000 * mult:.3f} "
              f"| ask_Z_30 {(ask_moyenne_30 - ask_moyenne_1000)/ask_std_1000:.3f}")
        print(f"bid   - Moy(30/1000): {bid_moyenne_30 * mult:.3f} / {bid_moyenne_1000 * mult:.3f} "
              f"| Std(30/1000): {bid_std_30 * mult:.3f} / {bid_std_1000 * mult:.3f} "
              f"| bid_Z_30 : {(bid_moyenne_30 - bid_moyenne_1000)/bid_std_1000:.3f}")
        print(f"spread- Moy(30/1000): {spread_moyenne_30:.3f} / {spread_moyenne_1000:.3f} "
              f"| Std(30/1000): {spread_std_30 * mult:.3f} / {spread_std_1000 * mult:.3f} "
              f"| spread_Z_30 : {((spread_moyenne_30 - spread_moyenne_1000)/spread_std_1000):.3f}")
        print("-" * 60)

    nombre = 0
    # Simulation de lecture du flux des ticks restants ou en temps réel
    for tick in ticks[SIZE:]:
        ask = tick['ask']
        bid = tick['bid']
        monitor.update(ask, bid)
        ind = monitor.get_indicators()
        if ind is None:
            continue

        # Exemple de structure pour les règles de décision :
        (ask_moyenne_30, ask_std_30) = ind.get('ask_diff_30', (None, None))
        (ask_moyenne_1000, ask_std_1000) = ind.get('ask_diff_1000', (None, None))
        (bid_moyenne_30, bid_std_30) = ind.get('bid_diff_30', (None, None))
        (bid_moyenne_1000, bid_std_1000) = ind.get('bid_diff_1000', (None, None))
        (spread_moyenne_30, spread_std_30) = ind.get('spread_30', (None, None))
        (spread_moyenne_1000, spread_std_1000) = ind.get('spread_1000', (None, None))

        nombre += 1
        if nombre <= 10:
            affichage()
        spread_Z_30 = (spread_moyenne_30 - spread_moyenne_1000) / spread_std_1000
        ls = BUY if spread_Z_30 > 1 else SELL if spread_Z_30 < -1 else NONE
        if position[SENS] == NONE:
            if ls == BUY:
                position[SENS] = ls
                position[OPEN_TIME] = tick['time_msc']
                position[OPEN_PRICE] = ask
            elif ls == SELL :
                position[SENS] = ls
                position[OPEN_TIME] = tick['time_msc']
                position[OPEN_PRICE] = bid
        else:
            if position[SENS] == BUY:
                position[CURRENT_PRICE] = bid
            else:
                position[CURRENT_PRICE] = ask
            if ls != position[SENS]:
                position[CLOSE_TIME] = tick['time_msc']
                position[PROFIT] = (position[CURRENT_PRICE] - position[OPEN_PRICE]) * position[SENS]
                positions.append(position)
                position = [NONE, 0, 0, 0.0, 0.0, 0.0]

        # ask_std_30 < ask_std_1000 (Stabilité relative du court terme par rapport au long terme)
        # ask_mean_30 encadrée par un multiple de son écart-type (ex: |moyenne| < K * std)
        # bid_30 volatile (ex: std_30 > Seuil de volatilité)

        # --- RÈGLE ACHAT ---
        # cond_ask_stable = ind['ask_diff_30'][1] < ind['ask_diff_1000'][1]
        # cond_ask_mean_bounded = abs(ind['ask_diff_30'][0]) < (2.0 * ind['ask_diff_30'][1])
        # cond_bid_volatile = ind['bid_diff_30'][1] > Seuil_Volatilite

        # if cond_ask_stable and cond_ask_mean_bounded and cond_bid_volatile:
        #     if current_position == -1:
        #         print("Clôture position VENTE")
        #         # Code MT5 pour fermer vente
        #     if current_position != 1:
        #         print("Ouverture position ACHAT")
        #         # Code MT5 pour acheter
        #         current_position = 1

        # --- RÈGLE VENTE (Inverse) ---
        # cond_bid_stable = ind['bid_diff_30'][1] < ind['bid_diff_1000'][1]
        # cond_bid_mean_bounded = abs(ind['bid_diff_30'][0]) < (2.0 * ind['bid_diff_30'][1])
        # cond_ask_volatile = ind['ask_diff_30'][1] > Seuil_Volatilite

        # if cond_bid_stable and cond_bid_mean_bounded and cond_ask_volatile:
        #     if current_position == 1:
        #         print("Clôture position ACHAT")
        #         # Code MT5 pour fermer achat
        #     if current_position != -1:
        #         print("Ouverture position VENTE")
        #         # Code MT5 pour vendre
        #         current_position = -1
    if position[SENS] != NONE:
        position[PROFIT] = (position[CURRENT_PRICE] - position[OPEN_PRICE]) * position[SENS]
        positions.append(position)
    total_profit = np.sum(positions[PROFIT])
    num_trades = len(positions)
    print(f"total profit : {total_profit} pour {num_trades} trades")

if __name__ == "__main__":
    run_test_program()
