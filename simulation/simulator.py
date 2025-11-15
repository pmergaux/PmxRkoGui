# simulation/simulator.py
import time
from threading import Timer

from PyQt6.QtCore import QTimer, QThread
from decision.candle_decision import calculate_indicators, choix_features
from utils.renko_utils import tick2renko, colonnesRko, update_renko_bricks, tick21renko
from utils.utils import BUY, SELL, NONE
import pandas as pd
import pickle

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2500)

class RenkoSimulator(QThread):
    def __init__(self, parent, config):

        self.parent = parent
        self.config = config
        self.bricks = None
        self.df = None
        self.ticks = pd.DataFrame()
        self.timer = QTimer()
        self.timer.timeout.connect(self.run)
        self.tick_idx = 0
        self.load_data()
        self.ticks_load = 800000
        self.last_brick = None
        if self.config.get('features', None) is None:
            self.config['features'] = ['EMA', 'RSI', 'MACD_hist']

    def start(self):
        while self.tick_idx < len(self.all_ticks):
            self.run()

    def load_data(self):
        with open("data/ETHUSD.pkl", "rb") as f:
            self.all_ticks = pickle.load(f)
        self.tick_idx = 0
        print("simule start")

    def run(self):
        if self.all_ticks is None or self.tick_idx >= len(self.all_ticks) or self.ticks_load > len(self.all_ticks):
            self.timer.stop()
            return

        # --- Prendre x ticks ---
        new_ticks = self.all_ticks.iloc[self.tick_idx:self.tick_idx + self.ticks_load]
        self.tick_idx += self.ticks_load
        self.ticks_load = 1

        # --- Créer / Mettre à jour briques ---
        first = False
        if self.bricks is None:
            first = True
        self.bricks = tick21renko(new_ticks, self.bricks, self.config['renko_size'], 'bid')
        #if first:            print(self.bricks.tail(3))
        if self.bricks is None or len(self.bricks) < 80:
            return
        if len(self.bricks) > 100:
            self.bricks = self.bricks[-82:]
        if self.last_brick is None or self.last_brick != self.bricks.index[-1]:
            self.last_brick = self.bricks.index[-1]
            print(self.bricks.tail(3))
            self.df = calculate_indicators(self.bricks, self.config)
            self.df = choix_features(self.df, self.config)

        # --- Transmettre UNIQUEMENT ce qui est nécessaire ---
        display_data = {
            'df': self.df[-11:-1],
            'current_bid': new_ticks['bid'].iloc[-1]  # ← DERNIER, pas premier
        }
        self.parent.update_display(display_data)
        time.sleep(15)
        print('fin time')

"""
# lorsque l'on génère les tickd et non chargés depuis un .pkl
class RenkoSimulator:
    def __init__(self, chart):
        self.chart = chart  # ← Référence au LiveChart
        self.price = 1.08500
        self.bricks = None
        self.ticks = pd.DataFrame()
        self.brick_size = 0.00015
        self.minimum = 35

    def generate_tick(self):
        self.price += random.gauss(0, 0.00008)
        bid = self.price - random.uniform(0.00003, 0.00008)
        now = pd.Timestamp.now()
        tick = pd.DataFrame({'bid': [bid]}, index=[now])
        self.ticks = pd.concat([self.ticks, tick])
        return bid

    def run(self):
        # Appelé à chaque tick (par timer du main_window)
        self.generate_tick()
        new_bricks = self.update_renko()
        if new_bricks is not None:
            data = self.get_display_data()
            if data:
                self.chart.update_display(data)  # ← APPEL DIRECT

    def update_renko(self):
        if self.bricks is None:
            self.bricks = tick2renko(self.ticks, None, self.brick_size, 'bid')
            if len(self.bricks) < self.minimum:
                return None
            return self.bricks
        else:
            self.bricks = tick2renko(self.ticks, self.bricks, step=self.brick_size, value='bid')
        return self.bricks

    def get_display_data(self):
        if not self.bricks or len(self.bricks) < self.minimum:
            return None
        df = calculate_indicators(self.bricks.tail(self.minimum + 10), {})
        choix_features(df, {})
        bricks_list = self.bricks.tail(50).reset_index().to_dict('records')
        candles = []
        for i, row in enumerate(bricks_list):
            prev_close = bricks_list[i-1]['close'] if i > 0 else row['open']
            candles.append({
                'open': prev_close,
                'close': row['close']
            })
        return {
            'candles': candles,
            'bid': self.price - random.uniform(0, 0.00003),
            'indicators': {
                'ema9': df['ema9'].iloc[-1] if 'ema9' in df else None,
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df else None
            }
        }
"""


