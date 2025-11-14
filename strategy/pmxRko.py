# strategy/pmxrko.py
from .base import Strategy
from utils.renko_utils import tick2renko
from decision.candle_decision import calculate_indicators, choix_features
from utils.utils import NONE, BUY, SELL, CLOSE
from datetime import datetime

class PmxRkoStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.bricks = None
        self.renko_size = config.get("renko_size", 0.00010)
        self.ema_period = config.get("ema_period", 9)
        self.rsi_period = config.get("rsi_period", 14)
        self.macd = config.get("macd", [12, 26, 9])
        self.th_high = config.get("ia_threshold_high", 0.6)
        self.th_low = config.get("ia_threshold_low", 0.4)
        self.max_time = config.get("max_position_time", 1800)
        self.max_bricks = config.get("max_position_bricks", 20)
        self.trailing = config.get("trailing_bricks", 1.5)
        self.tp = config.get("take_profit_bricks", 3.0)
        self.sl_percent = config.get("sl_percent", 1.0)
        self.time_vol = config.get("time_vol", False)
        self.minimum = max(self.macd[0] + self.macd[2], 20, self.rsi_period) + 1
        self.renko_time = None
        self.count_time = None

    def generate_signal(self, renko, bid, ema, rsi, macd_line, macd_signal):
        if len(renko) < 3: return 0
        last, prev = renko[-1], renko[-2]

        # === ACHAT ===
        if (last['up'] and not prev['up'] and
            ema[-1] > ema[-2] and
            rsi[-1] < 70 and
            macd_line[-1] > macd_signal[-1]):
            return 1

        # === VENTE ===
        if (not last['up'] and prev['up'] and
            ema[-1] < ema[-2] and
            rsi[-1] > 30 and
            macd_line[-1] < macd_signal[-1]):
            return -1

        return 0

    def decision(self):
        df = calculate_indicators(self.bricks, self._param)
        choix_features(df, self._param)
        row = self._param.get('rang_decision', -2)
        return df['direction'].iloc[row], df['sigc'].iloc[row], df['sigo'].iloc[row]

    def run(self, symbol):
        super().run(symbol)
        if self.df is None or len(self.df) == 0:
            return

        if self.bricks is None:
            self.bricks = tick2renko(self.ticks, None, self.renko_size, 'bid')
            if len(self.bricks) < self.minimum:
                return
            self.renko_time = self.bricks.index[-1]
            print(f"{datetime.now()} [STRAT] Renko initialisé")
            return
        else:
            self.bricks = tick2renko(self.ticks, self.bricks, step=self.renko_size, value='bid')
        if len(self.bricks) < self.minimum:
            return
        if len(self.bricks) > self.minimum + 9:
            self.bricks = self.bricks[-self.minimum:]
        if self.renko_time == self.bricks.index[-1]:
            return
        self.renko_time = self.bricks.index[-1]
        print(f"{datetime.now()} [STRAT] Nouvelle brique")

        dc, sc, so = self.decision()
        if self.ssens != NONE and sc == CLOSE:
            print(f"{datetime.now()} [STRAT] CLÔTURE")
            self.ssens = NONE
        elif self.ssens == NONE and so != NONE:
            print(f"{datetime.now()} [STRAT] OUVERTURE {'BUY' if so == BUY else 'SELL'}")
            self.ssens = so