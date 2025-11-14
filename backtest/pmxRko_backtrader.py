# backtest/pmxrko_backtrader.py
from strategies.pmxRko import PmxRkoStrategy
from BackTrader import BackTrader
from utils.renko_utils import tick2renko
from utils.utils import NONE, CLOSE

class PmxRkoBacktrader(PmxRkoStrategy, BackTrader):
    def __init__(self, param, data):
        BackTrader.__init__(self, param, data)
        PmxRkoStrategy.__init__(self, None, param)

    def next(self):
        try:
            # === RENKO ===
            if self.bricks is None:
                self.bricks = self.p['bricks']
            else:
                self.bricks = tick2renko(self.ti, self.bricks, step=self.renko_size, value='bid')
            if len(self.bricks) < self.minimum:
                return

            # === NOUVELLE BRIQUE ===
            if self.renko_time == self.bricks.index[-1]:
                return
            self.renko_time = self.bricks.index[-1]

            # === DÉCISION ===
            dc, sc, so = self.decision()

            # === GESTION POSITION ===
            if self.position is not None:
                if sc == CLOSE:
                    self.close_back(self.ssens, CLOSE, 'norm')
                    self.ssens = NONE
            else:
                if so != NONE:
                    self.open_back(so)
                    self.ssens = so

        except Exception as e:
            print(f"[BACKTEST] Erreur: {e}")
