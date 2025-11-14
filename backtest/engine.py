# backtest/engine.py
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import itertools

class OptimizationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, data, param_grid):
        super().__init__()
        self.data = data
        self.param_grid = param_grid
        self.best_result = None

    def run(self):
        valid_grid = {k: [v for v in vs if v > 0] for k, vs in self.param_grid.items()}
        total = len(list(itertools.product(*valid_grid.values())))
        if total == 0:
            self.finished.emit({})
            return

        current = 0
        for brick_size in valid_grid['brick_size']:
            for ema_p in valid_grid.get('ema_period', [9]):
                equity = self.backtest(brick_size, ema_p)
                if equity > (self.best_result or {}).get('equity', 0):
                    self.best_result = {'brick_size': brick_size, 'ema_period': ema_p, 'equity': equity}
                current += 1
                self.progress.emit(int(current / total * 100))
        self.finished.emit(self.best_result or {})

    def backtest(self, brick_size, ema_p):
        closes = self.data['close'].tolist()
        renko = self.build_renko(closes, brick_size)
        if len(renko) < 50: return 10000

        equity = 10000.0
        position = 0
        entry = 0
        for i in range(50, len(renko)):
            if position == 0 and renko[i]['up'] and not renko[i-1]['up']:
                position = 1
                entry = renko[i]['close']
            elif position == 1 and not renko[i]['up'] and renko[i-1]['up']:
                equity *= (renko[i]['close'] / entry)
                position = 0
        if position == 1:
            equity *= (closes[-1] / entry)
        return round(equity, 2)

    def build_renko(self, prices, brick_size):
        if brick_size <= 0: return []
        bricks = []
        p0 = prices[0]
        open_p = round(p0 - (p0 % brick_size), 5)
        for p in prices[1:]:
            diff = p - open_p
            n = int(abs(diff) // brick_size)
            if n == 0: continue
            is_up = diff > 0
            for _ in range(n):
                close_p = round(open_p + brick_size if is_up else open_p - brick_size, 5)
                bricks.append({'open': open_p, 'close': close_p, 'up': is_up})
                open_p = close_p
        return bricks
