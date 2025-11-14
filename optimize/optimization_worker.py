# optimize/optimization_worker.py
from PyQt6.QtCore import QThread, pyqtSignal
import time
from multiprocessing import Queue

class OptimizationWorker(QThread):
    progress = pyqtSignal(int)      # % avancement
    status = pyqtSignal(str)        # message
    finished = pyqtSignal(list)     # grille finale
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, grid, parent=None, ticks_path=""):
        super().__init__(parent)
        self.grid = grid
        self._stop = False
        self.ticks_path = ticks_path
        self.queue = Queue()  # INITIALISÉ ICI

    def run(self):
        # === voir ici quelle optimisation placée lstm, bayesienne optuna, ray_tune ...
        from optimize.lstm_optimizer import run_optimization
        run_optimization(self.grid, self.ticks_path, self.queue)  # CORRECT

        # === LIRE LA QUEUE ===
        while True:
            try:
                msg = self.queue.get_nowait()
                if msg['type'] == 'progress':
                    self.progress.emit(msg['value'])
                elif msg['type'] == 'log':
                    self.log.emit(msg['text'])
                elif msg['type'] == 'result':
                    self.finished.emit(msg['data'])
                    break
            except:
                break

    # --- SIMULATION ====================================
    def run_simulate(self):
        total = len(self.grid)
        results = []
        self.status.emit("Démarrage de l'optimisation...")
        time.sleep(0.5)
        for i, params in enumerate(self.grid):
            if self._stop:
                self.status.emit("Arrêté par l'utilisateur")
                return
            # === SIMULATION D'UN BACKTEST (remplace par ton vrai moteur) ===
            time.sleep(0.05)  # 20 backtests/seconde
            profit = self._simulate_backtest(params)
            results.append({"params": params, "profit": profit})
            # Mise à jour UI
            percent = int((i + 1) / total * 100)
            self.progress.emit(percent)
            self.status.emit(f"Backtest {i+1}/{total} → Profit: {profit:.2f}%")
        self.status.emit(f"Optimisation terminée : {len(results)} runs")
        self.finished.emit(results)

    def _simulate_backtest(self, params):
        # === EXEMPLE DE SCORE (à remplacer par ton LSTM + Renko + MT5) ===
        import random
        base = 50
        for v in params.values():
            base += (v - 20) * 0.1 if isinstance(v, (int, float)) else 0
        noise = random.gauss(0, 15)
        return max(-100, min(300, base + noise))

    def stop(self):
        self._stop = True

