# optimization_worker.py
import os

from PyQt6.QtCore import QThread, pyqtSignal
import multiprocessing as mp
from optimize.lstm_optimizer import run_optimization, run_optimization_in_process
import time

# 2. Limite les threads OpenMP/oneDNN (sinon explosion)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# 3. Désactive les warnings chiants
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptimizationWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)  # ← TOUT le résultat final
    error = pyqtSignal(str)
    status = pyqtSignal(str)        # message

    def __init__(self, param_grid, config, ticks_data, parent):
        super().__init__()
        self.param_grid = param_grid
        self.config = config
        self.ticks_data = ticks_data
        self._stop = False
        self.parent = parent

        # CRUCIAL : Forcer le spawn (obligatoire avec TensorFlow + QThread)
        # dans main.py   mp.set_start_method('spawn', force=True)

    def run(self):
        try:
            # On passe QUE les données brutes au Process
            queue = mp.Queue()
            process = mp.Process(
                target=run_optimization_in_process,
                args=(self.param_grid, self.config, self.ticks_data, queue)
            )
            if hasattr(self.parent, "opt_status"):
                self.parent.opt_status.setText("Démarrage optimisation...")
            process.start()

            while process.is_alive() or not queue.empty():
                try:
                    msg = queue.get(timeout=0.1)
                    if msg['type'] == 'log':
                        self.log.emit(msg['text'])
                    elif msg['type'] == 'progress':
                        self.progress.emit(msg['value'])
                    elif msg['type'] == 'result':
                        self.finished.emit(msg['data'])
                    elif msg['type'] == 'done':
                        break
                except:
                    continue

            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                self.error_signal.emit("Processus bloqué → tué")

        except Exception as e:
            self.error_signal.emit(f"ERREUR CRITIQUE : {e}")

    def run_nopickable(self):
        try:
            # Queue interne pour communiquer avec les processus fils
            queue = mp.Queue()
            # On lance l'optimisation dans un Process séparé
            process = mp.Process(
                target=self._run_optimization_in_process,
                args=(self.param_grid, self.config, self.ticks_data, queue)
            )
            process.start()

            # Boucle de lecture de la queue (non bloquante pour le thread)
            while True:
                try:
                    msg = queue.get(timeout=0.1)
                    if msg['type'] == 'log':
                        self.log_signal.emit(msg['text'])
                    elif msg['type'] == 'progress':
                        self.progress_signal.emit(msg['value'])
                    elif msg['type'] == 'result':
                        self.finished_signal.emit(msg['data'])
                    elif msg['type'] == 'done':
                        break
                except:
                    if not process.is_alive():
                        break

            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.kill()

        except Exception as e:
            self.error_signal.emit(f"CRASH OPTIMISATION : {e}")

    def _run_optimization_in_process(self, param_grid, config, ticks_data, queue):
        # Cette fonction tourne dans un PROCESS séparé → peut utiliser Keras sans peur
        def send_log(msg):
            queue.put({'type': 'log', 'text': msg})

        def send_progress(pct):
            queue.put({'type': 'progress', 'value': pct})

        send_log("Démarrage optimisation dans processus dédié...")
        result = run_optimization(param_grid, config, ticks_data, queue)
        queue.put({'type': 'result', 'data': result})
        queue.put({'type': 'done', 'data': None})

        import tensorflow as tf
        # Dans _run_optimization_in_process, à la fin :
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

    # --- ancienne version ==============================
    def run_old(self):
        queue = mp.Queue()
        # === voir ici quelle optimisation placée lstm, bayesienne optuna, ray_tune ...
        run_optimization(self.param_grid, self.config, self.ticks_data, queue)  # CORRECT

        # === LIRE LA QUEUE ===
        while True:
            try:
                msg = queue.get_nowait()
                if msg['type'] == 'progress':
                    self.progress_signal.emit(msg['value'])
                elif msg['type'] == 'log':
                    self.log_signal.emit(msg['text'])
                elif msg['type'] == 'result':
                    self.finished_signal.emit(msg['data'])
                    break
            except:
                break

    # --- SIMULATION ====================================
    def run_simulate(self):
        total = len(self.grid)
        results = []
        self.status_signal.emit("Démarrage de l'optimisation...")
        time.sleep(0.5)
        for i, params in enumerate(self.grid):
            if self._stop:
                self.status_signal.emit("Arrêté par l'utilisateur")
                return
            # === SIMULATION D'UN BACKTEST (remplace par ton vrai moteur) ===
            time.sleep(0.05)  # 20 backtests/seconde
            profit = self._simulate_backtest(params)
            results.append({"params": params, "profit": profit})
            # Mise à jour UI
            percent = int((i + 1) / total * 100)
            self.progress_signal.emit(percent)
            self.status_signal.emit(f"Backtest {i+1}/{total} → Profit: {profit:.2f}%")
        self.status_signal.emit(f"Optimisation terminée : {len(results)} runs")
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

