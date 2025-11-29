# optimization_worker_plus.py
from PyQt6.QtCore import QThread, pyqtSignal
import multiprocessing as mp
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tensorflow as tf

from utils.lstm_utils import build_transformer


class OptimizationWorkerPLUS(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, param_grid, config, ticks_data, save_dir="best_models"):
        super().__init__()
        self.param_grid = param_grid
        self.config = config
        self.ticks_data = ticks_data
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        mp.set_start_method('spawn', force=True)

    def run(self):
        # Queue interne pour communiquer avec les processus fils
        queue = mp.Queue()
        # On lance l'optimisation dans un Process séparé
        proc = mp.Process(target=self._optimize_with_saving, args=(queue,))
        proc.start()

        # Boucle de lecture de la queue (non bloquante pour le thread)
        while True:
            try:
                msg = queue.get(timeout=0.1)
                if msg['type'] == 'log': self.log.emit(msg['text'])
                elif msg['type'] == 'progress': self.progress.emit(msg['value'])
                elif msg['type'] == 'result': self.finished.emit(msg['data'])
                elif msg['type'] == 'done': break
            except:
                if not proc.is_alive(): break

        proc.join(timeout=10)
        if proc.is_alive(): proc.terminate()

    def _optimize_with_saving(self, queue):
        # Cette fonction tourne dans un PROCESS séparé → peut utiliser Keras sans peur
        def log(msg): queue.put({'type': 'log', 'text': msg})
        def prog(p): queue.put({'type': 'progress', 'value': p})

        from optimize.lstm_optimizer import run_optimization
        from core.model_manager import ModelManager
        manager = ModelManager(self.save_dir)

        log("Sauvegarde des meilleurs modèles en cours...")
        # On récupère TOUS les résultats
        result = run_optimization(self.param_grid, self.config, self.ticks_data, queue)

        # Dans _run_optimization_in_process, à la fin :
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

        if 'best' not in result:
            log("ÉCHEC TOTAL")
            queue.put({'type': 'done', 'data': None})
            return

        # === SAUVEGARDE DES TOP 5 MODÈLES ===
        top_models = result.get('top5', [])[:5]
        for i, res in enumerate(top_models):
            renko = res['renko_size']
            seq = res['seq_len']

            # Recréer le modèle avec ces params
            df_bricks = self._load_or_create_renko(renko)
            df_cleaned = self._prepare_data(df_bricks, self.config)
            X_train, X_test, y_train, y_test, scaler = self._split_and_scale(df_cleaned)

            model = build_transformer(seq_len=seq, features_len=len(self.config['features']))
            model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0,
                      validation_split=0.2,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

            # Hash + sauvegarde
            config_for_hash = {
                "renko_size": renko,
                "lstm_seq_len": seq,
                "features": self.config['features'],
                "target": self.config['target']
            }
            hash_code, _ = manager.save(model, scaler, config_for_hash, res)

            # Graphique equity
            self._save_equity_curve(model, scaler, X_test, y_test, hash_code)

            log(f"TOP {i+1} SAUVEGARDÉ → {hash_code} | Sharpe {res['sharpe']:.3f}")

        # Rapport PDF final
        self._generate_pdf_report(result, top_models)
        log("RAPPORT PDF GÉNÉRÉ → best_models/report.pdf")

        queue.put({'type': 'result', 'data': result})
        queue.put({'type': 'done', 'data': None})

    def _save_equity_curve(self, model, scaler, X_test, y_test, hash_code):
        proba = model.predict(X_test, verbose=0).flatten()
        signal = np.where(proba > 0.5, 1, -1)
        returns = signal * y_test  # si target_type = return
        equity = (1 + returns).cumprod()

        plt.figure(figsize=(12,6))
        plt.plot(equity, label=f"Equity (Sharpe {np.mean(returns)/np.std(returns)*np.sqrt(252):.2f})")
        plt.title(f"Equity Curve – {hash_code}")
        plt.legend()
        path = f"{self.save_dir}/equity_{hash_code}.png"
        plt.savefig(path)
        plt.close()

    def _generate_pdf_report(self, result, top5):
        doc = SimpleDocTemplate(f"{self.save_dir}/GRAAL_REPORT.pdf", pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("GRAAL FINAL – RAPPORT D'OPTIMISATION", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Tableau top 5
        data = [["Rang", "Hash", "Renko", "Seq", "Sharpe", "PnL"]]
        for i, r in enumerate(top5):
            data.append([i+1, r.get('hash', 'N/A')[:8], r['renko_size'], r['seq_len'], f"{r['sharpe']:.3f}", f"{r['pnl']:.1%}"])

        table = Table(data)
        table.setStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)])
        story.append(table)

        # Images
        for r in top5[:3]:
            img_path = f"{self.save_dir}/equity_{r.get('hash', '')[:8]}.png"
            if os.path.exists(img_path):
                story.append(Spacer(1, 12))
                story.append(Image(img_path, width=500, height=300))

        doc.build(story)
