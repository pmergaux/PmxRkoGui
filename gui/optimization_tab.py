# optimization_dialog.py
from datetime import datetime, timedelta, time

import joblib
import numpy as np
import pandas as pd
import json
import os

from utils.utils import load_ticks

# ===================================================================
# 2. DIALOGUE D'OPTIMISATION (GUI + RETRAIN FINAL)
# ===================================================================
# optimize/optimization_tab.py
from PyQt6 import uic
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox


class OptimizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        uic.loadUi("ui/optimization_tab.ui", self)
        self.parent = parent

        self.btn_start_opt.clicked.connect(self.start_optimization)
        self.btn_export_params.clicked.connect(self.export_grid)

    def get_date_range(self):
        start = self.date_start.date().toPyDate()
        end = self.date_end.date().toPyDate()
        if start >= end:
            QMessageBox.warning(self, "Erreur", "Date début ≥ date fin !")
            return None
        return start, end

    def generate_grid(self):
        date_range = self.get_date_range()
        if not date_range:
            return []

        start, end = date_range
        params = {
            "renko_size": self._parse_values(self.opt_renko_size.text(), self.opt_renko_list.isChecked()),
            "lstm_seq_len": self._parse_values(self.opt_lstm_seq_len.text(), self.opt_lstm_seq_list.isChecked()),
            "lstm_units": self._parse_values(self.opt_lstm_units.text(), self.opt_lstm_units_list.isChecked()),
            "threshold_buy": self._parse_values(self.opt_threshold_buy.text(), self.opt_threshold_buy_list.isChecked()),
            "threshold_sell": self._parse_values(self.opt_threshold_sell.text(),
                                                 self.opt_threshold_sell_list.isChecked()),
            "rsi_period": self._parse_values(self.opt_rsi_period.text(), self.opt_rsi_period_list.isChecked()),
            "ema_period": self._parse_values(self.opt_ema_period.text(), self.opt_ema_period_list.isChecked()),
        }

        from itertools import product
        keys, values = zip(*[(k, v) for k, v in params.items() if v])
        if not keys:
            QMessageBox.warning(self, "Erreur", "Aucun paramètre défini !")
            return []

        grid = [dict(zip(keys, prod)) for prod in product(*values)]
        self.opt_status.setText(f"{len(grid)} combinaisons → {start} à {end}")
        return grid, start, end

    def export_grid(self):
        pass

    def start_optimization(self):
        result = self.generate_grid()
        if not result:
            return
        grid, start, end = result

        # === CHARGE DONNÉES RÉELLES MT5 ===   a adapter avec utils
        df = load_ticks('ETHUSD', None, start, end)
        if df is None or df.empty:
            QMessageBox.critical(self, "Erreur", "Aucune donnée pour cette période")
            return

        df['time'] = pd.to_datetime(df['time'])

        # === LANCE OPTIMISATION (exemple simple) ===
        best_profit = -float('inf')
        best_params = None

        self.opt_progress.setMaximum(len(grid))
        for i, params in enumerate(grid):
            profit = self.backtest(df, params)
            if profit > best_profit:
                best_profit, best_params = profit, params
            self.opt_progress.setValue(i + 1)
            self.opt_status.setText(f"Backtest {i + 1}/{len(grid)} → Profit: {profit:.1f}%")

        self.show_result(best_params, best_profit)

    def backtest(self, df, params):
        # === EXEMPLE DE BACKTEST (à remplacer par ton LSTM + Renko) ===
        import numpy as np
        returns = df['close'].pct_change().dropna()
        signals = np.random.choice([-1, 0, 1], size=len(returns), p=[0.2, 0.6, 0.2])
        profit = (signals * returns).sum() * 100
        return profit

    def show_result(self, params, profit):
        msg = f"<b>MEILLEUR RÉSULTAT :</b><br>Profit: <b>{profit:.2f}%</b><br><br>"
        for k, v in params.items():
            msg += f"• {k} = {v}<br>"
        QMessageBox.information(self, "Optimisation Terminées", msg)

    def _parse_values(self, text, is_list):
        if not text.strip():
            return []
        if is_list:
            try:
                return [float(x.strip()) for x in text.split(",")]
            except:
                return []
        else:
            try:
                start, end, step = map(float, text.replace(" ", "").split("-"))
                return list(np.arange(start, end + step, step))
            except:
                return []

        # --- old version ---------------------------------------

    """
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self):
        config = {
            "renko_size": self.get_list_from_layout(self.renko_inputs),
            "seq_len": [int(x) for x in self.get_list_from_layout(self.seq_inputs)],
            "ema_period": [int(x) for x in self.get_list_from_layout(self.ema_inputs)],
            "features": [["EMA", "RSI", "MACD_hist", "time_vol"]],
            "start_date": self.start_date.text(),
            "end_date": self.end_date.text()
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.parent.statusBar().showMessage("Config optimisation sauvegardée")

        """
"""
        filename = 'ticks_' + start_date.strftime('%Y_%m_%d_%H_%M_%S') + end_date.strftime('%Y_%m_%d_%H_%M_%S') + '.pkl'
        df_cache = pd.DataFrame()
        if not os.path.exists(filename):
            df_cache = load_ticks(self._param['symbol'], None, start_date, end_date)
            if df_cache.empty:
                print(f"{datetime.now()} Erreur: df_cache vide")
                self.stop_optimization.value = True
                self.btn_start.setEnabled(True)
                return

        if not config["renko_size"]:
            self.logs.append("ERREUR : Renko size vide")
            self.btn_start.setEnabled(True)
            return

        ticks_path = "data/ETHUSD.pkl"
        if not os.path.exists(ticks_path):
            self.logs.append("ERREUR : Fichier ticks manquant")
            self.btn_start.setEnabled(True)
            return

        self.worker = OptimizationWorker(config, ticks_path)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.logs.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        self.btn_start.setEnabled(True)

    def on_optimization_done(self, best_params, best_value):
        self.best_params = best_params
        self.best_value = best_value
        # → Sauvegarde dans config.json
        cfg = self.parent.config_tab.get_config()  # ← CORRIGÉ
        cfg.update(best_params)
        self.parent.config_tab.save_config(cfg)  # ← CORRIGÉ
        self.label.setText("Optimisation terminée. Config mise à jour.")
        self.btn_start.setEnabled(True)

        # === AFFICHAGE RÉSULTAT ===
        msg = f'''
        <b>MEILLEURS PARAMÈTRES</b><br>
        Renko size: {best_params['renko_size']:.5f}<br>
        EMA period: {best_params['ema_period']}<br>
        Seq len: {best_params['seq_len']}<br>
        LSTM units: {best_params['lstm_units']}<br>
        <b>PNL estimé: {best_value:.2f}</b>
        '''
        reply = QMessageBox.question(self, "Optimisation terminée", msg + "<br><b>Retraîner et sauvegarder le modèle ?</b>",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Met à jour config.json
            cfg = self.parent.load_config_from_file()
            cfg.update({
                'renko_size': best_params['renko_size'],
                'ema_period': best_params['ema_period'],
                # ...
            })
            with open("config.json", "w") as f:
                json.dump(cfg, f, indent=2)

            # Redémarre live avec nouveaux params
            #self.parent.tabs.widget(1).strategy = PmxRkoStrategy(cfg)

    def on_finished(self, result):
        self.btn_start.setEnabled(True)
        self.progress.setValue(0)

        if 'error' in result:
            self.logs.append(f"ERREUR : {result['error']}")
            return

        # VALEURS SÉCURISÉES
        sharpe = result.get('sharpe', 0.0) or 0.0
        pnl = result.get('pnl', 0.0) or 0.0
        renko = result.get('renko_size', '?')
        seq = result.get('seq_len', '?')

        self.logs.append(
            f"OPTIMISATION TERMINÉE !\n"
            f"→ Sharpe: {sharpe:.3f}\n"
            f"→ PNL: {pnl:+.1%}\n"
            f"→ Renko: {renko}\n"
            f"→ Seq Len: {seq}\n"
            f"→ Config live générée : config_live.json"
        )


    def retrain_and_save(self):
        try:
            self.label.setText("Retrain en cours...")
            full_params = {
                'renko_size': self.best_params['renko_size'],
                'ema_period': self.best_params['ema_period'],
                'features': ["EMA", "RSI", "MACD_hist"],
                'rsi_period': 14,
                'macd': [26, 12, 9]
            }
            if hasattr(self.parent(), 'cb_timevol') and self.parent().cb_timevol.isChecked():
                full_params['features'].append("time_vol")

            # === RENKO + INDICATEURS ===
            df_bricks = tick21renko(self.df_ticks, None, step=full_params['renko_size'], value='bid')
            df_bricks = calculate_indicators(df_bricks, full_params)
            df_bricks = choix_features(df_bricks, full_params)

            # === SÉQUENCES ===
            X, y = create_sequences(df_bricks[full_params['features']].values, self.best_params['seq_len'])

            # === SCALER ===
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

            # === MODÈLE ===
            model = Sequential([
                LSTM(self.best_params['lstm_units'],
                     input_shape=(self.best_params['seq_len'], len(full_params['features']))),
                Dropout(0.2),
                Dense(1, activation='tanh')
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

            # === SAUVEGARDE ===
            timestamp = int(time.time())
            model_path = f"models/lstm_opt_{timestamp}.keras"
            scaler_path = model_path.replace('.keras', '_scaler.pkl')
            os.makedirs("models", exist_ok=True)
            model.save(model_path)
            joblib.dump(scaler, scaler_path)

            self.model_updated.emit(model_path, model, scaler)
            QMessageBox.information(self, "Succès", f"Modèle sauvegardé :\n{os.path.basename(model_path)}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Échec du retrain :\n{str(e)}")
            self.label.setText("Erreur lors du retrain.")

"""
def save_optimization_result(result, model, scaler):
    os.makedirs("models", exist_ok=True)

    # 1. Modèle
    model.save("models/transformer_best.keras")

    # 2. Scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # 3. Config complète
    result['save_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("models/best_optimization.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # 4. Config live (prête à charger)
    live_config = {
        "renko_size": result['renko_size'],
        "seq_len": result['seq_len'],
        "features": ["EMA", "RSI", "MACD_hist", "time_vol"],
        "ema_period": result['ema_period'],
        "model_path": "models/transformer_best.keras",
        "scaler_path": "models/scaler.pkl"
    }
    with open("config_live.json", 'w') as f:
        json.dump(live_config, f, indent=2)
