# optimization_dialog.py
import glob

import joblib
import pandas as pd
import json
import os

from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox, QComboBox, QLineEdit, QDateEdit
from PyQt6 import uic
from datetime import datetime
from multiprocessing import Queue

from optimize.lstm_optimizer import run_optimization
from optimize.optimization_worker import OptimizationWorker
from utils.config_utils import indVal, tarVal
from utils.lstm_utils import create_sequences, generate_param_combinations
from utils.qwidget_utils import get_widget_from_dict, set_widget_from_dict, set_widget_from_list, get_widget_from_list, \
    qdate2datetime
from utils.utils import load_ticks, to_number, reload_ticks_from_pickle


# ===================================================================
# 2. DIALOGUE D'OPTIMISATION (GUI + RETRAIN FINAL)
# ===================================================================
# gui/optimisation_tab.py
class OptimizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        uic.loadUi("ui/config_optim_tab.ui", self)
        self.parent = parent
        self.cfg = parent.get_config_optim()
        # Connexions boutons
        self.btn_save.clicked.connect(self.save_config)
        self.btn_load.clicked.connect(self.load_config)
        self.btn_validate.clicked.connect(self.validate_and_show)
        self.btn_start_opt.clicked.connect(self.start_optimization)
        # NOTA : pour retrouver un widget par son nom widget = getattr(self, nom)

        self.mapping = {
            "period": {
                "date_start": self.date_start,
                "date_end": self.date_end
            },
            "parameters" : {
                "renko_size": (self.renko_size, self.opt_renko_list),
                "ema_period": (self.ema_period, self.opt_ema_period_list),
                "rsi_period": (self.rsi_period, self.opt_rsi_period_list),
                "rsi_high": (self.rsi_high, self.opt_rsi_high_list),
                "rsi_low": (self.rsi_low, self.opt_rsi_low_list),
            },
            "lstm": {
                "lstm_seq_len": (self.lstm_seq_len, self.opt_lstm_seq_list),
                "lstm_units": (self.lstm_units, self.opt_lstm_units_list),
                "lstm_threshold_buy": (self.lstm_threshold_buy, self.opt_threshold_buy_list),
                "lstm_threshold_sell": (self.lstm_threshold_sell, self.opt_threshold_sell_list),
            },
            "features": [self.feat_1, self.feat_2, self.feat_3, self.feat_4, self.feat_5, self.time_live],
            "target": {
                "target_col":self.target_col, "target_type":self.target_type, "target_include": self.target_include
            },
            "open_rules":{
                'rule_ema': self.rule_ema, 'rule_rsi': self.rule_rsi, 'rule_macd': self.rule_macd },
            "close_rules": {"close_ema": self.close_ema, "close_sens": self.close_sens},
            "live": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "sl": self.sl,
                "tp": self.tp,
                "volume": self.volume,
                "magic": self.magic,
                "name": self.name
            }
        }
        for value in self.mapping["features"]:
            if value.objectName() != "time_live":
                value.addItems(indVal)
        self.target_col.addItems(tarVal)
        self.set_optim_config(self.cfg)
        self.queue = None

    def parse_range_or_list(self, text: str):
        """
        Parse un texte comme "10,15,20" ou "10-30-5"
        Retourne une liste de valeurs (float/int)
        """
        text = text.strip()
        if not text:
            return []
        if ',' in text:
            # Format liste : 10,15,20
            return [to_number(x.strip()) for x in text.split(',') if x.strip()]
        elif '-' in text and text.count('-') == 2:
            # Format range : min-max-step
            try:
                min_val, max_val, step = text.split('-')
                min_val, max_val, step = float(min_val), float(max_val), float(step)
                return [round(min_val + i * step, 6) for i in range(int((max_val - min_val) / step) + 1)]
            except:
                return []
        else:
            # Valeur unique
            val = to_number(text)
            return [val] if val is not None else []

    def get_optim_config(self):
        """Retourne un dict prêt à être sauvegardé ou utilisé pour l'optimisation"""
        self.cfg["period"] = {
                "date_start": self.date_start.date().toString("yyyy-MM-dd"),
                "date_end": self.date_end.date().toString("yyyy-MM-dd")
            }
        get_widget_from_dict(self, self.cfg["period"])
        parameters = self.mapping["parameters"]
        for name, valeurs in parameters.items():
            line_edit, checkbox = valeurs
            text = line_edit.text().strip()
            is_list = checkbox.isChecked()
            if text is None or len(text) == 0:
                continue
            values = self.parse_range_or_list(text)
            if values:
                self.cfg["parameters"][name] = {
                    "values": values,
                    "type": "list" if is_list or len(values) == 1 else "range",
                    "source": text
                }
        lstm = self.mapping["lstm"]
        for name, valeurs in lstm.items():
            line_edit, checkbox = valeurs
            text = line_edit.text().strip()
            is_list = checkbox.isChecked()
            if not text:
                continue
            values = self.parse_range_or_list(text)
            if values:
                self.cfg["lstm"][name] = {
                    "values": values,
                    "type": "list" if is_list else "range",
                    "source": text
                }
        self.cfg["features"] = get_widget_from_list(self, "features")
        get_widget_from_dict(self, self.cfg["target"])
        get_widget_from_dict(self, self.cfg["open_rules"])
        get_widget_from_dict(self, self.cfg["close_rules"])
        get_widget_from_dict(self, self.cfg["live"])
        return self.cfg

    def set_optim_config(self, config):
        # Période
        if "period" in config:
            start = QDate.fromString(config["period"].get("date_start", "2023-01-01"), "yyyy-MM-dd")
            end = QDate.fromString(config["period"].get("date_end", "2025-11-13"), "yyyy-MM-dd")
            self.date_start.setDate(start)
            self.date_end.setDate(end)
        # Paramètres
        paramMap = self.mapping["parameters"]
        for name, data in config.get("parameters", {}).items():
            if name in paramMap:
                line_edit, checkbox = paramMap[name]
                source = data.get("source", ",".join(map(str, data["values"])))
                line_edit.setText(source)
                checkbox.setChecked(data.get("type") == "list")
        # lstm
        lstmMap = self.mapping['lstm']
        for name, data in config.get("lstm", {}).items():
            if name in lstmMap:
                line_edit, checkbox = lstmMap[name]
                source = data.get("source", ",".join(map(str, data["values"])))
                line_edit.setText(source)
                checkbox.setChecked(data.get("type") == "list")
        set_widget_from_list(self, "features", self.cfg["features"])
        set_widget_from_dict(self, self.cfg["target"])
        set_widget_from_dict(self, self.cfg["open_rules"])
        set_widget_from_dict(self, self.cfg["close_rules"])
        set_widget_from_dict(self, self.cfg["live"])

        self.cfg = config

    def save_config(self):
        config = self.get_optim_config()
        if not config["parameters"]:
            QMessageBox.warning(self, "Vide", "Aucun paramètre défini !")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder config optimisation", "config_optim.json", "JSON (*.json)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                QMessageBox.information(self, "Succès", f"Sauvegardé : {path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec sauvegarde :\n{e}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger config optimisation", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.set_optim_config(config)
            QMessageBox.information(self, "Chargé", f"Config chargée")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger :\n{e}")

    def validate_and_show(self):
        """Affiche un aperçu lisible des paramètres (pour debug ou lancement)"""
        config = self.get_optim_config()
        total = 1
        details = ["=== PARAMÈTRES D'OPTIMISATION ==="]
        details.append(f"Période : {config['period']['date_start']} → {config['period']['date_end']}")

        for name, data in config["parameters"].items():
            count = len(data["values"])
            total *= count
            details.append(f"{name}: {data['values']} ({count} valeurs) ← {data['source']}")
        for name, data in config["lstm"].items():
            count = len(data["values"])
            total *= count
            details.append(f"{name}: {data['values']} ({count} valeurs) ← {data['source']}")

        details.append(f"\nTOTAL COMBINAISONS : {total:,}")

        QMessageBox.information(self, "Validation Optimisation", "\n".join(details))
        print(json.dumps(config, indent=2))  # Pour debug console

# ==========================================================
    # dans ton widget principal
    def start_optimization(self):
        config = self.get_optim_config()
        start = qdate2datetime(self.date_start.date())
        end = qdate2datetime(self.date_end.date())
        # === CHARGE DONNÉES RÉELLES MT5 ===   a adapter avec utils
        base_name = f"data/ETHUSD_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
        if not os.path.exists(base_name):
            fbrick = os.path.join('data', 'renko_*.pkl')
            nbrick = glob.glob(fbrick)
            if nbrick:
                try:
                    for fic in nbrick:
                        os.remove(fic)
                except OSError as e:
                    print(f"err supr fichiers {e}")
        df = reload_ticks_from_pickle(base_name, 'ETHUSD', None, start, end)
        if df is None or df.empty:
            QMessageBox.critical(self, "Erreur", "Aucune donnée pour cette période")
            return
        df['time'] = pd.to_datetime(df['time'])
        grid = {}
        paramMap = self.mapping["parameters"]
        for name, data in config.get('parameters', {}).items():
            if name in paramMap:
                values = data.get('values', [])
                if len(values) > 0:
                    grid[name] = values
        paramMap = self.mapping["lstm"]
        for name, data in config.get('lstm', {}).items():
            if name in paramMap:
                values = data.get('values', [])
                if len(values) > 0:
                    grid[name] = values

        self.worker = OptimizationWorker(grid, config, df, self)

        self.worker.log.connect(self.update_log)
        self.worker.progress.connect(self.opt_progress.setValue)
        self.worker.finished.connect(self.on_optimization_done)
        self.worker.error.connect(self.on_error)

        self.btn_start_opt.setEnabled(False)
        self.worker.start()

    def update_log(self, text):
        self.opt_status.setText(text)
        # ou self.text_edit.append(text)

    def on_optimization_done(self, result):
        self.btn_start_opt.setEnabled(True)
        self.opt_status.setText("OPTIMISATION TERMINÉE – Meilleur Sharpe: {:.3f}".format(result['best']['sharpe']))

        # Sauvegarde auto
        with open("last_optimization.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print("MEILLEUR PARAMÈTRE :", result['best'])
        print("TOP 5 :", result['top5'])

    def on_error(self, err):
        self.opt_status.setText("ERREUR CRITIQUE")
        QMessageBox.critical(self, "Erreur", err)
# --------------------------------------------------------
        """
        self.opt_progress.setMaximum(len(generate_param_combinations(grid)))
        for i, params in enumerate(grid):
            profit = self.backtest(df, params)
            if profit > best_profit:
                best_profit, best_params = profit, params
            self.opt_progress.setValue(i + 1)
            self.opt_status.setText(f"Backtest {i + 1}/{len(grid)} → Profit: {profit:.1f}%")
        """

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
"""
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
