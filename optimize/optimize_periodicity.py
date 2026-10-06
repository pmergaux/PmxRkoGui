import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time

import joblib
import numpy as np
import pandas as pd
from mt5linux import MetaTrader5

from strategy.pmxRko import PmxRkoStrategy, get_adaptive_window_size
# Importations de votre architecture
from utils.utils import connectMt5, JAUNE, VERT, BLEU, ROUGE, RESET, get_clean_timestamp, BLEU_CIEL, SELL, BUY, NONE, CLOSE, FCLOSE
from utils.model_utils import load_model
from live.connexion import Connexion, select_positions_magic
from utils.renko_utils import tick21renko, analyse_renko_iqr
from decision.candle_decision import add_indicators_optimized, choix_features_numba, fast_stats_numba
from decision.trading_decision import decision_ai, proba_final, IndicatorMonitor, calcul_bornes, calcul_situation, \
    decision_rates_optimized, get_last_decision, trading_decision, monitoring, get_rnn_only_decision
from utils.scaler_utils import load_and_transform
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

version_rnn = True
MIN_PROFIT = 20.0
INITIAL_CAPITAL = 10000.0

@dataclass
class Position:
    ticket: int = 0
    time_open: datetime = datetime(1970, 1, 1)
    time_close: datetime = datetime(1970, 1, 1)
    type: int = 0  # BUY (1) ou SELL (-1)
    volume: float = 1.0
    price_open: float = 0.0
    price_current: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    reason: str = ""
    magic: int = 125788

class MockConnexion:
    def __init__(self, backtester):
        self.backtester = backtester
    def get_positions_symbol(self, symbol):
        return self.backtester.positions
    def get_positions_ticket(self, ticket):
        for pos in self.backtester.positions:
            if pos.ticket == ticket:
                return pos
        return None
    def market_order_trade_execution(self, sens, lot, live, sl, tp):
        self.backtester.open_position(sens)
        return True
    def close_one(self, position, trace, msg):
        return self.backtester.close_position(msg)

class PmxRkoBacktester(PmxRkoStrategy):
    def __init__(self, config, ticks_df=None):
        PmxRkoStrategy.__init__(self, None, config)
        self.backtest_mode = True
        self.spread_dollar = 2.97  # spread en $
        self.cl = MockConnexion(self)

        # Suivi des positions en mémoire
        self.positions_history: list = []  # Historique des positions clôturées
        self.ticket_counter = 0
        self.total_pnl = 0.0

        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_time = None
        self.ticks = ticks_df
        self.all_bricks = None

        self.local = False

    def modify_sl(self, position, new_sl):
        position.sl = new_sl

    def get_position_type(self, position):
        ls = position.type
        return ls

    def run_periodicity_optimization(self):
        # 1. Normalisation du volume[cite: 7]
        if 'volume' not in self.ticks.columns and (
                'tick_volume' in self.ticks.columns or 'volume_real' in self.ticks.columns):
            col = 'tick_volume' if 'tick_volume' in self.ticks.columns else 'volume_real'
            self.ticks['volume'] = self.ticks[col]
        if 'volume' not in self.ticks.columns or self.ticks['volume'].sum() < len(self.ticks):
            self.ticks['volume'] = 1

        param_cfg = self.cfg.get("parameters", {})
        self.renko_size = param_cfg.get("renko_size", 20.4)
        minimum = 96

        print("\n🧱 Génération globale unique des briques Renko et des indicateurs...")
        # Génération des briques sur tout l'historique en une seule fois
        self.all_bricks = tick21renko(self.ticks, None, self.renko_size, value='bid')

        if self.all_bricks is None or len(self.all_bricks) < minimum:
            print("❌ Pas assez de briques Renko générées pour lancer le backtest.")
            return
        self.run_backtest_sequential(minimum)

    def generation_proba(self, minimum=128):
        start = time.time()
        probabilities_list = []

        print("⏳ Étape 1/2 : Pré-calcul global des indicateurs et features...")

        # 1. ON CALCULE TOUT UNE SEULE FOIS SUR TOUT LE DATASET GLOBAL
        # (Attention : assurez-vous que vos indicateurs n'utilisent pas de données du futur)
        df_global = add_indicators_optimized(self.all_bricks, self.cfg)
        df_global = choix_features_numba(df_global, self.cfg)

        print(f"⏳ Étape 2/2 : Inférence par {len(self.all_bricks) - minimum} fenêtres glissantes sur {len(df_global)}...")

        if 'TABFF' in self.models:
            from tabicl import TabICLRegressor
            process_id = os.getpid()
            output_dir = f"./ckpts_{process_id}/best.ckpt"
            # Option 1 (Le plus courant si le paramètre s'appelle model_path ou checkpoint_path) :
            try:
                self.models['TABFF'] = TabICLRegressor(model_path=output_dir, device="cpu")
            except Exception as e:
                raise e
        self.renko_time = self.all_bricks['time'].iloc[-1]
        # 2. La boucle se contente de trancher dans le DataFrame déjà calculé
        for i in range(minimum, len(df_global)):
            # Extraction de la tranche de 128 briques déjà prête
            current_slice = df_global.iloc[i - minimum: i]
            # ----------------------------------------------------
            # SYNCHRONISATION ROBUSTE :
            # On récupère les indices exacts de current_slice pour extraire
            # les briques correspondantes dans self.all_bricks sans décalage.
            # ----------------------------------------------------
            current_bricks = self.all_bricks.loc[current_slice.index]
            # Sécurité anti-fuite temporelle (votre vérification de l'heure)
            if not current_slice.empty and current_slice['time'].iloc[-1] == self.renko_time:
                # On applique la même coupe aux briques !
                # Inférence IA (seule partie lourde inévitable pour garder le flux temporel strict)
                current_slice = current_slice.iloc[:-1]
                current_bricks = current_bricks.iloc[:-1]
            if current_slice['time'].iloc[-1] == self.renko_time:
                current_slice = current_slice.iloc[:-1]
            proba = decision_ai(current_slice, current_bricks, self.cfg, self.scaler, self.models, test=True)
            proba_fin = proba_final(proba, self._param.get("weights", None))
            if proba_fin is not None and len(proba_fin) > 0:
                probabilities_list.append(proba_fin[-1])
            else:
                probabilities_list.append(0.5)  # Valeur neutre de secours
        print(
            f"✅ sur {len(df_global)-minimum} {len(probabilities_list)} Prédictions terminées en {(time.time() - start):.1f} s. Démarrage du backtest rapide...")
        return probabilities_list

    def run_backtest_sequential(self, minimum=128):
        self.monitor_indic = IndicatorMonitor(window_size=self.slope_window, min_samples=self.slope_window)
        self.monitor_means = IndicatorMonitor(window_size=self.slope_window, min_samples=self.slope_window)
        self.situation = 0
        proba_list = self.generation_proba(minimum)
        llp = len(proba_list)
        warm_up_limit = 0
        cnt = self.slope_window * self.slope_window
        for i in range(cnt):
            proba_local = proba_list[i]
            """
            try:
                if proba_local is not None:
                    # Conversion systématique en tableau numpy pour manipuler listes ou scalaires uniformément
                    proba_arr = np.asarray(proba_local, dtype=float).ravel()
                    # 1. Gestion sécurisée des NaN sur un tableau ou une liste
                    if np.isnan(proba_arr).any():
                        continue
                else:
                    continue
            except TypeError:
                continue
            """
            if np.isnan(proba_local):
                continue
            monitoring(proba_local, self.monitor_indic, self.monitor_means)
            if self.monitor_means.is_ready:
                break
        warm_up_limit = cnt + minimum
        llp = len(proba_list)   # - cnt nb de tests à faire
        jk = len(self.all_bricks)
        # analyse_renko_iqr(self.all_bricks.iloc[warm_up_limit:])
        # Simulation brique par brique de manière fluide
        deb = jk - llp + cnt
        print(f"🚀 Lancement du backtest séquentiel de {llp-cnt} proba de {deb} à {jk}"
              f" briques Renko ...")
        ind_cfg = self.cfg.get("indicators_and_filters", {})
        reg_window = ind_cfg.get("regression", {}).get("window", 18)
        for i in range(deb, jk):
            self.bricks = self.all_bricks.iloc[i-reg_window:i]
            # Récupération des derniers prix pour le suivi des positions
            self.last_bid = self.all_bricks['close'].iloc[i]
            self.last_ask = self.last_bid + self.spread
            self.last_time = self.all_bricks['time'].iloc[i]

            # Gestion du Stop Loss / Take Profit des positions ouvertes
            if len(self.positions) > 0:
                position = self.positions[0]
                ls = position.type
                price_for_pnl = self.last_bid if ls == BUY else self.last_ask
                position.price_current = price_for_pnl
                position.profit = (price_for_pnl - position.price_open) * position.volume * ls
                """
                if position.sl != 0.0 and (position.price_current - position.sl) * ls < 0:
                    self.close_position("sl")
                elif position.tp != 0.0 and (position.price_current - position.tp) * ls > 0:
                    self.close_position("tp")
                """
            self.proba = proba_list[i-deb+cnt]
            # Exécution de la stratégie sur cette brique
            self.run_trade()

        # Clôturer la dernière position si elle est restée ouverte à la fin
        if len(self.positions) > 0:
            self.close_position("end")

        profits = [pos.profit for pos in self.positions_history]
        total_trades = len(profits)
        total_profit = sum(profits)
        print(f"✅ Backtest séquentiel par briques terminé avec succès. profit total:{total_profit:.2f} en {total_trades} trades")
        return self.positions_history

    def open_position(self, sens):
        price = self.last_ask if sens == BUY else self.last_bid
        base = self.last_ask if sens == SELL else self.last_bid
        # a modifier selon
        self.ssl = self.live.get("sl", 0)
        self.stp = self.live.get("tp", 0)
        # self.ssl = self.renko_size * 2
        # self.stp = self.renko_size
        self.ticket_counter += 1
        self.positions.append(Position(
            ticket=self.ticket_counter,
            time_open=self.last_time,
            type=sens,
            volume=self.live.get("volume", 1.0),
            price_open=price,
            price_current=price,
            sl=(base - (self.live.get("sl", 0) * sens)) if self.live.get("sl", 0) > 0 else 0.0,
            tp=(base + (self.live.get("tp", 0) * sens)) if self.live.get("tp", 0) > 0 else 0.0,
            #sl=(base - self.renko_size * 2 * sens),
            #tp=(base + self.renko_size * sens),
            magic=self.live.get("magic", 125788)
        ))
        print(f"At {self.last_time} ouverture d'une position {sens} price {price:.2f} {self.positions[-1].sl:.2f} {self.positions[-1].tp:.2f}")

    def close_position(self, msg):
        if len(self.positions) == 0:
            return True
        position = self.positions[0]
        position.time_close = self.last_time
        sens = position.type
        price = self.last_bid if sens == BUY else self.last_ask
        position.price_current = price
        # Calcul du profit avec déduction réaliste du spread
        net_profit = (price - position.price_open) * position.volume * sens
        if position.time_close and position.time_open and position.time_close > position.time_open:
            # .date() extrait uniquement l'année, le mois et le jour (ignore l'heure)
            # La soustraction gère automatiquement les fins de mois et d'années !
            delta_days = (position.time_close.date() - position.time_open.date()).days
            # Si delta_days > 0, la position a traversé au moins un changement de jour
            if delta_days > 0:
                # Option A : 0.6$ par nuit passée (recommandé en trading)
                swap_fee = delta_days * 0.6
                # Option B : 0.6$ fixe unique si le jour est différent (décommentez si c'est ce que vous voulez)
                # swap_fee = 0.6
            else:
                swap_fee = 0.0
            # Vous pouvez impacter directement le profit net de la position
            net_profit -= swap_fee
        position.profit = np.round(net_profit, 2)
        position.reason = msg
        self.positions_history.append(position)
        self.positions = []
        profits = [pos.profit for pos in self.positions_history]
        total_trades = len(profits)
        total_profit = sum(profits)
        print(f"At {self.last_time} fermeture d'une position {msg} price {price:.2f} profit {net_profit:.2f} total {total_profit:.2f} en {total_trades}")
        return True

    def performance(self, trace=False):
        """Affiche les résultats statistiques détaillés de la simulation."""
        if not self.positions_history:
            print("Aucun trade exécuté durant le backtest.")
            return None

        profits = [pos.profit for pos in self.positions_history]
        durations_seconds = []
        for pos in self.positions_history:
            # S'assurer que les dates sont valides et que la fermeture est postérieure à l'ouverture
            if pos.time_close and pos.time_open and pos.time_close > pos.time_open:
                duration = (pos.time_close - pos.time_open).total_seconds()
                durations_seconds.append(duration)
        # Calcul de la moyenne et du max (par exemple convertis en minutes ou en heures)
        if durations_seconds:
            mean_duration_min = np.mean(durations_seconds) / 60.0
            max_duration_min = np.max(durations_seconds) / 60.0
        else:
            mean_duration_min = 0.0
            max_duration_min = 0.0
        delta = self.all_bricks['time'].iloc[-1] - self.all_bricks['time'].iloc[0]
        nb_jours = delta.total_seconds() / 86400.0
        if nb_jours < 1: nb_jours = 1.0  # Sécurité anti-division par zéro
        a = np.array(profits)
        nb_trades = len(a)
        win_trades = np.sum(a > 0)
        loss_trades = np.sum(a < 0)
        none_trades = np.sum(a == 0)
        if none_trades != 0:
            print(f"---------------------- alerte ----------none_trades: {none_trades}")
            #return None
        if win_trades + loss_trades + none_trades != nb_trades:
            print(f"alerte {win_trades}+{loss_trades}+{none_trades} != {nb_trades}")
            return None
        profit = np.sum(a)
        # 3. Calcul du score Sharpe avec pénalité de volatilité
        # On utilise l'écart-type (std) pour favoriser la régularité
        win_rate = win_trades / nb_trades
        mean = a.mean()
        std_pnl = a.std() if len(a) > 1 else 0.0
        if nb_trades > 0:
            sum_gains = float(np.sum(a[a > 0]))  # Somme des gains
            sum_pertes = float(np.sum(a[a < 0]))  # Somme des pertes (négatif)
        else:
            sum_gains = 0.0
            sum_pertes = 0.0
        # Max Drawdown
        # 1. Calcul du Max Drawdown (en points cumulés)
        # On transforme les résultats en Equity Curve (bilan cumulé)
        equity_curve = np.cumsum(a)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        max_drawdown = np.max(drawdown)

        # 1. Calculs intermédiaires de base pour le Profit Factor et le Drawdown
        if nb_trades > 0 and 'sum_gains' in locals() and 'sum_pertes' in locals():
            # PLANCHER DE SÉCURITÉ DU PROFIT FACTOR (Règle des 10% minimum de pertes)
            # Si sum_pertes est nul ou trop faible, on s'assure qu'au moins 10% des gains
            # sont comptabilisés en pertes pour éviter l'explosion vers l'infini.
            plancher_perte_min = max(abs(sum_gains) * 0.10, 1e-4)
            safe_sum_pertes = max(abs(sum_pertes), plancher_perte_min)

            profit_factor = sum_gains / safe_sum_pertes
        else:
            profit_factor = 1.0

        # Sécurisation et calcul du Max Drawdown en pourcentage du capital
        safe_initial_capital = max(INITIAL_CAPITAL, 1.0)
        max_drawdown_pct = abs(max_drawdown) / safe_initial_capital

        # RÈGLE DU PALIER DE 10% POUR LE DRAWDOWN :
        # Si le drawdown est à 0% ou inférieur à 10%, on le sature à 0.10 (10%).
        safe_max_dd_pct = max(max_drawdown_pct, 0.10)

        # 2. Calmar / Profit-to-Drawdown Ratio encadré par le palier
        calmar_like_ratio = profit / safe_max_dd_pct

        arg_log = profit * 100
        MIN_TRADES = max(5, int(nb_jours / 2.2))

        if arg_log <= 0.0:
            # Plus la perte est grande, plus le score est sévèrement négatif
            score = float(arg_log * 2.0)
        elif profit < MIN_PROFIT or nb_trades < MIN_TRADES:
            # Pénalité négative progressive pour guider Optuna vers la zone valide
            missing_trades = max(0, MIN_TRADES - nb_trades)
            missing_profit = max(0.0, MIN_PROFIT - profit)
            score = -50.0 - (missing_trades * 5.0) - (missing_profit * 2.0)
        else:
            # 3. Facteur de pénalité douce si le Drawdown dépasse le seuil tolérable (ex: 20%)
            seuil_dd_acceptable = 0.20  # 20% de Max DD max toléré de référence
            facteur_dd = 1.0 / (1.0 + (max(max_drawdown_pct, 1e-4) / seuil_dd_acceptable))

            # 4. Score final combiné et totalement stabilisé
            score = float(
                profit_factor *
                np.log1p(arg_log) *
                np.log1p(calmar_like_ratio) *
                facteur_dd
            )

        mean_gain = np.mean(a[a > 0]) if np.any(a > 0) else 0
        mean_loss = abs(np.mean(a[a < 0])) if np.any(a < 0) else 1e-6  # éviter division par 0

        if trace:
            print("\n" + "=" * 50)
            print("📊 RÉSULTATS DU BACKTEST COMPLET (TICK-BASED)")
            print("=" * 50)
            print(f"💰 Profit Total : {profit:.2f} $")
            print(f"🔄 Nombre de Trades : {nb_trades}")
            print(f"🎯 Taux de Réussite (Win Rate) : {win_rate:.2f}%")
            print(f"📉 Max Drawdown : {max_drawdown:.2f} $")
            print(f"  Durations : Mean = {mean_duration_min:.2f} min, Max = {max_duration_min:.2f} min")
            print(f"trades {a}")
            print("=" * 50)
        """
        print("\n📋 DÉTAILS DES TRADES :")
        for pos in self.positions_history:
            type_str = "BUY" if pos.type == BUY else "SELL"
            dur = pos.time_close - pos.time_open
            print(
                f"  #{pos.ticket} | {type_str} | Open: {pos.price_open:.2f} ({pos.time_open}) | Close: {pos.price_current:.2f} ({pos.time_close}) | Durée: {dur} | Net: {pos.profit:.2f} $ ({pos.reason})")
        """
        result = {
            'score': score,
            "profit": profit,
            "trades": nb_trades,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "mean": mean,
            "std": std_pnl,
            'mean_gain': mean_gain,
            'sum_gains': sum_gains,
            'mean_loss': mean_loss,
            'sum_pertes': sum_pertes,
            'P/L': mean_gain / mean_loss,
            'profit_factor': profit_factor,
            'days': np.round(nb_jours, 1),
            'duration moy ': mean_duration_min,
            'duration max': max_duration_min,
            "hcode": self.hcode,
            "version": self.version,
            "positions": profits,
        }
        return result

    def evaluate_and_save_champion(self):
        """Calcule le score final du backtest et sauvegarde les modèles/scalers/configs sous des noms standards, puis les renomme à la fin."""
        result = self.performance()
        if result is None:
            return
        if not self.positions_history:
            print("Aucune position clôturée. Impossible d'évaluer.")
            return 0.0

        # Formule hybride comparable à backtest_module.py
        profits = result['positions']   #[pos.profit for pos in self.positions_history]
        score = result['score']
        profit = result['profit']
        nb_trades = result['trades']
        print(f"\n📊 Évaluation de l'essai : Score = {score:.2f}")

        # Ne pas sauvegarder de champion si le profit est négatif ou trop peu de trades
        min_trades = max(5, int(len(self.df) * 0.03)) if self.df is not None else 5
        if profit <= 100 or nb_trades < min_trades:
            print(
                f"⚠️ Conditions de champion non remplies (Profit requis > 100 $, actuel: {profit:.2f} $ | Trades requis >= {min_trades}, actuels: {nb_trades})")
            return score

        # Vérifier si c'est un record battu
        best_score_so_far = -float('inf')
        record_path = os.path.join(ROOT_DIR, "best_score.json")
        if os.path.exists(record_path):
            try:
                with open(record_path, 'r') as f:
                    meta = json.load(f)
                    best_score_so_far = meta.get("score", -float('inf'))
            except Exception:
                pass

        if score > best_score_so_far:
            print(
                f"✨ NOUVEAU RECORD DE BACKTEST DE FLUX : {score:.3f} > {best_score_so_far:.3f}. Sauvegarde sous les noms standards...")
            """
            # 1. Calcul du hcode dynamique
            from utils.config_utils import config_to_hash, prepare_to_hashcode
            try:
                hcode = config_to_hash(prepare_to_hashcode(self.cfg))
            except Exception:
                hcode = self.hcode
            self.cfg['live']['hcode'] = hcode
            """
            hcode = self.hcode
            """
            # 2. Sauvegarde de la Config sous le nom standard "config_hcode.json"
            strategy_name = self.live.get('name', 'unknown_strategy')
            nested_config = {strategy_name: self.cfg}
            config_path = os.path.join(ROOT_DIR, "config_hcode.json")
            with open(config_path, 'w') as f:
                json.dump(nested_config, f, indent=4)
            print(f"  -> Config standard sauvegardée sous {config_path}")

            # 3. Sauvegarde du modèle RNN (Keras, XGB, LGBM...) sous le nom standard "model_hcode"
            model_rnn_path = ""
            if self.model_rnn is not None:
                model_base = os.path.join(ROOT_DIR, "models/model_hcode")
                try:
                    if "XGB" in self.version:
                        model_rnn_path = f"{model_base}.json"
                        self.model_rnn.save_model(model_rnn_path)
                    elif "LGBM" in self.version:
                        model_rnn_path = f"{model_base}.txt"
                        self.model_rnn.save_model(model_rnn_path)
                    else:
                        model_rnn_path = f"{model_base}.keras"
                        self.model_rnn.save(model_rnn_path)
                    print(f"  -> Modèle RNN standard sauvegardé sous {model_rnn_path}")
                except Exception as e:
                    print(f"  ⚠️ Erreur lors de la sauvegarde du modèle RNN standard: {e}")

            # 4. Sauvegarde du modèle TabICL sous le nom standard "temp_tabicl.joblib"
            tabicl_path = ""
            if self.model_tabicl is not None and self.scaler is not None:
                tabicl_path = os.path.join(ROOT_DIR, "models/temp_tabicl.joblib")
                try:
                    import joblib
                    bundle = {
                        "model": self.model_tabicl,
                        "scaler": self.scaler,
                        "features": self.cfg.get("features", [])
                    }
                    joblib.dump(bundle, tabicl_path)
                    print(f"  -> Modèle TabICL standard sauvegardé sous {tabicl_path}")
                except Exception as e:
                    print(f"  ⚠️ Erreur lors de la sauvegarde du modèle TabICL standard: {e}")

            # 5. Sauvegarde du Scaler sous le nom standard "scaler_hcode.pkl"
            scaler_path = ""
            if self.scaler is not None:
                scaler_path = os.path.join(ROOT_DIR, "models/scaler_hcode.pkl")
                try:
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    print(f"  -> Scaler standard sauvegardé sous {scaler_path}")
                except Exception as e:
                    print(f"  ⚠️ Erreur lors de la sauvegarde du Scaler standard: {e}")
            """
            # 6. Sauvegarde des métadonnées sous le nom standard "best_score.json"
            result["date_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(record_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"🏆 Fichier record standard sauvegardé : {record_path}")

            # ==========================================================
            # 7. RENOMMAGE FINAL DES FICHIERS STANDARDS AVEC LEUR HCODE
            # ==========================================================
            """
            print(f"\n🔄 Renommage final avec le hcode '{hcode}'...")

            # Renommer la config : config_hcode.json -> config_{hcode}.json
            final_config_path = os.path.join(ROOT_DIR, f"config_{hcode}.json")
            if os.path.exists(config_path):
                if os.path.exists(final_config_path):
                    os.remove(final_config_path)
                os.rename(config_path, final_config_path)
                print(f"  -> {config_path} renommé en {final_config_path}")

            # Renommer le modèle RNN
            if model_rnn_path and os.path.exists(model_rnn_path):
                ext = ".json" if "XGB" in self.version else ".txt" if "LGBM" in self.version else ".keras"
                final_model_path = os.path.join(ROOT_DIR, f"models/model_{hcode}{ext}")
                if os.path.exists(final_model_path):
                    os.remove(final_model_path)
                os.rename(model_rnn_path, final_model_path)
                print(f"  -> {model_rnn_path} renommé en {final_model_path}")

            # Renommer le modèle TabICL : temp_tabicl.joblib -> tabicl_{hcode}.joblib
            if tabicl_path and os.path.exists(tabicl_path):
                final_tabicl_path = os.path.join(ROOT_DIR, f"models/tabicl_{hcode}.joblib")
                if os.path.exists(final_tabicl_path):
                    os.remove(final_tabicl_path)
                os.rename(tabicl_path, final_tabicl_path)
                print(f"  -> {tabicl_path} renommé en {final_tabicl_path}")

            # Renommer le Scaler : scaler_hcode.pkl -> scaler_{hcode}.pkl
            if scaler_path and os.path.exists(scaler_path):
                final_scaler_path = os.path.join(ROOT_DIR, f"models/scaler_{hcode}.pkl")
                if os.path.exists(final_scaler_path):
                    os.remove(final_scaler_path)
                os.rename(scaler_path, final_scaler_path)
                print(f"  -> {scaler_path} renommé en {final_scaler_path}")

            # Sauvegarde d'un fichier de métadonnées nommé d'après le hcode pour contrôle des records
            final_record_path = os.path.join(ROOT_DIR, f"best_{hcode}_backtest.json")
            with open(final_record_path, 'w') as f:
                json.dump(meta_record, f, indent=4)
            print(f"🏆 Métadonnées finales enregistrées : {final_record_path}")

            # Mettre à jour l'attribut local
            self.hcode = hcode
            """
        else:
            print(
                f"📉 Score actuel ({score:.3f}) insuffisant pour battre le record ({best_score_so_far:.3f}). Pas de sauvegarde.")

        return score


def get_ticks(cfg):

    live = cfg.get("live", {})
    filename = f"/media/pierre/datad/data/{live.get('symbol', 'ETHUSD')}_120.csv"

    if os.path.exists(filename):
        ticks = pd.read_csv(filename, sep=";")
        # On s'assure de garder time_msc en mémoire avant d'assigner l'index datetime
        if 'time_msc' not in ticks.columns and ticks.index.name == 'time_msc':
            ticks.reset_index(inplace=True)
        ticks['datetime'] = pd.to_datetime(ticks['time_msc'], unit='ms')
        ticks.set_index('datetime', inplace=True)
    else:
        print(f"[{datetime.now()}] Connexion à MetaTrader 5 pour {live.get('symbol')}...")

        mt5 = connectMt5(live)
        clg = Connexion(mt5, live.get("mt5_login"), live.get("mt5_password"), mt5_path=live.get("mt5_path"))

        if not clg.login():
            print("❌ Connexion MT5 rejetée.")
            return

        # 2. Chargement des 2 mois de ticks
        dn = datetime.now()
        dc = dn - timedelta(days=120)
        symbol = live.get("symbol", "ETHUSD")

        print(f"📥 Téléchargement des ticks du {dc.strftime('%Y-%m-%d')} au {dn.strftime('%Y-%m-%d')}...")
        dfo = mt5.copy_ticks_range(symbol, dc, dn, mt5.COPY_TICKS_ALL)
        mt5.shutdown()

        ticks = pd.DataFrame(dfo)
        # On sauvegarde time_msc explicitement dans le CSV et le DataFrame
        ticks['datetime'] = pd.to_datetime(ticks['time_msc'], unit='ms')
        ticks.to_csv(filename, sep=";", index=False)
        ticks.set_index('datetime', inplace=True)

    if ticks is None or len(ticks) == 0:
        print("❌ Aucun tick récupéré.")
        return None

    return ticks

def verify_alignment_structure(dfo, probas_finales_global):
    """
    Vérifie que la taille et la correspondance des index
    entre le DataFrame d'origine et le tableau global sont parfaites.
    """
    print("🔍 Validation structurelle de l'alignement...")

    assert len(dfo) == len(
        probas_finales_global), "❌ Erreur de taille : Le DataFrame et les probas ont des longueurs différentes."

    # Vérification des NaN sur la période de chauffe initiale
    nan_count = np.isnan(probas_finales_global).sum()
    print(f"   • Nombre total de briques : {len(dfo)}")
    print(f"   • Briques avec proba valide : {len(dfo) - nan_count}")
    print(f"   • Briques en période de chauffe (NaN) : {nan_count}")

    print("✨ Structure validée avec succès : Aucun décalage d'index détecté.")
    return True


if __name__ == "__main__":
    config_path = f"../config_live.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f).get("pmxRKO", {})
    ticks = get_ticks(cfg)
    if ticks is None:
        exit(10)
    # 1. Chargement de la configuration (ex: config_live.json)
    for filename in ["config_4614cc97532a.json",
                     #"config_8d0fc1bbd5b7.json",
                     ]:
        config_path = f"../{filename}"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_o = json.load(f)
        else:
            print("❌ Fichier de configuration introuvable.")
            continue
        cfg = cfg_o.get("pmxRKO", {})
        if len(cfg) == 0:
            cfg = cfg_o
        pmxrko = PmxRkoBacktester(cfg, ticks.copy(), ['T', 'F', 'D'])
        pmxrko.load_model_scaler()
        pmxrko.local = True
        pmxrko.run_periodicity_optimization()
        pmxrko.evaluate_and_save_champion()
        del pmxrko
