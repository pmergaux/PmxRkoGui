# strategy/pmxrko.py
import math
import os
import json
import joblib
import numpy as np
import pandas as pd
from colorlog import exception
from samba.kcc.graph_utils import verify_graph_directed_double_ring_or_small

from strategy.base import Strategy
from utils.renko_utils import tick21renko
import utils.config_utils
from utils.config_utils import set_option
from decision.candle_decision import fast_stats_single, is_market_exploding, calculate_atr_4sl, \
    add_indicators_optimized, choix_features_numba, calculate_atr
from decision.trading_decision import trading_decision, decision_bricks, get_last_decision, \
    decision_ai, decision_rates, proba_final, decision_monitor, \
    calcul_situation, calcul_bornes, calcul_bornes_dynamiques, monitoring, IndicatorMonitor, \
    enhanced_decision, ZoneStabilityFilter, weighted_decision, detect_market_regime, soft_zone_score, discretize_score, \
    get_open_decision, get_close_decision
from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE, calculer_stats, JAUNE, RESET, VIOLET, ROUGE, VERT, BLEU, \
    BLEU_CIEL, get_clean_timestamp, get_dynamic_sensitivity, get_linear_slope
from datetime import datetime, timedelta, time
from mt5linux import MetaTrader5
from live.connexion import select_positions_magic

# On remonte d'un niveau si on est dans un sous-dossier (comme 'strategies/')
# ou on reste à la racine si on est dans 'main.py'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Si votre fichier est dans 'strategies/', la racine est le parent :
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
BUFFER_FILE = os.path.join(DATA_DIR, "accuracy_buffer.json")
version_rnn = True
version_enhanced = True
TEST_PROBA = False

class PmxRkoStrategy(Strategy):
    def __init__(self, parent, config):
        super().__init__(parent, config)
        self.backtest_mode = False
        set_option(config.get("live", {}).get("option", "TFD"))
        self.bricks = None
        self.renko_size = config.get("parameters", {}).get("renko_size", 20)
        self.display = None
        self.renko_time = None
        self.current_time = None
        self.next_time = None
        self.interval = config.get("probability", {}).get("interval", 5)
        self.alert_window = config.get("probability", {}).get("alert_window", 4)
        self.slope_window = config.get("probability", {}).get("slope_window", 4)
        self.z_proba_min = config.get("probability", {}).get("z_proba_min", 1.2)
        self.slope_base = config.get("probability", {}).get("slope_base", 0.005)
        self.v_thresh = config.get("probability", {}).get("v_thresh", 5)
        self.zone_filter = ZoneStabilityFilter(min_stability_time=self._param.get("min_stability_time", 300))

        #self.renko_buffer = self.load_buffer()  # Le collecteur de briques
        #self.min_buffer_size = 50  # Seuil pour le refresh
        self.force_strict_veto = False
        self.rate_time = None
        self.count_time = None
        self.debut = False
        self.num_pos_changed = 0
        if 'XGB' in self.version:
            self.minimum = 256
        if 'JEPA' in self.version:
            self.minimum = self.cfg['jepa']['SEQ_LEN'] + 12
        else:
            self.minimum = config.get("lstm", {}).get("lstm_seq_len", 24) + 24
        self.time_display = None
        if TEST_PROBA:
            self.monitor_proba = IndicatorMonitor(12, 12)
    """
    def calculate_accuracy(self, data_list):
        '''
        Calcule l'accuracy de l'IA.
        data_list: liste de dict contenant 'ia_proba', 'ia_signal' et 'close'
        '''
        if len(data_list) < 2:
            return 50.0

        results_active = [] # Uniquement sur les signaux non-nuls
        results_raw = []    # Sur tous les signaux (proba > 0.5)
        
        for i in range(1, len(data_list)):
            prev_data = data_list[i-1]
            curr_data = data_list[i]
            
            prev_signal = prev_data.get('ia_signal', 0)
            prev_proba = prev_data.get('ia_proba', 0.5)
            
            real_dir = 1 if curr_data['close'] > prev_data['close'] else -1
            
            # 1. Accuracy sur signaux actifs (Precision)
            if prev_signal != 0:
                pred_dir = 1 if prev_signal > 0 else -1
                results_active.append(1 if real_dir == pred_dir else 0)
                
            # 2. Accuracy brute (Directional)
            # On considère que > 0.5 est un achat et < 0.5 une vente théorique
            if prev_proba != 0.5:
                raw_pred_dir = 1 if prev_proba > 0.5 else -1
                results_raw.append(1 if real_dir == raw_pred_dir else 0)

        # Calcul des métriques
        precision = (sum(results_active) / len(results_active)) * 100 if results_active else 50.0
        raw_acc = (sum(results_raw) / len(results_raw)) * 100 if results_raw else 50.0
        signal_rate = (len(results_active) / (len(data_list) - 1)) * 100 if len(data_list) > 1 else 0.0

        print(f"📊 Estimation RNN ({len(data_list)} briques) :")
        print(f"   - Precision (Signaux) : {precision:.1f}% [{len(results_active)} trades]")
        print(f"   - Dir. Accuracy (Raw) : {raw_acc:.1f}%")
        print(f"   - Taux d'activité     : {signal_rate:.1f}%")
        
        return precision # On retourne la précision pour les alertes existantes

    def on_new_renko_brick(self, brick_data):
        self.renko_buffer.append(brick_data)
        # On augmente la taille du buffer pour avoir plus de signaux en cas de faible volatilité
        if len(self.renko_buffer) > 200:
            self.renko_buffer.pop(0)
        # On sauvegarde à chaque mise à jour
        self.save_buffer()
        # On calcule l'accuracy à chaque nouvelle brique
        acc = self.calculate_accuracy(self.renko_buffer)
        # Affichage sélectif comme vous l'avez souhaité
        if acc < 45:
            print(f"⚠️ Alerte Divergence : Accuracy IA à {acc:.1f}%")
        elif acc > 60:
            print(f"✅ Haute Confiance : Accuracy IA à {acc:.1f}%")

    def save_buffer(self):
        try:
            with open(BUFFER_FILE, "w") as f:
                # On convertit le timestamp en string pour le JSON
                buffer_to_save = []
                for b in self.renko_buffer:
                    copy_b = b.copy()
                    copy_b['timestamp'] = str(copy_b['timestamp'])
                    buffer_to_save.append(copy_b)
                json.dump(buffer_to_save, f)
        except Exception as e:
            print(f"Erreur sauvegarde buffer: {e}")

    def load_buffer(self):
        filepath = BUFFER_FILE
        # 1. Créer le dossier 'data' s'il n'existe pas
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 2. Vérifier si le fichier existe
        if not os.path.exists(filepath):
            print("📝 Premier lancement : création du fichier buffer.")
            # On crée un fichier vide avec une liste vide []
            with open(filepath, "w") as f:
                json.dump([], f)
            return []
        # 3. Charger le fichier s'il existe
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Erreur de lecture buffer, réinitialisation : {e}")
            return []
    """
    def to_follow(self,  r2_val, cfg):  # atr_window=14):
        # Simule le stop suiveur ATR dynamique (Trailing Stop) de production
        if len(self.positions) == 0:
            return
        atr_window = cfg.get('atr', {}).get('window', 14)
        if r2_val is not None:
            is_exploding = is_market_exploding(self.df.tail(40), cfg)
            # Calcul du multiplicateur ATR
            for position in self.positions:
                is_buy = (position.type == MetaTrader5.POSITION_TYPE_BUY)
                po = position.price_open
                current_price = position.price_current
                current_sl = position.sl
                if current_sl != 0 and (po - current_sl) * (1 if is_buy else -1) > 0:
                    # SL déjà serré → on reste prudent
                    dynamic_multiplier = 1.1
                else:
                    dynamic_multiplier = 1.2 if is_exploding else max(1.8, 2.8 - (r2_val * 1.8))
                dist_volatilite = calculate_atr_4sl(self.df, multiplier=dynamic_multiplier, window=atr_window)
                if is_buy:
                    target_sl = current_price - dist_volatilite
                    # On bouge uniquement si on améliore le SL et qu'on est en profit
                    if target_sl > current_sl and target_sl > po * 0.999:  # petite marge
                        self.modify_sl(position, target_sl)
                else:
                    target_sl = current_price + dist_volatilite
                    if target_sl < current_sl and target_sl < po * 1.001:
                        self.modify_sl(position, target_sl)
        else:
            dist_volatilite = calculate_atr_4sl(self.df, multiplier=0.8, window=atr_window)
            #print(f"follow de ATR: {dist_volatilite}")
            for position in self.positions:
                is_buy = (position.type == MetaTrader5.POSITION_TYPE_BUY)
                po = position.price_open
                current_price = position.price_current
                current_sl = position.sl
                if current_sl != 0 and (current_sl - po) * 1 if is_buy else -1 > 0:
                    return
                if (po - current_sl) * 1 if is_buy else -1 > 0 or current_sl == 0:
                    dist_volatilite = (self.renko_size-self.spread)/2
                if is_buy:
                    target_sl = current_price - dist_volatilite
                    # On bouge uniquement si on améliore le SL et qu'on est en profit
                    if target_sl > current_sl and target_sl > po + self.spread/4:  # petite marge
                        self.modify_sl(position, target_sl)
                else:
                    target_sl = current_price + dist_volatilite
                    if target_sl < current_sl and target_sl < po - self.spread/4:
                        self.modify_sl(position, target_sl)

    def modify_sl(self, position, new_sl):
        digits = self.mt5.symbol_info(position.symbol).digits
        new_sl = float(round(new_sl, digits))

        # EVITER l'appel si le SL est déjà le bon
        if abs(position.sl - new_sl) < (1 / (10 ** digits)):
            return
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp
        }
        result = self.mt5.order_send(request)
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            # Affiche le code erreur réel (ex: 10015 pour stops invalides)
            print(f"❌ {self.live.get('name', '')} Erreur SL: Code {result.retcode} | Prévu: {new_sl} | Bid: {self.ticks['bid'].iloc[-1]}")
        else:
            print(f"🚀 {self.live.get('name', '')} Trailing ATR mis à jour: {new_sl}")

    def TimeisOpen(self):
        now = datetime.now().time()
        cls = time(22, 50, 00)
        opn = time(23, 30, 00)
        return now < cls or now > opn

    def _xTimeIsOpen(self):
        now = datetime.now()
        deb = datetime.now().replace(hour=23, minute=30)
        return now > deb

    def close_position(self, msg):
        # Cette erreur empêche l'exécution et indique clairement le problème
        raise NotImplementedError(
            f"La méthode 'methode_a_ne_pas_utiliser' doit être implémentée "
            f"par la classe fille '{self.__class__.__name__}'."
        )

    def close_locale(self, rc, msg):
        if self.parent:
            ls = self.get_position_type(self.positions[0])
            self.reprise = self.positions[0].price_current
            self.close_one(ls, rc, self.positions[0], True, msg)
            import time
            self.count_time = time.time()
        else:
            self.close_position(msg)

    def open_position(self, sens):
        # Cette erreur empêche l'exécution et indique clairement le problème
        raise NotImplementedError(
            f"La méthode 'methode_a_ne_pas_utiliser' doit être implémentée "
            f"par la classe fille '{self.__class__.__name__}'."
        )
    def open_locale(self, so):
        self.stp = self.live['tp']
        self.ssl = self.live['sl']
        if not version_rnn:
            #self.live['tp'] = 0
            #self.live['sl'] = 0
            self.open(so, 0)
            self.live['tp'] = self.stp
            self.live['sl'] = self.ssl
        else:
            self.live['tp'] = 0
            self.live['sl'] = 0
            self.open(so, 0)
            self.live['tp'] = self.stp
            self.live['sl'] = self.ssl
            # si version vSLTP
            self.stp = self.renko_size * 2
            self.ssl = self.renko_size * 2

    def run(self):
        if self.debut:
            return
        try:
            import time
            if self.bricks is None:
                start_time = time.time()
                self.live['init_decal'] = 1800
                self.debut = True
                Strategy.run(self)
                try:
                    if self.ticks is None or len(self.ticks) == 0:
                        self.debut = False
                        return
                    # print(f"{self.live['name']} ticks {len(self.ticks)} size {self.renko_size}")
                    self.bricks = tick21renko(self.ticks, None, self.renko_size, 'bid')
                    if self.bricks is None or len(self.bricks) == 0:
                        raise Exception('bricks is None or empty')
                    print(
                        f"{self.live['name']} start len {len(self.ticks)}/{len(self.bricks)} en {(time.time() - start_time) / 60:.1f}")
                    if len(self.bricks) < self.minimum:
                        self.tickLast = None
                        self.debut = False
                        self.bricks = None
                        self.ticks = None
                        self.df = None
                        time.sleep(300)
                        return
                except Exception as e:
                    print(f'Rko err / bricks create: {e}')
                    self.tickLast = None
                    self.debut = False
                    self.bricks = None
                    return
                self.debut = False
            else:
                Strategy.run(self)
                if self.ticks is None or len(self.ticks) == 0:
                    return
                try:
                    if self.bricks is None or len(self.bricks) == 0:
                        raise Exception('bricks empty')
                    self.bricks = tick21renko(self.ticks, self.bricks, step=self.renko_size, value='bid')
                    if self.bricks is None or len(self.bricks) == 0:
                        raise Exception('bricks empty after update')
                except Exception as e:
                    print(f'Rko err / bricks suite: {e}')
                    self.tickLast = None
                    self.renko_time = None
                    return
            if len(self.bricks) < self.minimum:
                print(f'pas assez de renko {len(self.bricks)}, attendu={self.minimum}')
                return
            if len(self.bricks) > self.minimum + 150:
                self.bricks = self.bricks[-self.minimum - 120:]
            if self.regression:
                if not self.monitor_rnn.is_ready:
                    display_local = add_indicators_optimized(self.bricks, self.cfg)
                    display_local = choix_features_numba(display_local, self.cfg)
                    # print(f"display_local: {len(display_local)} bricks {len(self.bricks)}")
                    proba = decision_ai(display_local, self.bricks, self.cfg, self.scaler, self.models)
                # raise ValueError("test stop demandé")
            if not self.monitor_indic.is_ready:
                # 1. Calcul du nombre de fois que slope_window rentre dans proba (division entière)
                # print("len bricks",len(self.bricks))
                cnt = self.slope_window * self.slope_window
                for i in range(cnt):
                    bricks_local = self.bricks.iloc[:-cnt+i]
                    display_local = add_indicators_optimized(bricks_local, self.cfg)
                    display_local = choix_features_numba(display_local, self.cfg)
                    # DIAGNOSTIC : Afficher les colonnes disponibles pour comprendre
                    proba = decision_ai(display_local, bricks_local, self.cfg, self.scaler, self.models)
                    if proba is None or len(proba) == 0:
                        print(f"proba after decision ai is None or empty")
                    proba = proba_final(proba, self._param.get("weights", None))
                    self.monitor_indic.update(proba[-1])
                    _, moy = self.monitor_indic.get_current_z()
                    self.monitor_means.update(moy)
            self.run_trade()
        except Exception as e:
            print(f"{self.live['name']} err generale : {e}")

    def get_position_type(self, position):
        ls = BUY if position.type == MetaTrader5.POSITION_TYPE_BUY else SELL
        return ls

    def is_blocked(self):
        # --------------------------------------------------- calcul situation etc ...
        blocked = NONE
        recent_bricks = self.bricks.tail(3)
        # Conversion rapide uniquement sur ces quelques lignes
        times_ms = recent_bricks['time'].to_numpy().astype('int64') / 1_000_000
        recent_bricks['direction'] = np.where(recent_bricks['close_renko'] > recent_bricks['open_renko'], 1,-1)
        # np.where(recent_bricks['open_renko'] > recent_bricks['close_renko'], -1, 0))
        durations = np.diff(times_ms)
        # La durée de la toute dernière brique clôturée est le dernier élément du tableau
        last_duration = durations[-1]
        prev_duration = durations[-2]
        action = BUY if all(np.diff(recent_bricks['direction']) > 0) else SELL if all(np.diff(recent_bricks['direction']) < 0) else NONE
        # Votre seuil (ex: 60 secondes ou un pourcentage calculé globalement une seule fois au début)
        SEUIL_EMBALLEMENT = self.v_thresh
        if last_duration < 5000 and last_duration > 1500 and prev_duration > 5000 and action != NONE:
            # Neutralisation du signal de vente ou achat
            blocked = action
            current_time = self.bricks['time'].iloc[-1]
            last_duration /= 1000
            msg = (
                f"🚨 **ALERTE EMBALLEMENT MARCHÉ** 🚨\n"
                f"Symbole : {self.symbol} sens {'BUY' if action == 1 else 'SELL'} "
                f"⏱️ Brique bouclée en ** {last_duration:.1f} secondes ** "
                f"📅 Heure : {current_time}"
            )

            # Envoi effectif de l'alerte
            # (Astuce : en backtest, vous voudrez peut-être désactiver l'envoi réel pour ne pas spammer votre téléphone)
            print(f"{ROUGE}{msg}{RESET}")
            if not self.backtest_mode:
                send_telegram_alert(msg)
        return blocked

    def market_analysis(self, proba, z_indic, moy, z_means, trace=False):
        # ==================== calcul sens du marché =======================
        recent_probas = self.monitor_indic.get_partial(self.slope_window)    #get_adaptive_window_size(self.interval))
        slope = get_linear_slope(recent_probas)
        sensitivity = get_dynamic_sensitivity(self.slope_base, self.interval)
        # Si le marché est haussier (probas en baisse), on cherche une pente négative inférieure à un seuil dynamique
        trend_down_valid = slope < -sensitivity
        trend_up_valid = slope > sensitivity
        # et le z_means pour valider la situation
        is_strong_market_push = abs(z_means) > self.z_proba_min
        sens = define_sens(is_strong_market_push, trend_down_valid, trend_up_valid, z_means)
        if trace:
            clr = RESET if not is_strong_market_push else VERT if z_means < 0 else ROUGE
            cli = RESET if not is_strong_market_push else VERT if z_indic < 0 else ROUGE
            cln = RESET
            if self.live['name'] == 'pmxRKO':
                cln = JAUNE
            # _, mmy = self.monitor_means.get_current_z()
            print(f"{get_clean_timestamp()} {cln}{self.live['name']}{RESET} z_score {clr}{z_means:.3f}{RESET} & "
                  f"{cli}{z_indic:.3f}{RESET} "
                  f"moy {BLEU_CIEL}{moy:.3f}{RESET} "  #et {BLEU}{mmy:.3f}{RESET} "
                  f"up {ROUGE}{trend_up_valid}{RESET} dn {VERT}{trend_down_valid}{RESET} "
                  f"sty {sensitivity:.4f} slope {slope:.4f} "
                  # f"i_pente {VERT if i_pente < 0 else ROUGE}{i_pente:.3f}{RESET} "
                  f"ss {self.situation}/{sens} "
                  f"close {self.bricks['close'].iloc[-1]:.2f}")
        return sens, is_strong_market_push, trend_down_valid, trend_up_valid

    def run_trade(self):
        ind_cfg = self.cfg.get("indicators_and_filters", {})
        # Mode Backtest piloté par les briques pré-générées
        if not self.backtest_mode:
            # Les briques et le display global sont déjà injectés pas à pas par le backtester
            bricks_local = self.bricks.tail(self.minimum).copy()
            # --------------------------------------- La strategie ------------------------------
            opTrade = False
            onDisplay = False
            if self.renko_time is None or self.renko_time != self.bricks['time'].iloc[-1]:     #self.bricks.index[-1]:
                if self.renko_time is not None:
                    opTrade = True
                self.renko_time = self.bricks['time'].iloc[-1]     #self.bricks.index[-1]
                if self.parent:
                    print(
                        f"{JAUNE}--------------------- {datetime.now()}{RESET} {self.live['name']} start renko {self.renko_time} at {self.bricks['open_renko'].iloc[-1]:.2f}")
            self.current_time = datetime.now()
            if self.time_display is None or self.time_display + timedelta(seconds=self.interval) < self.current_time:
                self.time_display = self.current_time
                onDisplay = True
            #self.display = decision_bricks(self.bricks, self.cfg)
            self.display = add_indicators_optimized(bricks_local, self.cfg)
            self.display = choix_features_numba(self.display, self.cfg)
            positionsTotales = self.cl.get_positions_symbol(self.live['symbol'])
            self.positions = select_positions_magic(positionsTotales, self.live['magic'])
            if self.parent:
                try:
                    self.parent.update_display(
                        {"df": self.display.tail(13), "current_bid": self.ticks['bid'].iloc[-1], "strategy": self})
                except Exception as e:
                    print(f"err display 3 {e}")
            # Extraction des configurations d'indicateurs et filtres depuis config
            if not version_rnn:
                dj = decision_rates(self.df, ind_cfg)
                reg_window = ind_cfg.get("regression", {}).get("window", 18)
                segment = dj['close'].values[-reg_window:]  # Fenêtre de régression configurable
                # Calcul instantané des indicateurs
                pente, _, vol_log, r2, _ = fast_stats_single(segment)
                er = dj['er'].iloc[-1]  # Alignement parfait sur l'ER configurable (MQL5)
                self.to_follow(r2, ind_cfg)  # stop suiveur
                try:
                    sar, sar_vote, stoch, stoch_vote = get_last_decision(dj, vol_log, r2, ind_cfg['veto']['er_min'],
                                                                         False)
                    bbo = int(dj['sigo'].iloc[-1])
                    bbc = int(dj['sigc'].iloc[-1])
                    # .iloc[-1] pour la Series Pandas
                    # [-1] pour l'Array NumPy
                    # sar_val = sar.iloc[-1] if isinstance(sar, pd.Series) else sar[-1]
                    # stoch_val = stoch.iloc[-1] if isinstance(stoch, pd.Series) else stoch[-1]
                    extra = {
                        'sar': [int(sar), int(sar_vote)],
                        'stoch': [int(stoch), int(stoch_vote)],
                        'bbc': bbc
                    }
                except Exception as e:
                    print(f"bb err 2 {e}")
                    return
            else:
                # self.to_follow(None, ind_cfg)
                if TEST_PROBA and onDisplay:
                    proba = decision_ai(self.display, self.bricks, self.cfg, self.scaler, self.models)
                    if proba is None or len(proba) == 0:
                        print(f"proba test after decision ai is None or empty")
                    else:
                        proba = proba_final(proba, self._param.get("weights", None))
                        self.monitor_proba.update(proba[-1])
                    print(f"test proba {proba[-1]} close {self.display['close'].iloc[-1]} brick close {self.bricks['close'].iloc[-1]}")
            if not opTrade:
                return
            if self.display['time'].iloc[-1] == self.renko_time:
                self.display = self.display.iloc[:-1]
            self.all_probas = decision_ai(self.display, self.bricks, self.cfg, self.scaler, self.models)
            if self.regression:
                """
                z_rnn, z_tab = decision_monitor(self.monitor_rnn, [proba_rnn[-1]], self.monitor_tabicl,
                                                [proba_tabicl[-1]])
                self.proba = proba_final(z_rnn, z_tab, 0.7, False)
                # print(f"update 1 {type(proba)} {proba}")
                """
                self.monitor_indic.update(self.proba)
            else:
                self.proba = proba_final(self.all_probas, self._param.get("weights", None))
        else:
            opTrade = True
        # ------------------------- prises et application des décisions ---------------------
        proba, z_indic, z_means, moy = monitoring(self.proba, self.monitor_indic, self.monitor_means)

        # ============================================================================
        # NOUVELLE LOGIQUE HYBRIDE
        # ============================================================================
        if version_enhanced:
            # 1. Préparation des données pour la décision améliorée
            reg_window = self.cfg.get('market_regime', {}).get("regression_window", 14)
            # Segment des 14 dernières briques Renko
            y_seg = self.display['close'].iloc[-reg_window:].values
            slope, std, vol_log_pct, r2, er_val = fast_stats_single(y_seg)
            # 2. Collecte des probabilités de tous les modèles
            proba_dict = {}
            # Si on a plusieurs modèles, récupérer leurs prédictions
            if hasattr(self, 'models') and self.all_probas is not None and len(self.all_probas) > 0:
                proba_dict = self.all_probas
            else:
                proba_dict['default'] = [self.proba]
            # 3. Utilisation de la décision améliorée
            # Vérifier si on a assez de données pour les indicateurs
            # Appel à la décision hybride
            try:
                situation, regime = enhanced_decision(
                    proba_dict=proba_dict,
                    weights=self._param.get("weights", {}),
                    df=self.display,
                    param=self.cfg,
                    r2=r2, er=er_val,slope=slope,volatility=std,
                    zone_filter=self.zone_filter,
                    time_current=self.tickLast
                )
            except Exception as e:
                print(f"Erreur dans enhanced_decision: {e}")
                # Retour à l'ancienne méthode en cas d'erreur
                bornes = calcul_bornes(self.regression, self._param)
                if utils.config_utils.VDIRECT:
                    dest = [-2, -1, 0, 1, 2]
                else:
                    dest = [2, 1, 0, -1, -2]
                situation = calcul_situation(self.monitor_indic, self.bricks.tail(4), bornes, dest, True)
            # Debug: Afficher le régime détecté
            if self.parent:
                print(
                    f"{BLEU_CIEL}[HYBRID] Régime: {regime}, Situation: {situation}, Proba: {proba:.4f}{RESET}"
                    f" ssl {self.ssl} stp {self.stp}")
        else:
            bornes = calcul_bornes(self.regression, self._param)
            if utils.config_utils.VDIRECT:
                dest = [-2, -1, 0, 1, 2]
            else:
                dest = [2, 1, 0, -1, -2]
            situation = calcul_situation(self.monitor_indic, self.bricks.tail(4), bornes, dest, True)
        blocked = self.is_blocked()
        if blocked != NONE and abs(situation) == 2:
            situation = blocked * 2
        #sens, is_strong_market_push, trend_down_valid, trend_up_valid = self.market_analysis(proba, z_indic, moy, z_means, True)

        sigOpen = get_open_decision(situation)
        sigClose = 0
        lp = len(self.positions)
        if lp > 0:
            for position in self.positions:
                ls = self.get_position_type(position)
                # if not self.TimeisOpen():
                # self.close_one(ls, CLOSE, self.positions[0], True, "time")
                # return
                try:
                    if not self.regression:
                        if version_rnn:
                            sigClose = get_close_decision(ls, position.price_current,
                                                          position.price_open,
                                                          self.ssl, self.stp,
                                                          situation, sigOpen,
                                                          True)
                        else:
                            sigClose, sigOpen, co_pure, co_end = trading_decision(ls, position.price_open,
                                                                                  position.price_current,
                                                                                  self.display.tail(3), dj.tail(3),
                                                                                  proba,
                                                                                  self.ssl, self.stp,
                                                                                  self._param['threshold_buy'],
                                                                                  self._param['threshold_sell'],
                                                                                  self._param['close_buy'],
                                                                                  self._param['close_sell'],
                                                                                  pente, r2, er, extra=extra,
                                                                                  trace=True,
                                                                                  config=ind_cfg)
                    else:
                        if version_rnn:
                            sigClose = get_close_decision(ls, position.price_current,
                                                          position.price_open,
                                                          self.ssl, self.stp,
                                                          situation, sigOpen,
                                                          True)
                except Exception as e:
                    print(f"err Trading decision {e}")
                    return
                # print('signaux', sigClose, sigOpen, self.ppente)
                msg = 'norm' if sigClose == CLOSE else 'sltp' if sigClose == FCLOSE else 'sens'
                askClose = False
                """
                if sigClose != FCLOSE:
                    if ls * sens > 0 or (sigOpen * ls > 0 and blocked != ls):
                        continue
                    if (sens * ls < 0 or sigOpen * ls < 0 or
                            is_strong_market_push and ((ls==SELL and trend_down_valid) or (ls==BUY and trend_up_valid))):
                        msg = 'sens'
                        sigClose = CLOSE
                        askClose = True
                    else:
                        if sigClose != NONE:
                            #print(f"rejet close {sigClose}")
                            askClose = True
                else:
                    askClose = True
                """
                # comme avant
                if sigClose == FCLOSE and sigOpen * ls > 0:
                    sigClose = NONE			# cloture inutile serait ré ouverte immédiatement
                askClose = (sigClose > 3)
                if askClose:
                    # print("ask close")
                    self.close_locale(sigClose, msg)
                    self.situation = 0
        positionsTotales = self.cl.get_positions_symbol(self.live['symbol'])
        lpc = len(positionsTotales)
        self.positions = select_positions_magic(positionsTotales, self.live['magic'])
        lp = len(self.positions)
        if lp == 0:
            try:
                if not self.regression:
                    if version_rnn:
                        pass
                    else:
                        sigClose, sigOpen, co_pure, co_end = trading_decision(NONE,
                                                                              0.0,
                                                                              0.0,
                                                                              self.display.tail(3), dj.tail(3), proba,
                                                                              self.ssl,
                                                                              self.stp,
                                                                              self._param['threshold_buy'],
                                                                              self._param['threshold_sell'],
                                                                              self._param['close_buy'],
                                                                              self._param['close_sell'],
                                                                              pente, r2, er, extra=extra,
                                                                              trace=True, config=ind_cfg)
            except Exception as e:
                print(f"err trading decision 0 {self.live['name']}: {e}")
                return
            copen = (abs(sigOpen) == 4)
            if copen:
                sigOpen = SELL if sigOpen < 0 else BUY
            """
            if abs(sigOpen) == 1 and blocked == sigOpen:
                sigOpen = sigOpen * -1
                if sigOpen == sens or sens == NONE:
                    pass
                else:
                    return
            elif (sigOpen == SELL and trend_down_valid) or (sigOpen == BUY and trend_up_valid):
                return
            """
            # print(f"{self.live['name']} {VIOLET}opTrade {opTrade}{RESET}")
            if ((copen and not version_rnn) or opTrade) and abs(sigOpen) == 1:
                self.open_locale(sigOpen)
                self.situation = situation

    def lance(self):
        print(f"{datetime.now()} {self.live['name']} Début de pmxRko {self.live['symbol']}")
        Strategy.lance(self)

def get_adaptive_window_size(interval_seconds, target_window_seconds=30):
    """
    Calcule le nombre de points (briques ou échantillons) nécessaires
    pour couvrir une durée temporelle cible (ex: 30 secondes).
    """
    # On s'assure d'avoir au moins 3 points minimum pour faire une régression
    n_points = target_window_seconds / max(interval_seconds, 1)
    return max(int(round(n_points)), 3)

def define_sens(is_strong_market_push, trend_down_valid, trend_up_valid, z_means):
    if is_strong_market_push and trend_down_valid and z_means < 0:
        return BUY
    elif is_strong_market_push and trend_up_valid and z_means > 0:
        return SELL
    return NONE

import requests

def send_telegram_alert(message):
    """Envoie une alerte instantanée sur Telegram."""
    TOKEN = "8962407629:AAGvCinibIK3W8yvE3q-sTNPa3F-WQL3"
    CHAT_ID = "5665262737"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️ Erreur d'envoi Telegram : {e}")
        return False

"""
import os
from gtts import gTTS

def send_telegram_voice_alert(message):
	# Génère un message vocal et l'envoie sur Telegram.
	TOKEN = "8962407629:AAGvCinibIK3W8yvE3q-sTNPa3F-WQL3"
	CHAT_ID = "5665262737"

	# 1. Transformer le texte en fichier audio mp3 en français
	tts = gTTS(text=message, lang='fr', slow=False)
	audio_path = "alerte_emballement.mp3"
	tts.save(audio_path)

	# 2. Envoyer le fichier audio via l'API Telegram
	url = f"https://api.telegram.org/bot{TOKEN}/sendVoice"
	with open(audio_path, 'rb') as voice_file:
		files = {'voice': voice_file}
		data = {'chat_id': CHAT_ID, 'caption': "🚨 Alerte Emballement Marché"}
		try:
			requests.post(url, data=data, files=files, timeout=10)
		except Exception as e:
			print(f"Erreur d'envoi vocal : {e}")

	# 3. Nettoyer le fichier local
	if os.path.exists(audio_path):
		os.remove(audio_path)
"""

def check_telegram_acknowledgment(last_update_id=0):
    """
    Vérifie si un message (/ok ou autre) a été envoyé au bot.
    Renvoie True si un AR a été reçu, ainsi que le dernier update_id.
    """
    TOKEN = "8962407629:AAGvCinibIK3W8yvE3q-sTNPa3F-WQL3"
    CHAT_ID = "5665262737"
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=1"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            results = data.get("result", [])
            if results:
                for update in results:
                    update_id = update.get("update_id", 0)
                    message = update.get("message", {})
                    text = message.get("text", "").strip().lower()
                    # Si vous envoyez "/ok" ou "ok"
                    if text in ["/ok", "ok", "/vu", "vu"]:
                        return True, update_id
        return False, last_update_id
    except Exception as e:
        print(f"⚠️ Erreur lecture Telegram : {e}")
        return False, last_update_id

