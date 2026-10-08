# strategy/pmxCandle.py
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .base import Strategy
from decision.candle_decision import calculate_indicators, choix_features, calculate_japonais, calculate_stochastic, \
    calculate_sar, calculate_atr_4sl, calculate_vwap_zscore, calculate_efficiency_ratio, fast_stats_single, is_market_exploding
from decision.trading_decision import trading_decision, decision_ai, decision_rates, get_last_decision
from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE
from mt5linux import MetaTrader5

class PmxCandleStrategy(Strategy):
    def __init__(self, parent, config):
        super().__init__(config)
        self.parent = parent
        self.display = None
        self.display_rnn = None
        self.last_candle_time = None
        self.last_rnn_time = None
        self.minimum = config.get("lstm", {}).get("lstm_seq_len", 24)
        
        # Gestion du deuxième timeframe pour le RNN
        from utils.utils import timeFrame2num
        self.timeframe_rnn = self.live.get('timeframe_rnn', self.live['timeframe'])
        unit, num = timeFrame2num(self.timeframe_rnn)
        if unit == 'm': self._period_rnn = 60 * int(num)
        elif unit == 'h': self._period_rnn = 3600 * int(num)
        else: self._period_rnn = self._period

    def run(self):
        # On récupère les données OHLC de base via la classe de base
        Strategy.run(self)
        
        if self.df is None or len(self.df) < self.minimum + 20:
            return

        # Mise à jour du DataFrame RNN si différent
        from utils.rate_utils import ticks2rates
        if self.timeframe_rnn != self.live['timeframe']:
            self.df_rnn = ticks2rates(self.ticks, self.timeframe_rnn, 'bid')
        else:
            self.df_rnn = self.df

        # 1. Calcul des indicateurs sur le timeframe RNN (pour l'IA)
        # On ne le fait que si on a une nouvelle bougie RNN
        current_rnn_time = self.df_rnn.index[-1]
        if self.last_rnn_time is None or current_rnn_time != self.last_rnn_time:
            self.last_rnn_time = current_rnn_time
            try:
                # On utilise une copie de config pour forcer les features RNN
                # (on pourrait imaginer des features différentes pour le RNN)
                self.display_rnn = calculate_indicators(self.df_rnn, self.cfg)
                self.display_rnn = choix_features(self.display_rnn, self.cfg)
            except Exception as e:
                print(f"[{self.live['name']}] Erreur calcul RNN indicators: {e}")

        # 2. Vérification nouvelle bougie standard pour la décision
        current_time = self.df.index[-1]
        is_new_candle = (self.last_candle_time is None or current_time != self.last_candle_time)
        
        if not is_new_candle:
            return

        self.last_candle_time = current_time
        
        # Calcul des indicateurs standard
        try:
            self.display = calculate_indicators(self.df, self.cfg)
            self.display = choix_features(self.display, self.cfg)
        except Exception as e:
            print(f"[{self.live['name']}] Erreur calcul indicateurs standard: {e}")
            return

        # 2. Récupération des positions
        self.positions = self.mt5.positions_get(magic=self.live['magic'])
        lp = len(self.positions)
        ls = NONE
        if lp > 0:
            ls = BUY if self.positions[0].type == MetaTrader5.POSITION_TYPE_BUY else SELL

        # 3. Calcul de l'IA (RNN) sur le timeframe RNN
        # On récupère la dernière probabilité disponible du timeframe RNN
        proba = decision_ai(self.display_rnn, self.cfg, self.scaler, self.model, False)
        
        if proba is None:
            return

        # Extraction des configurations d'indicateurs et filtres depuis config
        ind_cfg = self.cfg.get("indicators_and_filters", {})
        stoch_cfg = ind_cfg.get("stoch", {'window': 21, 'slow': 5, 'signal': 5})
        sar_cfg = ind_cfg.get("sar", {'window': 0.02, 'maxi': 0.2})
        zscore_cfg = ind_cfg.get("zscore", {'window': 24})
        er_window = ind_cfg.get("er", {}).get("window", 18)
        reg_window = ind_cfg.get("regression", {}).get("window", 18)
        veto_cfg = ind_cfg.get("veto", {})

        # 4. Autres indicateurs pour la décision finale (Stoch, SAR, etc.)
        dj = decision_rates(self.df, {
            'stoch': stoch_cfg, 
            'sar': sar_cfg, 
            'er': {'window': er_window}, 
            'zscore': zscore_cfg
        })
        
        # Stats rapides sur le dernier segment pour filtrage (R2, Pente)
        segment = dj['close'].values[-reg_window:] # Fenêtre de régression configurable
        pente, std, vol_log, r2, _ = fast_stats_single(segment)
        er = dj['er'].iloc[-1]  # Alignement parfait sur l'ER configurable (MQL5)
        
        # Mise à jour graphique
        try:
            self.parent.update_display({
                "df": self.display.tail(20), 
                "current_bid": self.ticks['bid'].iloc[-1] if self.ticks is not None else self.df['close'].iloc[-1], 
                "strategy": self
            })
        except Exception as e:
            print(f"err display candle {e}")

        # 5. Décision de trading
        # On passe extra avec les indicateurs techniques formatés via get_last_decision
        sar, sar_vote, stoch, stoch_vote = get_last_decision(dj, vol_log, r2)
        extra = {
            'sar': [int(sar), int(sar_vote)],
            'stoch': [int(stoch), int(stoch_vote)],
            'bbc': int(dj['sigc'].iloc[-1]) if 'sigc' in dj.columns else NONE
        }
        
        sigClose, sigOpen, co_pure, co_end = trading_decision(
            ls, 
            self.positions[0].price_open if lp > 0 else 0.0,
            self.positions[0].price_current if lp > 0 else 0.0,
            self.display.tail(3), 
            dj.tail(3), 
            proba, 
            self.ssl, 
            self.stp,
            self.cfg['parameters']['threshold_buy'], 
            self.cfg['parameters']['threshold_sell'],
            self.cfg['parameters']['close_buy'], 
            self.cfg['parameters']['close_sell'], 
            pente, 
            r2, 
            er, 
            extra=extra, 
            trace=True,
            veto_config=veto_cfg
        )

        # 6. Exécution des ordres
        if lp > 0:
            # Gestion de la clôture
            if sigClose > 3 or (sigOpen != NONE and sigOpen != ls):
                msg = 'norm' if sigClose == CLOSE else 'sltp' if sigClose == FCLOSE else 'sens'
                self.close_one(ls, sigClose, self.positions[0], True, msg)
        
        if lp == 0:
            # Gestion de l'ouverture
            if sigOpen == BUY or sigOpen == SELL:
                # On ajuste temporairement TP/SL pour l'ouverture
                # (La classe base les gère)
                self.open(sigOpen, 0)

        # 7. Gestion de la Sortie Rapide (Explosion & Trailing ATR)
        if lp > 0:
            try:
                # On calcule un ATR rapide sur les dernières bougies
                df_atr = self.df.tail(20).copy()
                high_low = df_atr['high'] - df_atr['low']
                tr = high_low # Simplifié
                atr = tr.rolling(14).mean().iloc[-1]
                
                # Détection d'explosion (Z-Score de la volatilité)
                if is_market_exploding(self.df.tail(40), threshold=2.5):
                    po = self.positions[0].price_open
                    cp = self.positions[0].price_current
                    profit = (cp - po) * ls
                    
                    # Si on est en profit et que le marché s'emballe
                    if profit > atr:
                        # On resserre le SL à 1.5 * ATR du prix actuel pour "coller" au mouvement
                        dist = 1.5 * atr
                        # Le nouveau SL relatif à l'entrée
                        # Pour un BUY (ls=1), sl_price = cp - dist => ssl = po - sl_price = po - cp + dist
                        # Pour un SELL (ls=-1), sl_price = cp + dist => ssl = sl_price - po = cp + dist - po
                        new_ssl = (po - cp + dist) if ls == BUY else (cp + dist - po)
                        
                        # On ne resserre que si le nouveau SL est meilleur (plus proche du prix actuel que l'ancien)
                        if new_ssl < self.ssl:
                            print(f"[{self.live['name']}] 🚀 EXPLOSION détectée ! Trailing ATR activé. Nouveau SL à {dist:.5f}")
                            self.ssl = new_ssl
            except Exception as e:
                print(f"err fast exit logic: {e}")

    def lance(self):
        print(f"{datetime.now()} {self.live['name']} Début de pmxCandle {self.live['symbol']}")
        Strategy.lance(self)
