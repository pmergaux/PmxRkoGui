# strategy/pmxrko.py
import os

import joblib
import numpy as np
import pandas as pd

from train.trainer import lgbm_predit, xgb_predict, lstm_predict_model, mlp_predict
from utils.model_utils import create_sequences_numba
from utils.scaler_utils import load_and_transform
from .base import Strategy
from utils.renko_utils import tick21renko
from decision.candle_decision import calculate_indicators, choix_features, calculate_japonais
from decision.trading_decision import trading_decision
from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE
from datetime import datetime, timedelta, time
from mt5linux import MetaTrader5
from live.connexion import select_positions_magic
import tensorflow as tf
from tensorflow.keras.models import load_model

class PmxRkoStrategy(Strategy):
    def __init__(self, parent, config):
        super().__init__(config)
        self.parent = parent
        self.bricks = None
        self.renko_size = config.get("parameters").get("renko_size", 20)
        self.display = None
        self.renko_time = None
        self.count_time = None
        self.debut = False
        self.minimum = config["lstm"]["lstm_seq_len"] * 4
        self.fopen = False
        self.model = None
        self.scaler = None

    def calcul_ai(self, trace=False):
        test_p = self.bricks[:-1]
        lstm = self.cfg["lstm"]
        target = self.cfg["target"]
        features = self.cfg["features"]
        features_cols = [] #features.copy()  # pour une distinction y compris des objets internes copie=copy.deepcopy(orig)
        for col in features:
            if len(col) > 1:
                features_cols.append(col)
        if target['target_include'] and not target["target_col"] in features_cols:
            features_cols.append(target["target_col"])
        X_test, y_test = self.create_sequences(test_p, features_cols, lstm['lstm_seq_len'], target)
        if len(X_test) == 0:
            return NONE
        # Normaliser
        #self.scaler.fit(X_test.reshape(-1, len(features_cols)))
        #X_train = self.scaler.transform(X_train.reshape(-1, len(features_cols)).reshape(X_train.shape)
        #X_val = self.scaler.transform(X_val.reshape(-1, len(features_cols)).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, len(features_cols))).reshape(X_test.shape)
        # Entraîner
        #model = train_model(X_train, y_train, X_val, y_val, config['lstm_units'], config['seq_len'])
        #model.save("models/temp.keras")
        # Prédire
        proba = self.model.predict(X_test, verbose=0).flatten()
        buy = proba > lstm['lstm_threshold_buy']
        sell = proba < lstm['lstm_threshold_sell']
        test_p['signal'] = np.where(buy, 1, np.where(sell, -1, 0))
        # Backtest normalement inutile juste pour info
        returns = np.diff(test_p[lstm['target_col']].values[-len(proba):])
        profit = 0
        trades = 0
        wins = 0
        for i in range(len(proba)):
            if i >= len(returns): break
            if buy[i]:
                profit += returns[i]
                trades += 1
                if returns[i] > 0: wins += 1
            elif sell[i]:
                profit -= returns[i]
                trades += 1
                if returns[i] < 0: wins += 1
        winrate = wins / trades if trades > 0 else 0
        score = profit * 1000 + winrate * 100
        print(f"Config: {self._param['renko_size']:.1f}, {lstm['lstm_seq_len']}, {lstm['lstm_units']} → Score: {score:.1f}")
        if trace:
            print("signal lstm\n", test_p['signal'].tail(-3))
        return test_p['signal'].iloc[-1]

    def decision_ai(self, trace=False):
        try:
            features_cols = self.cfg['features']
            print("feat", features_cols)
            target_cols = self.cfg['target']['target_col']
            print("tg", target_cols)
            if not isinstance(target_cols, list):
                target_cols = [target_cols]
            #target_type = self.cfg['target']['target_type']
            df = self.display[-256:].copy()
            try:
                X_test = load_and_transform(self.scaler, df[features_cols])
            except BaseException as e:
                print("err create Xtest", e)
            print("len Xtest", len(X_test) if X_test is not None else 'None')
            y_test = df[target_cols].to_numpy(dtype=np.float32)
            test_r = np.hstack([X_test, y_test])
            print("len Rtest", len(test_r) if test_r is not None else 'None')
            X_test_seq, _ = create_sequences_numba(test_r, self.cfg['lstm']['lstm_seq_len'], len(features_cols))
            print("len Xseq", len(X_test_seq) if X_test_seq is not None else 'None', self.cfg['lstm']['lstm_seq_len'])
            if 'LGBM' in self.version:
                proba = lgbm_predit(self.model, X_test)
            elif 'XGB' in self.version:
                proba = xgb_predict(self.model, X_test)
            elif 'LSTM' in self.version:
                proba = lstm_predict_model(self.model, X_test_seq)
            elif 'MLP' in self.version:
                proba = mlp_predict(self.model, X_test)
            else:
                proba = self.model.predict(X_test_seq, verbose=0).flatten()
        except BaseException as e:
            print(f"err prediction {self.version} ", e)
            return None
        if trace:
            print("pba", proba[-3:])
        return proba[-3:]

    def decision(self, trace=False):
        try:
            self.display = calculate_indicators(self.bricks, self.cfg)
            self.display = choix_features(self.display, self.cfg)
        except Exception as e:
            print(f"decision err {e}")
        jp = calculate_japonais(self.df, self.cfg)
        dc = self.display[['direction']].tail(3)
        if not 'sigc' in self.display.columns:
            self.display['sigc'] = NONE
        sc = self.display[['sigc']].tail(3)
        if not 'sigo' in self.display.columns:
            self.display['sigo'] = NONE
        so = self.display[['sigo']].tail(3)
        dd = pd.concat([dc, sc, so], axis=1)
        djd = jp[['direction']].tail(3)
        djc = jp[['sigc']].tail(3)
        djo = jp[['sigo']].tail(3)
        dj = pd.concat([djd, djc, djo], axis=1)
        if trace:
            print(dd)
            print(dj)
            # print sur une ligne             print(*dj['sigc'].to_list(), sep = '|')
        return dd, dj  # dc['direction'].iloc[-2], sc['sigc'].iloc[-2], so['sigo'].iloc[-2]

    def TimeisOpen(self):
        now = datetime.now().time()
        cls = time(22, 50, 00)
        opn = time(23, 30, 00)
        return now < cls or now > opn

    def _xTimeIsOpen(self):
        now = datetime.now()
        deb = datetime.now().replace(hour=23, minute=30)
        return now > deb

    def run(self):
        if self.debut:
            return
        try:
            if self.bricks is None:
                self.live['init_decal'] = 1000
                self.debut = True
                Strategy.run(self)
                try:
                    self.bricks = tick21renko(self.ticks, self.bricks, self.renko_size, 'bid')
                    if self.bricks is None or len(self.bricks) == 0:
                        raise Exception('bricks is None or empty')
                except Exception as e:
                    print(f'Rko err / bricks create: {e}')
                    self.tickLast = None
                    self.debut = False
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
            if len(self.bricks) > self.minimum + 15:
                self.bricks = self.bricks[-self.minimum-3:]
            if self.renko_time is None:
                self.renko_time = self.bricks.index[-1]
                print(f"{datetime.now()} {self.live['name']} start renko {self.renko_time} at {self.bricks['open_renko'].iloc[-1]:.2f}")
                self.decision()
                self.positions = select_positions_magic(self.cl.get_positions_symbol(self.live['symbol']),
                                                        self.live['magic'])
                try:
                    self.parent.update_display({"df": self.display.tail(13), "current_bid": self.ticks['bid'].iloc[-1], "strategy": self})
                except Exception as e:
                    print(f"err display 1 {e}")
                return
            tdelta=0
            self.positions = select_positions_magic(self.cl.get_positions_symbol(self.live['symbol']), self.live['magic'])
            lp = len(self.positions)
            if self.renko_time == self.bricks.index[-1]:
                if not self.fopen or lp > 0:
                    self.positions = select_positions_magic(self.cl.get_positions_symbol(self.live['symbol']),
                                                            self.live['magic'])
                    self.decision()
                    try:
                        self.parent.update_display({"df": self.display.tail(13), "current_bid": self.ticks['bid'].iloc[-1], "strategy": self})
                    except Exception as e:
                        print(f"err display 2 {e}")
                        return
                    if self.count_time is None:
                        return
                    tdelta = (self.ticks.index[-1] - self.count_time).total_seconds()
                    if tdelta < 3600:
                        return
                    #print("timedelta", tdelta)
            else:
                self.renko_time = self.bricks.index[-1]
                if self.count_time is not None:
                    self.count_time = self.bricks.index[-1]
                tdelta = 0
                self.fopen = False
                print(f"{datetime.now()} {self.live['name']} changement renko {self.renko_time} "
                      f"ex {'sell' if self.bricks['open_renko'].iloc[-2] > self.bricks['close_renko'].iloc[-2] else 'buy'}")
            dd, dj = self.decision(tdelta==0 and not self.fopen)
            dai = self.decision_ai(True)
            try:
                self.parent.update_display({"df": self.display.tail(13), "current_bid": self.ticks['bid'].iloc[-1], "strategy": self})
            except Exception as e:
                print(f"err display 3 {e}")
            if (tdelta != 0 or self.count_time is not None) and lp == 0:
                self.count_time = None   # cas d'un arret externe au prograùmùe
            ls = NONE
            if lp > 0:
                ls = BUY if self.positions[0].type == MetaTrader5.POSITION_TYPE_BUY else SELL
                if not self.TimeisOpen():
                    self.close_one(ls, CLOSE, self.positions[0], True, "time")
                    return
            signal = trading_decision(ls, self.positions[0].price_open, self.positions[0].price_current,
                                      dd, None, dai, self.ssl, self.stp,
                                      self._param['threshold_buy'], self._param['threshold_sell'])
            self.fopen = False
            if not self.TimeisOpen():                 return
            if signal == BUY or signal == SELL:
                self.stp = self.live['tp']
                self.live['tp'] = 0
                self.ssl = self.live['sl']
                self.live['sl'] = 0
                self.open(signal, 0)
                self.live['tp'] = self.stp
                self.live['sl'] = self.ssl
                self.futurClosed = NONE
                self.count_time = self.ticks.index[-1]
                return
            msg = 'norm' if signal == CLOSE else 'sltp' if signal == FCLOSE else ''
            if signal > 3:
                self.close_one(ls, signal, self.positions[0], True, msg)
                self.count_time = None
        except Exception as e:
            print(f"err generale {self.live['name']}: {e}")

    def lance(self):
        print(f"{datetime.now()} {self.live['name']} Début de pmxRko {self.live['symbol']}")
        Strategy.lance(self)


