# strategy/pmxrko.py
import os

import joblib
import numpy as np
import pandas as pd

from .base import Strategy
from utils.renko_utils import tick21renko
from decision.candle_decision import calculate_indicators, choix_features, calculate_japonais
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

    def create_sequences(self, df, features_cols, seq_len, target: dict):
        X, y = [], []
        data = df.copy()
        data = data.dropna()
        target_col = target['target_col']
        try:
            values = data[target_col].values
            features = data[features_cols].values
            target_type = target['target_type']
            for i in range(len(data) - seq_len):
                X.append(features[i:i + seq_len])
                if target_type == 'direction':
                    y.append(1 if values[i + seq_len] > values[i + seq_len - 1] else 0)
                elif target_type == 'value':
                    y.append(values[i+seq_len-1])
                else:
                    y.append((values[i + seq_len] - values[i + seq_len - 1]) / values[i + seq_len - 1])
        except Exception as e:
            print("pmxRko err create sequence ", e)
        return np.array(X), np.array(y)

    # si necessaire car load_model fourni de model déjà entraîné
    def train_model(self, X_train, y_train, X_val, y_val, units, seq_len):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(seq_len, 5)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)
        return model

    def pred_model(self):
        pass

    def load_model_scaler(self):
        self.scaler = joblib.load("models/simple_scaler.pkl") if os.path.exists("models/simple_scaler.pkl") else None
        if not self.scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        self.model = load_model('models/simple.keras')

    def decision_ai(self, trace=False):
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

    def decision(self, trace=False):
        try:
            self.display = calculate_indicators(self.bricks, self.cfg)
            self.display = choix_features(self.display, self.cfg)
        except Exception as e:
            print(f"decision err {e}")
        japon = calculate_japonais(self.df, self.cfg)
        line = self.display[['MACD_line']].tail(3)
        sign = self.display[['MACD_signal']].tail(3)
        diff = self.display[['MACD_hist']].tail(3)
        dc = self.display[['direction']].tail(3)
        if not 'sigc' in self.display.columns:
            self.display['sigc'] = NONE
        sc = self.display[['sigc']].tail(3)
        if not 'sigo' in self.display.columns:
            self.display['sigo'] = NONE
        so = self.display[['sigo']].tail(3)
        dd = pd.concat([dc, sc, so, line, sign, diff], axis=1)
        if not 'sigc' in japon.columns:
            japon['sigc'] = NONE
        dj = japon[['sigc']].tail(3)
        if trace:
            print(dd)
            print(*dj['sigc'].to_list(), sep = '|')
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
                self.decal = self.live.get('init_decal', 1000)
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
            if self.renko_time == self.bricks.index[-1]:
                if not self.fopen:
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
                print(f"{datetime.now()} {self.live['name']} changement renko {self.renko_time} "
                      f"ex {'sell' if self.bricks['open_renko'].iloc[-2] > self.bricks['close_renko'].iloc[-2] else 'buy'}")
            self.positions = select_positions_magic(self.cl.get_positions_symbol(self.live['symbol']), self.live['magic'])
            lp = len(self.positions)
            dd, dj = self.decision(tdelta==0 and not self.fopen)
            dai = self.decision_ai(True)
            try:
                self.parent.update_display({"df": self.display.tail(13), "current_bid": self.ticks['bid'].iloc[-1], "strategy": self})
            except Exception as e:
                print(f"err display 3 {e}")
            if (tdelta != 0 or self.count_time is not None) and lp == 0:
                self.count_time = None   # cas d'un arret externe au prograùmùe
            ls = NONE
            sdc, ssc, sso = dd['direction'].iloc[-2], dd['sigc'].iloc[-2], dd['sigo'].iloc[-2]
            pdc, psc, pso = dd['direction'].iloc[-3], dd['sigc'].iloc[-3], dd['sigo'].iloc[-3]
            ss = NONE
            if lp > 0:
                ls = BUY if self.positions[0].type == MetaTrader5.POSITION_TYPE_BUY else SELL
                if not self.TimeisOpen():
                    self.close_one(ls, CLOSE, self.positions[0], True, "time")
                    return
                # inversion sens ou tp ou sl ou risk etc...
                ss, msg = self.to_be_closed(ls, self.positions[0],False)
                if ss == NONE:
                    msg = 'norm'
                    if tdelta > 3600:
                        #ss = FCLOSE   # a revoir
                        msg = 'tOut'
                if ssc == 4 or ss == FCLOSE or (ls != sso and sso != NONE):
                    sc = CLOSE
                    if ss != NONE:
                        sc = ss
                    self.close_one(ls, sc, self.positions[0], True, msg)
                    self.count_time = None
                self.positions = select_positions_magic(self.cl.get_positions_symbol(self.live['symbol']), self.live['magic'])
                lp = len(self.positions)
            djc = dj['sigc'].iloc[-2]
            self.fopen = False
            if not self.TimeisOpen():                 return 0
            if lp == 0 and sso != NONE and (ssc != psc or (psc == ssc and ssc != 4)) and (ss != FCLOSE or (ss == FCLOSE and ls != sso)):
                if (sso == BUY and djc == BUY) or (sso == SELL and djc == SELL):
                    print(f"{datetime.now()} {self.live['name']} sensO={sso}")
                    self.stp = self.live['tp']
                    self.live['tp'] = 0
                    self.ssl = self.live['sl']
                    self.live['sl'] = 0
                    self.open(sso, 0)
                    self.live['tp'] = self.stp
                    self.live['sl'] = self.ssl
                    self.futurClosed = NONE
                    self.count_time = self.ticks.index[-1]
                else:
                    self.fopen = True   # revenir
        except Exception as e:
            print(f"err generale {self.live['name']}: {e}")

    def lance(self):
        print(f"{datetime.now()} {self.live['name']} Début de pmxRko {self.live['symbol']}")
        Strategy.lance(self)

    def calculate_japonais(self, df, cfg):
        pass

