# strategy/pmxrko.py
import pandas as pd

from .base import Strategy
from utils.renko_utils import tick21renko
from decision.candle_decision import calculate_indicators, choix_features, calculate_japonais
from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE
from datetime import datetime, timedelta, time
from mt5linux import MetaTrader5
from live.connexion import select_positions_magic

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
        self.minimum = 50
        self.fopen = False

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
                self.decal = self.live.get('init_decal', 240)
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
            if len(self.bricks) > self.minimum + 9:
                self.bricks = self.bricks[-self.minimum:]
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
            if not self.TimeisOpen():
                return
            if lp == 0 and sso != NONE and (ssc != psc or (psc == ssc and ssc != 4)) and (ss != FCLOSE or (ss == FCLOSE and ls != sso)):
                if (sso == BUY and djc != 6) or (sso == SELL and djc != 7):
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

