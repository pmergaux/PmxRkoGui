# strategy/base.py
from datetime import datetime, timedelta
from utils.rate_utils import ticks2rates, ticks22rates
from utils.utils import timeFrame2num, NONE, BUY, SELL, CLOSE


class Strategy:
    def __init__(self, param):
        self._param = param
        self.df = None
        self.ticks = None
        self.tickLast = None
        self._period = self.initPeriod(param['timeframe'])
        self.cl = None  # À connecter
        self.ssens = NONE

    def initPeriod(self, timeframe):
        unit, num = timeFrame2num(timeframe)
        if unit == 'm': return 60 * int(num)
        if unit == 'h': return 3600 * int(num)
        if unit == 'd': return 86400 * int(num)
        return 60

    def change_param(self):
        period = self.initPeriod(self._param['timeframe'])
        if period != self._period:
            self.df = None
            self.ticks = None
            self.tickLast = None
            self._period = period

    def run(self, symbol):
        try:
            dn = datetime.now() + timedelta(hours=3)
            if self.tickLast is None:
                dc = datetime.now() - timedelta(hours=self._param.get('init_decal', 240))
                self.ticks = self.cl.get_ticks_from(symbol, dc, dn)
                if self.ticks is None or len(self.ticks) == 0:
                    return
                self.df = ticks2rates(self.ticks, self._param['timeframe'], 'bid')
                self.tickLast = self.ticks.index[-1]
            else:
                db = datetime.fromtimestamp(self.tickLast.timestamp())
                new_ticks = self.cl.get_ticks_from(symbol, db, dn)
                if new_ticks is not None and len(new_ticks) > 0:
                    while len(new_ticks) > 0 and new_ticks.index[0] <= self.tickLast:
                        new_ticks = new_ticks.iloc[1:]
                    if len(new_ticks) > 0:
                        self.df = ticks22rates(self.df, new_ticks, self._period)
                        self.tickLast = new_ticks.index[-1]
        except Exception as e:
            print(f"[BASE] Erreur run: {e}")

