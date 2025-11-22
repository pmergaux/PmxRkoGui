# strategy/base.py
import time
from datetime import datetime, timedelta

import pandas as pd
from PyQt6.QtCore import QThread, QTimer
from mt5linux import MetaTrader5

from live.connexion import Connexion, select_positions_magic
from live.periodic_timer import Periodic_Timer_Thread
from utils.rate_utils import ticks2rates, ticks22rates, Tf2Rs
from utils.utils import timeFrame2num, NONE, BUY, SELL, CLOSE, FCLOSE, connectMt5

spread_dict = {'ETHUSD':250, 'SOLUSD':300}
VERSION = '0.0.3'


class Strategy(QThread):
    def __init__(self, config):
        super().__init__()
        self.timer = QTimer()
        self.timer.timeout.connect(self.run)
        self.cfg = config
        self._param = config.get("parameters")
        self.df = None
        self.df_size = 60
        self.ticks = None
        self.tickLast = None
        self.live = config.get('live', {})
        self.symbol = self.live.get("symbol", "")
        self._period = self.initPeriod(self.live['timeframe'])
        self.cl = None  # À connecter
        self.ssens = NONE
        if self.live.get('lot_size') is None:
            self.live['lot_size'] = self.live['volume']
        self.decal = 0
        self.ssl = self.live.get("sl", 0)
        self.stp = self.live.get("tp", 0)
        self.last_open_time = None
        self.unit_pandas = ""
        self.num_str = ""
        self.mySocket = None
        self.spread = 0.0
        self.lastIndex = None
        self.ticket = 0
        self.positions = []
        self.futurClosed = NONE

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

    def open(self, sens, row):
        self.futurClosed = NONE
        if sens == NONE:
            print(f"{datetime.now()} {self.live['name']} open: sens=NONE, aucune position ouverte")
            return False
        #print(f"{datetime.now()} {self.live['name']} Tentative d'ouverture position : sens={sens}, row={row}")
        if not self.cl.market_order_trade_execution(sens, 0, self.live, 0, 0):
            print(f"{datetime.now()} {self.live['name']} Échec ouverture position : {'buy' if sens == BUY else 'sell'} à row {row}")
            self.ssens = NONE
            self.last_open_time = None
            return False
        time.sleep(0.5)
        self.positions = select_positions_magic(self.cl.get_positions_symbol(self.symbol), self.live['magic'])
        if len(self.positions) > 0:
            self.lastIndex = pd.Timestamp(self.positions[0].time, unit='s').floor(Tf2Rs[self.live['timeframe']])
            self.lastIndex = self.df.index[row] if row > 0 else self.lastIndex
            self.ssens = sens
            self.last_open_time = datetime.now()
            print(
                f"{datetime.now()} {self.live['name']} open {'buy' if sens == BUY else 'sell'} row {row}/{len(self.df)} à {self.lastIndex}")
        else:
            print(f"{datetime.now()} {self.live['name']} Échec ouverture position : {'buy' if sens == BUY else 'sell'} à row {row}")
            self.ssens = NONE
            self.last_open_time = None
            return False
        return True

    def to_be_opened(self, sens, trace=True):
        if spread_dict[self.symbol] < self.spread:
            return NONE
        return sens

    def to_be_closed(self, sens, position, trace=True):
        rc = NONE
        msg = ''
        ls = BUY if position.type == MetaTrader5.POSITION_TYPE_BUY else SELL
        cp = position.price_current
        pc = position.price_open
        if hasattr(self, 'bbs'):
            bup = self.bbs.bollinger_hband().iat[-1]
            bdn = self.bbs.bollinger_lband().iat[-1]
            bec = (bup - bdn) * self._param.get('niveau', 0.9)/2   # = size * 0.9 = 10% par rapport à  la moyenne
            ext = (ls == BUY and cp > bup) or (ls == SELL and cp < bdn)
            out = (ls == BUY and cp < bdn) or (ls == SELL and cp > bup)
            inn = (ls == BUY and cp < bup - bec) or (ls == SELL and cp > bdn + bec)
            inx = (ls == BUY and cp > bdn + bec) or (ls == SELL and cp < bup - bec)
            if self.futurClosed == NONE:
                self.futurClosed = ls if out else -ls if ext else NONE
            elif self.futurClosed != NONE:
                if self.futurClosed == ls:
                    if out:
                        rc = FCLOSE
                        msg = 'out_lim'
                        return rc, msg
                    if ext:
                        self.futurClosed = -ls
                    elif inx:
                        self.futurClosed = NONE
                else:
                    if out:
                        self.futurClosed = ls
                    elif inn:
                        rc = FCLOSE
                        msg = 'blim'
                        return rc, msg
        if VERSION == '0.0.2' or VERSION == '0.0.3':
            # === stop loss suiveur ====
            if self._param.get('suivi', 0) != 0 and self.ssl != 0:
                SL = (pc - self.ssl * ls) if self.ssl < 0 else pc
                situ = (cp - pc) * ls
                suivi = self._param['suivi']
                if False:
                    print(
                        f"{datetime.now()} {self.live['name']} CP={'%.3f' % cp} PC={'%.3f' % pc} SL={'%.3f' % SL} situ={'%.3f' % situ} sens={'buy' if ls == 1 else 'sell'} suivi {'%.2f' % suivi} ssl={'%.3f' % self.ssl}")
                if situ > 0:
                    TP = (pc + self.stp * ls) if self.stp != 0 else 0
                    SL = (pc - self.ssl * ls) if self.ssl < 0 else pc
                    c1 = (situ > max(suivi * 0.66, self.spread) and self.ssl >= 0)
                    c2 = (self.ssl < 0 and situ > suivi * 1.5 - self.ssl)
                    c3 = ((TP - (SL + suivi * ls)) * ls > 0 if TP != 0 else True)
                    if (c1 or c2) and c3:
                        self.ssl = (self.ssl - suivi) if self.ssl < 0 else -suivi * 0.6
                        if False:
                            print(
                                f"{datetime.now()} {self.live['name']} new ssl: {'%.3f' % self.ssl} for sens={'buy' if ls == 1 else 'sell'}, suivi={suivi}")
        # === SL et TP ===
        TP = (cp - (pc + self.stp * ls)) * ls if self.stp != 0 else 0
        SL = ((pc - self.ssl * ls) - cp) * ls if self.ssl != 0 else 0
        pp, bal = 0.0, 0.1
        if VERSION == '0.0.2' and self._param.get('risk', 0.0) != 0:
            pp = -position.profit
            """
            # a mettre après if..
            elif isinstance(self, bt.Strategy):
                    bal = bt.Strategy(self).broker.getcash()
            """
            if self.cl is not None:
                bal = self.cl.account_info().balance * (self._param['risk'] / 100)
            else:
                pp, bal = 0.0, 0.1
        if VERSION == '0.0.2':
            topen_arrondi = pd.Timestamp(self.positions[0].time, unit='s').floor(Tf2Rs[self._param['timeframe']])
            if self.df.index[-5] >= topen_arrondi:
                if ls == BUY and (self.df.iloc[-5]['high'] > self.df.iloc[-4]['high'] > self.df.iloc[-3]['high'] > self.df.iloc[-2]['high']):
                    rc = FCLOSE
                    msg = 'down'
                elif ls == SELL and (self.df.iloc[-5]['low'] < self.df.iloc[-4]['low'] < self.df.iloc[-3]['low'] < self.df.iloc[-2]['low']):
                    rc = FCLOSE
                    msg = 'up'
        #print(f"to be closed sens {ls} sl {SL:.2f} tp {TP:.2f}")
        if rc == NONE and ((sens != ls and sens != NONE) or TP > 0 or SL > 0 or pp > bal):
            rc = FCLOSE
            if TP > 0:
                msg = 'tp'
            elif SL > 0:
                msg = 'sl'
            elif pp > bal:
                'risk'
            else:
                msg = 'sens'
        return rc, msg

    def close_all(self, sens, positions, msg):
        for position in positions:
            ls = BUY if position.type == MetaTrader5.POSITION_TYPE_BUY else SELL
            if not self.close_one(ls, sens, position, True, msg):
                return False
            """
            if msg[0:5] == 'cntrl':
                self.ssens = NONE
                p = msg.find('/')
                row = int(msg[6:p])
                if len(self.df) - row < 20:
                    sens = -1*sens
                    self.open(sens, row)
            """
        return True

    def close_one(self, ls, sens, position, trace=False, msg=None):
        if self.ticket == position.ticket or position.ticket == 0:
            return True
        if self.cl.mt5.positions_get(ticket=position.ticket) is None:
            print(f"{datetime.now()} {self.live['name']} No positions on {position.symbol} ticket {position.ticket} error code={self.cl.mt5.last_error()}")
            return True
        if sens != CLOSE and sens != FCLOSE:
            cp = position.price_current
            pc = position.price_open
            TP = (cp - (pc + self.stp*ls)) * ls if self.stp != 0 else 0
            SL = ((pc - self.ssl*ls) - cp) * ls if self.ssl != 0 else 0
            if self._param.get('risk', 0.0) != 0:
                bal = self.cl.account_info().balance*(self._param['risk']/100)
                pp = -position.profit
            else:
                pp, bal = 0.0, 0.1
        else:
            TP = SL = 0
            pp, bal = 0.0, 0.1
        if sens == CLOSE or sens == FCLOSE or (sens != ls and sens != NONE) or TP > 0 or SL > 0 or pp > bal:
            rt = self.cl.close_one(position, self.lastIndex, trace, msg)
            if rt:
                print(datetime.now(), self.live['name'], 'closed', 'sell' if ls == SELL else 'buy',
                      'asked', 'sell' if sens == SELL else 'buy' if sens == BUY else 'close', msg)
                if msg == 'norm':
                    if TP > 0:
                        msg = 'tp'
                    elif SL > 0:
                        msg = 'sl'
                    elif pp > bal:
                        'risk'
                    else:
                        msg = 'sens'
                self.ticket = position.ticket
                self.ssens = NONE
                self.last_open_time = None
            else:
                self.ticket = 0
                print(datetime.now(), self.live['name'], 'err close', 'sell' if ls == SELL else 'buy',)
            return rt
        return True

    def run(self):
        try:
            dn = datetime.now() + timedelta(hours=3)
            if self.tickLast is None:
                dc = datetime.now() - timedelta(hours=self._param.get('init_decal', 240))
                self.ticks = self.cl.get_ticks_from(self.live['symbol'], dc, dn)
                if self.ticks is None or len(self.ticks) == 0:
                    return
                self.df = ticks2rates(self.ticks, self.live['timeframe'], 'bid')
                self.tickLast = self.ticks.index[-1]
            else:
                db = datetime.fromtimestamp(self.tickLast.timestamp())
                new_ticks = self.cl.get_ticks_from(self.live['symbol'], db, dn)
                if new_ticks is not None and len(new_ticks) > 0:
                    while len(new_ticks) > 0 and new_ticks.index[0] <= self.tickLast:
                        new_ticks = new_ticks.iloc[1:]
                    if len(new_ticks) > 0:
                        _, self.df = ticks22rates(self.df, new_ticks, self._period)
                        self.tickLast = new_ticks.index[-1]
                self.ticks = new_ticks
        except Exception as e:
            print(f"[BASE] Erreur run: {e}")

    def lance(self):
        if self.live['symbol'] is None:
            print('symbol is None')
            return
        # Connexion MT5
        mt5 = connectMt5()
        print(f"{datetime.now()} MT5 initialisé")
        clg = Connexion(mt5, 0, "")
        clg.login()
        self.cl = clg
        print(f"{datetime.now()} Connexion MT5 établie")
        """
        try:
            self.mySocket = socket(AF_INET, SOCK_STREAM)
            print("Socket successfully created.")
        except error as err:
            print("socket creation failed with error %s" % err)
            exit()

        self.mySocket.connect((host, port))
        # self.mySocket.setblocking(False)
        print("The socket has successfully connect to google on port =", host, port)
        receive_thread = threading.Thread(target=self.discours)
        receive_thread.start()
        """
        print(datetime.now(),'start', self.live['name'], self.live['symbol'])
        #self.timer = Periodic_Timer_Thread(interval=period, function=self.run, comment=self.live['symbol'])
