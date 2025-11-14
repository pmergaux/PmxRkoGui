import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from dataclasses import dataclass
#from typing import List, Dict, Any, Union

from mt5linux import MetaTrader5

from utils.utils import CLOSE, NONE, BUY, SELL


@dataclass
class Position:
    ticket: int=0
    time_open: datetime=datetime(1970, 1, 1)
    time_close: datetime=datetime(1970,1, 1)
    delta: timedelta=timedelta(seconds=0)
    type: int=0
    magic: int=0
    identifier: int=0
    reason: str=''
    volume: float=0.0
    price_open: float=0.0
    sl: float=0.0
    tp: float=0.0
    price_current: float=0.0
    swap: float=0.0
    profit: float=0.0
    symbol: str=''
    comment: str=''

class BackTrader:
    def __init__(self, param, data):
        self.data = data
        self.xfin = self.data.index[-1]
        self.p = param
        self.ti = None
        self.position = None
        self.positionsList = []
        self.start_date = None

    def next(self):
        pass

    def set_result(self):
        if not self.positionsList:
            return {
                'total_profit': 0.0,
                'total_trades': 0,
                'sharpe': 0.0,
                'drawdown': 0.0,
                'win_rate': 0.0
            }

        # Extract profits and timestamps
        profits = [pos.profit for pos in self.positionsList]
        total_trades = len(profits)
        total_profit = sum(profits)

        # Simplified Drawdown Calculation
        initial_capital = 10000.0  # Adjust based on your strategy
        portfolio_value = initial_capital
        peak = initial_capital
        max_drawdown = 0.0

        for profit in profits:
            portfolio_value += profit
            peak = max(peak, portfolio_value)
            if peak > 0:  # Avoid division by zero
                drawdown = (portfolio_value - peak) / peak * 100
                max_drawdown = min(max_drawdown, drawdown)

        # Win Rate Calculation
        winning_trades = sum(1 for profit in profits if profit > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Sharpe Ratio Calculation (simplified, using numpy for returns)
        returns = np.array([profit / self.p['volume'] for profit in profits])
        risk_free_rate = 0.02  # Annual risk-free rate (2%)
        periods_per_year = 252 * 24 * 60  # Minutes per year
        mean_return = np.mean(returns) * periods_per_year if len(returns) > 0 else 0.0
        std_return = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
        sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0.0

        return {
            'total_profit': np.round(total_profit, 2),
            'total_trades': total_trades,
            'sharpe': sharpe_ratio,
            'drawdown': max_drawdown,  # Negative percentage
            'win_rate': win_rate
        }

    def close_last(self):
        if self.position is not None:
            self.close_back(BUY if self.position.type==MetaTrader5.POSITION_TYPE_BUY else SELL, CLOSE, 'last')

    def exec(self):
        pd.set_option('mode.chained_assignment', None)
        # pd.options.mode.copy_on_write = True
        ddeb = self.data.index[0]
        dfin = self.p['renko_start']
        interval = self.p.get('interval', 0)
        #print(f"n° {self.p['num']} renko_start={dfin} à {self.xfin} pour size={self.p['renko_size']:.2f} interval={interval}")
        if ddeb < dfin:
            try:
                self.ti = self.data.loc[ddeb:dfin]
            except Exception as e:
                print(f"err Backtrader init data {e}")
                return
        self.next()
        if interval == 0:
            self.tick_idx = len(self.ti)  #  index flou self.data.index.get_loc(pd.to_datetime(dfin))
            while self.tick_idx < len(self.data):
                self.ti = self.data.iloc[self.tick_idx:self.tick_idx + 1]
                self.next()
                self.tick_idx += 1
        else:
            ddeb = dfin
            dfin = ddeb + timedelta(seconds=interval)
            if dfin > self.xfin:
                dfin = self.xfin
            #print(f"n° {self.p['num']} ddeb={ddeb} à {dfin}")
            while ddeb < dfin:
                self.ti = self.data.loc[ddeb:dfin]
                #print(f"{datetime.now()} n° {self.p['num']} prkT data xfin={self.xfin} deb={ddeb} fin={dfin} len ti={len(self.ti) if self.ti is not None else 0}")
                if self.ti is None or len(self.ti) == 0:
                    ddeb = dfin
                    dfin = dfin + timedelta(seconds=interval)
                    if dfin > self.xfin and ddeb >= self.xfin:
                        break
                    continue
                while len(self.ti) > 0 and self.ti.index[0] <= ddeb:
                    self.ti.drop(self.ti.index[0], inplace=True)
                if len(self.ti) > 0:
                    self.next()
                ddeb = dfin
                if ddeb >= self.xfin:
                    break
                dfin = ddeb + timedelta(seconds=interval)
                if dfin > self.xfin:
                    dfin = self.xfin
        self.close_last()

        r = self.set_result()
        #print("---------------------------------------------------------- RESULTS : ", r)
        return self.positionsList, r

    def close_back(self, ls, sens, msg=''):
        if sens == CLOSE:
            #print(datetime.now(), self.p['name'], 'closed', 'sell' if ls == SELL else 'buy',
            #      'asked', 'sell' if sens == SELL else 'buy' if sens == BUY else 'close', msg)
            profit = np.round((self.position.price_current - self.position.price_open) * self.position.volume * ls, 2)
            #print(f"n° {self.p['num']} {self.ti.index[-1]} cloture cp={self.position.price_current} op={self.position.price_open} vol={self.position.volume} pf={profit:.2f}")
            self.position.time_close = self.ti.index[-1]
            self.position.delta = self.position.time_close - self.position.time_open
            self.position.profit = profit
            self.position.reason = msg
            self.positionsList.append(self.position)
            self.position = None

    def open_back(self, so, sl=0.0, tp=0.0):
        if so != NONE and not self.position:
            price = np.round(self.ti.iloc[-1]['ask'] if so==BUY else self.ti.iloc[-1]['bid'], 2)
            self.position = Position()
            self.position.symbol = self.p['symbol']
            self.position.ticket = len(self.positionsList) + 1
            self.position.time_open = self.ti.index[-1]
            sl_value = (sl if sl > 0 else self.p['sl'] if self.p['sl'] > 0 else 0) * so
            tp_value = (tp if tp > 0 else self.p['tp'] if self.p['tp'] > 0 else 0) * so
            self.position.volume = self.p['volume']
            self.position.price_open = price
            self.position.price_current = price
            self.position.sl = (price - sl_value) if sl_value > 0 else 0
            self.position.tp = (price + tp_value) if tp_value > 0 else 0
            self.position.type = MetaTrader5.POSITION_TYPE_BUY if so == BUY else MetaTrader5.POSITION_TYPE_SELL
            #print(
            #    f"n° {self.p['num']} {self.ti.index[-1]} Ouverture position {'buy' if so == BUY else 'sell'} : price={price}, sl={sl_value}, tp={tp_value}")

