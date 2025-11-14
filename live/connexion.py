import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# from mt5linux import MetaTrader5

from utils.utils import sens_lib, BUY, SELL, TF2MT, BUY_STOP, SELL_STOP, CLOSE, mt5


# point 2 # decimales
def p2d(p):
    d = 0
    if p < 1:
        while p < 1:
            d += 1
            p *= 10
    return d

profit = {}


class Connexion(object):

    def __init__(self, mt5,
                 login=None,  # no du compte
                 password=None,  # pass du compte
                 maxposition=5,  # limitation du nb de positions ouvertes
                 mt5_path=None  # chemin du terminal.exe
                 ):

        self.mt5 = mt5
        self.connected = False
        self.login_id = login
        self.password = password
        self.mt5_path = mt5_path
        self.maxposition = maxposition

    def login(self):
        if self.connected:
            if self.mt5_path is not None:
                if not self.mt5.initialize(path=self.mt5_path):
                    print("MT5 chemin：{}".format(self.mt5_path))
                    print("initialisation self.mt5 le client a échoué")
                    self.mt5.shutdown()
                    self.connected = False
                    return False
            else:
                if not self.mt5.initialize():
                    print("initialisation self.mt5 le client a échoué")
                    print("Veuillez essayer après avoir spécifié le chemin self.mt5")
                    self.mt5.shutdown()
                    self.connected = False
                    return False
            return True
        # display data on the MetaTrader 5 package
        # print("MetaTrader5 package author: ", self.mt5.__author__)
        # print("MetaTrader5 package version: ", self.mt5.__version__)
        # establish connection to the MetaTrader 5 terminal
        if self.mt5_path is not None:
            if not self.mt5.initialize(path=self.mt5_path):
                print("MT5 chemin：{}".format(self.mt5_path))
                print("initialisation self.mt5 le client a échoué")
                self.mt5.shutdown()
                return False
        else:
            if not self.mt5.initialize():
                print("initialisation self.mt5 le client a échoué")
                print("Veuillez essayer après avoir spécifié un chemin self.mt5")
                self.mt5.shutdown()
                return False
            else:
                print('MT5 init')

        # self.mt5_path = self.mt5.terminal_info()[19]

        # Demander l'état et les paramètres de la connexion
        # print("Client self.mt5 initialisé avec succès, chemin：{}".format(mt5_path))
        # Obtenir des informations sur MetaTrader 5 données de version
        # print(mt5.version())

        if self.login_id is None or self.login_id == 0:
            print("login without account, but terminal account")
            account_info = self.mt5.account_info()
            print("name   = {}".format(account_info.name))
            print("login  = {}".format(account_info.login))
            print("server = {}".format(account_info.server))
            self.connected = True
            return True

        authorized = self.mt5.login(self.login_id, password=self.password)
        if authorized:
            print("Les informations utilisateur sont les suivantes：")
            account_info = self.mt5.account_info()
            print("name   = {}".format(account_info.name))
            print("login  = {}".format(account_info.login))
            print("server = {}".format(account_info.server))
            self.connected = True
        else:
            print('LOGIN FAILED!!!')
            self.mt5.shutdown()
            return False
        return True

    def shutdown(self):
        self.mt5.shutdown()
        self.connected = False

    def account_info(self):
        return self.mt5.account_info()

    def avoid_multiple_positions(self, POSITION, name, symbol):
        # THIS FUNCTION RESTRICTS THE BOT TO ONLY HAVE OPEN MAXPOSITION OF THE SAME INSTRUMENT PER STRATEGY
        # ALSO IT CLOSES AN EXISTING POSITION IF STRATEGY GIVES OPPOSITE OF ONGOING OPEN POSITION
        if not (POSITION == BUY or POSITION == SELL):
            return
        if not self.login():
            return None
        positions = self.mt5.positions_get(symbol)
        position_count = 0
        for open_position in positions:
            if open_position.comment.find(name) != -1:
                self.close_one(open_position)
            else:
                position_count = position_count + 1
        if position_count >= self.maxposition:
            action = 'block'
        else:
            action = 'noblock'
        return action

    def ticks2frame(self, df_ticks):
        if df_ticks is None:
            print("ticks2frame", self.mt5.last_error())
            return None
        # print(type(df_ticks))
        df_dfo = pd.DataFrame(df_ticks)
        df_dfo = df_dfo.set_index('time_msc', drop=True)
        df_dfo.index = pd.to_datetime(df_dfo.index, unit='ms')
        return df_dfo

    def rates2frame(self, npa):
        if npa is None:
            print("rates2frame", self.mt5.last_error())
            return None
        df = pd.DataFrame(npa)
        df = df.set_index('time', drop=True)
        df.index = pd.to_datetime(df.index, unit='s')
        return df

    def get_rates_pos(self, symbol: str, timeframe: str, quant: int, start=0):
        if not self.login():
            return None
        npa = self.mt5.copy_rates_from_pos(symbol, TF2MT[timeframe], start, quant)
        return self.rates2frame(npa)

    def get_rates_from(self, symbol, timeframe, date_from, date_to):
        if not self.login():
            return None
        npa = self.mt5.copy_rates_range(symbol, TF2MT[timeframe], date_from, date_to)
        return self.rates2frame(npa)

    def get_ticks_pos(self, symbol, size):
        if not self.login():
            return None
        df = datetime.now() + timedelta(hours=1)
        df_ticks = self.mt5.copy_ticks_from(symbol, df, size, self.mt5.COPY_TICKS_ALL)
        return self.ticks2frame(df_ticks)

    def get_ticks(self, symbol, date_from, size):
        if not self.login():
            return None
        df_ticks = self.mt5.copy_ticks_from(symbol, date_from, size, self.mt5.COPY_TICKS_ALL)
        return self.ticks2frame(df_ticks)

    def get_ticks_from(self, symbol, date_from, date_to):
        if not self.login():
            return None
        try:
            df_ticks = self.mt5.copy_ticks_range(symbol, date_from, date_to, self.mt5.COPY_TICKS_ALL)
        except Exception as e:
            print('err get ticks from',date_from, date_to, e)
            return None
        return self.ticks2frame(df_ticks)

    def get_ticks_dd(self, symbol, date_from, date_to, size):
        if not self.login():
            return None
        df = self.get_ticks_from(symbol, date_from, date_to)
        if len(df) > size:
            df = df[-size:]
        return df

    def get_tick_info(self, symbol):
        self.login()
        return self.mt5.symbol_info_tick(symbol)

    def get_symbol_info(self, symbol):
        self.login()
        return self.mt5.symbol_info(symbol)

    def get_symbol_select(self, symbol, select=True):
        self.login()
        return self.mt5.symbol_select(symbol, select)

    # Pour les positions get, il faut preciser symbol, ticket ou group, ... sk rien donne tout, mais ? rien
    def get_positions_symbol(self, symbol):
        self.login()
        positions = self.mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return positions

    def get_orders_symbol(self, symbol):
        if not self.login():
            return None
        return self.mt5.orders_get(symbol=symbol)

    def get_positions_strategy(self, param: dict):
        return select_positions(self.get_positions_symbol(param['symbol']), param['name'])


    def get_orders_strategy(self, param: dict):
        self.login()
        orderg = self.get_orders_symbol(param['symbol'])
        orders = []
        if orderg is not None and len(orderg) > 0:
            nam = param['name']
            # print('get p', nam, len(orderg))
            for order in orderg:
                if order.comment.find(nam) == -1:
                    continue
                orders.append(order)
        # print(len(orders))
        return orders

    def market_order_trade_execution(self, POSITION, price, param: dict, sl: float = 0.0, tp: float = 0.0):
        """
        mql5 order types
        ENUM_ORDER_TYPE:
            ORDER_TYPE_BUY ; Market Strategy.BUY order
            ORDER_TYPE_SELL : Market Strategy.SELL order
            ORDER_TYPE_BUY_LIMIT : Strategy.BUY Limit pending order
            ORDER_TYPE_SELL_LIMIT : Strategy.SELL Limit pending order
            ORDER_TYPE_BUY_STOP : Strategy.BUY Stop pending order
            ORDER_TYPE_SELL_STOP : Strategy.SELL Stop pending order
            ORDER_TYPE_BUY_STOP_LIMIT : Upon reaching the order price, a pending Strategy.BUY Limit order is placed at the StopLimit price
            ORDER_TYPE_SELL_STOP_LIMIT : Upon reaching the order price, a pending Strategy.SELL Limit order is placed at the StopLimit price
            ORDER_TYPE_CLOSE_BY : Order to close a position by an opposite one
        """
        if not self.login():
            return
        if param['lot_size'] == 0 and not calc_volume(self, param):
            return False
        symbol = param['symbol']
        point = self.mt5.symbol_info(symbol).point
        dec = p2d(point)
        TP = param['tp']
        SL = param['sl']
        volume = float(param['volume'])
        deviation = 5
        comment = param['name']
        magic = param['magic']
        if POSITION == BUY or POSITION == BUY_STOP:
            order_type = self.mt5.ORDER_TYPE_BUY if POSITION == BUY else self.mt5.ORDER_TYPE_BUY_STOP
            action = self.mt5.TRADE_ACTION_DEAL if POSITION == BUY else self.mt5.TRADE_ACTION_PENDING
            if price == 0:
                price = self.mt5.symbol_info_tick(symbol).ask
            price = np.double(np.round(price, dec))
            if SL != 0:
                if sl == 0:
                    sl = np.double(price - SL)
                else:
                    sl = np.double(price - sl)
            else:
                sl = 0.0
            if TP != 0:
                if tp == 0:
                    tp = np.double(price + TP)
                else:
                    tp = np.double(price + tp)
            else:
                tp = 0.0
            request = {
                "action": action,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }
            # print('CNX', request)
            result = self.mt5.order_send(request)
            # print(result)
            if result is None:
                print(f"{datetime.now()} Buy error. order_send() by {comment} : {symbol} {volume} lots at {price} with deviation={deviation} points")
                print(self.mt5.last_error())
                return False
            elif self.health_check(result) == 'pass':
                print(f"{datetime.now()} Buy OK. order_send() by {comment} : {symbol} {volume} lots at {price} with deviation={deviation} points")
                return True
            return True
        elif POSITION == SELL or POSITION == SELL_STOP:
            order_type = self.mt5.ORDER_TYPE_SELL if POSITION == SELL else self.mt5.ORDER_TYPE_SELL_STOP
            action = self.mt5.TRADE_ACTION_DEAL if POSITION == SELL else self.mt5.TRADE_ACTION_PENDING
            if price == 0:
                price = self.mt5.symbol_info_tick(symbol).bid
            if SL != 0:
                if sl == 0:
                    sl = np.double(price + SL)
                else:
                    sl = np.double(price + sl)
            else:
                sl = 0.0
            if TP != 0:
                if tp == 0:
                    tp = np.double(price - TP)
                else:
                    tp = np.double(price - tp)
            else:
                tp = 0.0
            request = {
                "action": action,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }
            # print('CNX', request)
            result = self.mt5.order_send(request)
            if result is None:
                print(f"{datetime.now()} Sell error. order_send() by {comment} : {symbol} {volume} lots at {price} with deviation={deviation} points")
                print(self.mt5.last_error())
                return False
            elif self.health_check(result) == 'pass':
                print(f"{datetime.now()} Sell OK. order_send() by {comment} : {symbol} {volume} lots at {price} with deviation={deviation} points")
                return True
            return True
        elif POSITION == CLOSE:
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) > 0:
                for open_position in positions:
                    if open_position.comment.find(comment) == -1:
                        continue
                    self.close_one(open_position)
                return True
            else:
                return False
        else:
            raise 'POSITION TYPE IS INVALID FOR MARKET ORDERS!'

    def close_one(self, open_position, lastIndex=None, trace=False, cas=None):
        symbol = open_position.symbol
        ticket = open_position.ticket
        if ticket == 0:
            raise 'Position with no ticket'
        strat = open_position.comment
        magic = open_position.magic
        volume = open_position.volume
        pf = open_position.profit + open_position.swap
        price = 0
        order_type = 0
        # action = self.mt5.TRADE_ACTION_CLOSE_BY  # uniquement si on close en utilisant une autre position
        action = self.mt5.TRADE_ACTION_DEAL
        if open_position.type == self.mt5.POSITION_TYPE_BUY:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(symbol).bid
        elif open_position.type == self.mt5.POSITION_TYPE_SELL:
            order_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(symbol).ask
        comment = 'closed for ' + symbol + ' ' + strat
        deviation = 5
        ls = BUY if open_position.type == self.mt5.POSITION_TYPE_BUY else SELL
        request = {
            "action": action,
            "symbol": symbol,
            "position": ticket,
            "volume": volume,
            "price": price,
            "type": order_type,
            "deviation": deviation,
            "comment": comment,
            "magic": magic,
            # type_time=mt5.ORDER_TIME_GTC,
            # type_filling=mt5.ORDER_FILLING_FOK
        }
        self.login()
        result = self.mt5.order_send(request)
        txt = ''
        if cas is not None:
            if cas[0:5] == 'cntrl':
                cas = ';cntrl;'+cas[6:]
            else:
                cas = ';'+cas
        else:
            cas = ';'
        motif = ''
        rt = True
        price_close = 0
        commis = 0
        profit = 0
        if result is not None:
            if self.health_check(result) == 'pass':
                position_deals = self.mt5.history_deals_get(position=ticket)
                while position_deals is None or len(position_deals) < 2:
                    time.sleep(1)
                    position_deals = self.mt5.history_deals_get(position=ticket)
                try:
                    position_deal = position_deals[1]
                    print('cloture ?', position_deal)
                    price_close = position_deal.price
                    commis = position_deal.commission
                    profit = position_deal.profit
                except Exception as e:
                    print("err recup deal ", e)
                # profit[strat] += pf
                motif = 'Close OK. order_send(): by'
                print("{} {} Close OK. order_send(): symbol {} profit {} balance {}".
                      format(datetime.now(),strat, symbol, np.round(pf, 2),
                             np.round(self.mt5.account_info().balance,2)))
        else:
            motif = 'Close error. order_send(): by'
            print(f"{datetime.now()} Close error. order_send(): by {comment} ")
            rt = False
        if trace:
            point = self.mt5.symbol_info(symbol).point
            dec = p2d(point)
            txt = (f'{datetime.now()};{motif};{strat};{symbol};{sens_lib[ls + 3]};'
                   f'{datetime.fromtimestamp(open_position.time)};{lastIndex};'
                   f'{volume};'
                   f'{np.round(open_position.price_open, dec)};'
                   f'{np.round(open_position.price_current, dec)};'
                   f'{np.round(price, dec)};'
                   f'{np.round(price_close, dec)};'
                   f'{commis:.2f};'
                   f'{pf:.2f};{profit:.2f}{cas}')
            file_path = Path(strat+'-'+symbol+'.csv')
            exi = file_path.exists()
        # Ouvre le fichier en mode ajout ('a' pour ajout)
            with open(strat+'-'+symbol+'.csv', 'a') as f:
                if not exi:
                    f.write('date;opération;stratégie;symbol;sens;date ouv.;date signal;volume;prix ouv;prix actuel;prix ferm;prix_clot;commis;profit;p reel;messag\n')
                f.write(txt+'\n')
        return rt

    def close_order(self, order):
        # print('close order')
        strat = order.comment
        action = self.mt5.TRADE_ACTION_REMOVE
        comment = 'closed for ' + order.symbol + ' ' + strat
        deviation = 5

        request = dict(action=action,
                       # symbol=order.symbol,
                       order=order.ticket,
                       # type=order.type,
                       # magic=order.magic,
                       # comment = order.comment,
                       # deviation=deviation,
                       # type_time=mt5.ORDER_TIME_GTC,
                       # type_filling=mt5.ORDER_FILLING_FOK
                       )
        self.login()
        result = self.mt5.order_send(request)
        if result is None:
            print("Close error. order_send(): by {} symbol {}".
                  format(comment, order.symbol))
        elif self.health_check(result) == 'pass':
            print("Close OK. order_send(): by {} symbol {}"
                  .format(strat, order.symbol))

    def last_error(self):
        return self.mt5.last_error()

    def change_sl_tp(self, position, sl, tp):
        symbol = position.symbol
        position_type = position.type
        current_price = position.price_current
        open_price = position.price_open
        # point = self.mt5.symbol_info(symbol).point
        # dec = p2d(point)
        if position_type == self.mt5.POSITION_TYPE_BUY:
            sl_price = current_price - sl
            tp_price = current_price + tp
        else:
            sl_price = current_price + sl
            tp_price = current_price - tp
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": symbol,
            "sl": sl_price,
            "tp": tp_price,
            "magic": position.magic
        }
        result = self.mt5.order_send(request)
        if self.health_check(result) == 'pass':
            print(f"SL TP Changed!!! {position.symbol},  Strategy: {position.comment}")

    def change_sl(self, position, sl):
        self.change_sl_tp(position, sl, position.tp)

    def simplestoploss(self, open_position, timeframe):
        position_type = open_position.type
        symbol = open_position.symbol
        current_price = open_position.price_current
        open_price = open_position.price_open
        point = self.mt5.symbol_info(symbol).point
        dec = p2d(point)
        order_timeframe = timeframe
        sl_price = open_position.sl
        tp_price = open_position.tp
        # print(order_timeframe)
        # on recupere 3 candles calculees selon la periodicite
        past_data = self.mt5.copy_rates_from_pos(symbol, order_timeframe, 0, 3)
        # print('current close: ', past_data['close'].iloc[1], ' previous close: ', past_data['close'].iloc[0])
        if position_type == self.mt5.POSITION_TYPE_BUY:  # LONG ORDER
            if (past_data[-2]['close'] > past_data[-3]['close']) and open_position.profit > 0:
                dsl = open_price - sl_price
                if tp_price == 0:
                    tp_price = np.double(open_price + (open_price - sl_price) / 2)
                nsl = current_price - sl_price
                if nsl < dsl or nsl < 0:
                    return
                new_sl = np.double(np.round(sl_price + nsl - abs(dsl), dec))
                if new_sl <= sl_price or new_sl > tp_price or abs(new_sl - sl_price) / point < 100:
                    return
                print(open_position.symbol + ': new sl:', new_sl, ' old SL: ', sl_price)
                # on fixe TP s'il ne l'était pas
                request = {
                    "action": self.mt5.TRADE_ACTION_SLTP,
                    "position": open_position.ticket,
                    "symbol": open_position.symbol,
                    "sl": new_sl,
                    "tp": tp_price,
                    "magic": open_position.magic
                }
                result = self.mt5.order_send(request)
                if self.health_check(result) == 'pass':
                    print(open_position.symbol, ' SL Changed!!! Strategy: ')
                # if result['retcode'] != self.mt5.TRADE_RETCODE_DONE:
                #    print("2. order_send failed, retcode={}".format(result['retcode']))
        elif position_type == self.mt5.POSITION_TYPE_SELL:  # SHORT ORDER
            if (past_data[-2]['close'] < past_data[-3]['close']) and open_position.profit > 0:
                # print('change detected in w', open_position.symbol)
                if tp_price == 0:
                    tp_price = np.double(open_price - (sl_price - open_price) / 2)
                dsl = sl_price - open_price
                nsl = sl_price - current_price
                if nsl < dsl or nsl < 0:
                    return
                new_sl = np.double(np.round(sl_price - (nsl - abs(dsl)), dec))
                if new_sl >= sl_price or new_sl < tp_price or abs(new_sl - sl_price) / point < 100:
                    return
                print(open_position.symbol + ': new sl:', new_sl, ' old SL: ', sl_price)
                # on fixe TP s'il ne l'était pas
                request = {
                    "action": self.mt5.TRADE_ACTION_SLTP,
                    "position": open_position.ticket,
                    "symbol": open_position.symbol,
                    "sl": new_sl,
                    "tp": tp_price,
                    "magic": open_position.magic
                }
                result = self.mt5.order_send(request)
                if self.health_check(result) == 'pass':
                    print(open_position.symbol, ' SL Changed!!! Strategy: ')
                # if result['retcode'] != self.mt5.TRADE_RETCODE_DONE:
                #    print("2. order_send failed, retcode={}".format(result['retcode']))
        else:
            raise Exception('Error in order_type')

    def logout(self):
        print(f"{datetime.now()} Connexion.logout appelé")
        if self.mt5:
            self.mt5.shutdown()
        return True

    def calc_margin(self, action, symbol, price):
        if action == BUY:
            sens = self.mt5.ORDER_TYPE_BUY
        elif action == SELL:
            sens = self.mt5.ORDER_TYPE_SELL
        else:
            return 0.0
        return self.mt5.order_calc_margin(sens, symbol, 1, price)

    def get_history_order(self, symbol, date_from, date_to):
        return self.mt5.history_orders_get(date_from, date_to, group=symbol)

    def get_history_deal(self, symbol, date_from, date_to):
        return self.mt5.history_deals_get(date_from, date_to, group=symbol)

    def health_check(self, result):
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            print("2. order_send failed, retcode={}".format(result.retcode))
            # request the result as a dictionary and display it element by element
            result_dict = result._asdict()
            print("   {}={}".format('comment', result_dict['comment']))
            if result.retcode != 10013:
                return
            for field in result_dict.keys():
                print("   {}={}".format(field, result_dict[field]))
                # if this is a trading request structure, display it element by element as well
                if field == "request":
                    traderequest_dict = result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
        else:
            return 'pass'

    def daily_losslimit_check(self, from_date: datetime, to_date: datetime):
        position_history_orders = self.mt5.history_orders_get(from_date, to_date)
        # print(history_orders)
        df = pd.DataFrame(list(position_history_orders),
                          columns=position_history_orders[0]._asdict().keys())
        # df.drop(['time_expiration','type_time','state','position_by_id',
        # 'reason','volume_current','price_stoplimit','sl','tp'], axis=1, inplace=True)
        df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
        df['time_done'] = pd.to_datetime(df['time_done'], unit='s')
        print(df[['time_done', 'time_setup', 'symbol']])

def select_positions(positiong, name):
    # print(len(positiong))
    if positiong is None or len(positiong) == 0:
        return []
    positions = []
    # print('get p', nam, len(positiong))
    for position in positiong:
        if position.comment.find(name) == -1:
            continue
        positions.append(position)
    # print(len(positions))
    return positions

def select_positions_magic(positiong, magic):
    # print(len(positiong))
    if positiong is None or len(positiong) == 0:
        return []
    positions = []
    # print('get p', nam, len(positiong))
    for position in positiong:
        if position.magic == magic:
            positions.append(position)
    # print(len(positions))
    return positions


class RiskManagement_v1(object):
    def login(self):
        # display data on the MetaTrader 5 package
        print("MetaTrader5 package author: ", self.mt5.__author__)
        print("MetaTrader5 package version: ", self.mt5.__version__)
        # establish connection to the MetaTrader 5 terminal
        if not self.mt5.initialize(self.PATH):
            print("initialize() failed, error code =", self.mt5.last_error())
            quit()

        authorized = self.mt5.login(self.login_id, password=self.password)
        if not authorized:
            print('LOGIN FAILED!!!')
            self.mt5.shutdown()
            quit()
        else:
            print("Login with account: ", str(self.login_id), " successfull!!!")

    def __init__(self, mt5,
                 login,
                 password,
                 PATH,
                 daily_limit=0.05,
                 monthly_limit=0.10):
        self.mt5 = mt5
        self.login_id = login
        self.password = password
        self.PATH = PATH
        self.login()
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit

def TradeSizeOptimized(client, param: dict):
    symbol = param['symbol']
    info_symbol = client.get_info_symbol(symbol)
    price = info_symbol.ask
    margin = client.calc_margin(BUY, symbol, price)
    if margin is None:
        return 0.0
    if margin <= 0.0:
        return 0.0
    lot = client.account_info.margin_free * param['maximumRisk'] / margin
    #    lot = np.double(lot)
    # --- calculate number of losses orders without a break
    decreaseFactor = param['decreaseFactor']
    if decreaseFactor > 0:
        # --- select history for access
        dc = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
        orders = list(client.get_history_deal(symbol, 0, dc + 3600))
        if orders is None or len(orders) == 0:
            return 0.0
        losses = 0  # number of losses orders without a break
        orders.reverse()
        for order in orders:
            ticket = order.ticket
            if ticket == 0:
                print("HistoryDealGetTicket failed, no trade history")
                break
            # --- check symbol
            # --- check magic
            # --- check profit
            profit = order.profit
            if profit > 0.0:
                break
            if profit < 0.0:
                losses += 1
        # ---
        if losses > 1:
            lot = lot - lot * losses / decreaseFactor
    # --- normalize and check limits
    stepvol = info_symbol.volume_step
    lot = stepvol * int(lot / stepvol)

    minvol = info_symbol.volume_min
    if lot < minvol:
        lot = minvol
    maxvol = info_symbol.volume_max
    if lot > maxvol:
        lot = maxvol
    # --- return trading volume
    return np.double(lot)

def calc_volume(client, param):
    if param['lot_size'] == 0:
        v = TradeSizeOptimized(client, param)
        if v == 0:
            return False
        param['volume'] = v
    return True
