import argparse
import json
import platform

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import math

from PyQt5.QtWidgets import QFileDialog, QWidget
from sklearn.linear_model import LinearRegression
import sys
from mt5linux import MetaTrader5


MAXQ = 4        # nb max des signaux de trading
host = '127.0.0.1'
port = 10101
code = 'utf-8'
cmds = {'start': 'start on or stop off', 'alert': 'alert on or off',
        'block': 'on or stop or off',
        'info': 'print param',
        'use': 'use a strategy','vol':'volume',
        'lot': 'volume', 'risk': '% risk', 'gap': 'border value in indicator',
        'cld': '% period to close', 'opd': '% period to open', 'param': 'fixe les paramètres'}
cmd2 = ('on', 'off')

pd.options.mode.copy_on_write = True


BUY_STOP = 3
BUY_CONT = 2
BUY = 1
SELL = -1
SELL_CONT = -2
SELL_STOP = -3
CLOSE = 5
FCLOSE = 6
NONE = 0
UP = 1
DOWN = -1

TF2S = {
    'M1': 60,
    'M5': 300,
    'M10': 600,
    'M15': 900,
    'M30': 1800,
    'H1': 3600,
    'H4': 14400,
    'D1': 86400,
    'W1': 604800,
    'MN1': 2592000
}

TF2MT = {'1m':MetaTrader5.TIMEFRAME_M1,
        '2m':MetaTrader5.TIMEFRAME_M2,
        '3m':MetaTrader5.TIMEFRAME_M3,
        '4m':MetaTrader5.TIMEFRAME_M4,
        '5m':MetaTrader5.TIMEFRAME_M5,
        '6m':MetaTrader5.TIMEFRAME_M6,
        '10m':MetaTrader5.TIMEFRAME_M10,
        '15m':MetaTrader5.TIMEFRAME_M15,
        '20m':MetaTrader5.TIMEFRAME_M20,
        '30m':MetaTrader5.TIMEFRAME_M30,
        '1h':MetaTrader5.TIMEFRAME_H1,
        '2h':MetaTrader5.TIMEFRAME_H2,
        '3h':MetaTrader5.TIMEFRAME_H3,
        '4h':MetaTrader5.TIMEFRAME_H4,
        '6h':MetaTrader5.TIMEFRAME_H6,
        '8h':MetaTrader5.TIMEFRAME_H8,
        '12h':MetaTrader5.TIMEFRAME_H12,
        '1d':MetaTrader5.TIMEFRAME_D1,
        '1w':MetaTrader5.TIMEFRAME_W1,
        }


sens_lib = ['sell_stop', 'sell_cont', 'sell', 'none', 'buy', 'buy_cont', 'buy_stop']

mt5 = None

# connect to the server
# Détecter le système d'exploitation
os_name = platform.system()
"""
if os_name == 'Windows':
    import MetaTrader5 as mt5
else:  # Linux
"""
def connectMt5(host='localhost', port=18812):
    global mt5
    # Initialisation MT5
    if os_name == 'Windows':
        pass
        # if not mt5.initialize():
        #    raise Exception("Erreur d'initialisation MT5 (Windows) : " + str(mt5.last_error()))
    else:
        try:
            mt5 = MetaTrader5(host=host, port=port)  # Valeurs par défaut
            if not mt5.initialize():
                raise Exception("Erreur d'initialisation MT5 (Linux) : " + str(mt5.last_error()))
            mt5.execute("import numpy as np")  # Importe NumPy dans le namespace distant
        except Exception as e:
            raise Exception(f"Erreur lors de l'initialisation mt5linux : {e}")

    return mt5


def timeFrame2num(tf):
    # Try to match pattern: optional number + unit letters
    match = re.fullmatch(r"(\d+)?([A-Za-z]+)", tf)
    if not match:
        raise ValueError(f"Input frequency '{tf}' does not match expected format (e.g., '1m', '4h', '1D').")
    num_str, unit_pandas_raw = match.groups()
    unit_pandas = unit_pandas_raw.lower()  # Normalize unit to lower case for map lookup
    return unit_pandas, num_str

def linear_regression_sklearn(x, y) -> dict:
    """Simple Linear Regression in Scikit Learn for two 1d arrays for
    environments with the sklearn package."""

    # X = pd.DataFrame(x)
    X = x.reshape(-1, 1)
    Y = y   #y.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y=Y)
    r = lr.score(X, y=Y)
    b, a = lr.intercept_, lr.coef_[0]
    result = {
        "a": a, "b": b, "r": r,
        #"t": r / np.sqrt((1 - r * r) / (x.size - 2)),
        "line": b + a * x}
    return result


def to_number(x, strict=False):
    try:
        return int(x) if x.isdigit() else float(x)
    except:
        return None if strict else x

# ===========================================================================================
# Fonction pour parser les plages depuis les args (format: start,stop,step)
def parse_range_args(arg, as_int=False):
    try:
        start, stop, step = map(float, arg.split(','))
        range_values = np.arange(start, stop + step, step)
        if as_int:
            return range_values.astype(int)
        return range_values
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Format de plage invalide: {arg}. Utilisez start,stop,step (ex. 32.5,39.5,0.1)")


# Gestionnaire de signal pour Ctrl+C
def signal_handler(sig, frame):
    print(f"{datetime.now()} Signal d'interruption reçu (Ctrl+C)")
    global strategy
    if 'strategy' in globals() and strategy:
        print(f"{datetime.now()} Appel à strategy.stop()")
        try:
            strategy.arret()
        except Exception as e:
            print(f"{datetime.now()} Erreur lors de l'arrêt de strategy : {type(e).__name__}: {e}")
    if 'mt5' in globals():
        print(f"{datetime.now()} Arrêt de MT5")
        try:
            mt5.shutdown()
        except Exception as e:
            print(f"{datetime.now()} Erreur lors de l'arrêt de MT5 : {type(e).__name__}: {e}")
    print(f"{datetime.now()} Fin du programme")
    sys.exit(0)

# Fonction pour parser les plages
def parse_range(argList, as_int=False):
    try:
        start, stop, step = argList
        if as_int:
            range_values = range(int(start), int(stop + step), int(step))
            return [int(x) for x in range_values]
        if start + step >= stop:
            return [start]
        range_values = np.arange(start, stop, step)
        return [float(x) for x in range_values]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Format de plage invalide: {argList}. Utilisez start,stop,step (ex. 32.5,39.5,0.1). Erreur: {e}")

def load_ticks(symbol, cl, start_date, end_date):
    ticks = pd.DataFrame()
    if cl is None:
        mt50 = MetaTrader5()
        if not mt50.initialize():
            print(f"{datetime.now()} Avertissement : Initialisation MT5 échouée.")
            return ticks
        try:
            data = mt50.copy_ticks_range(symbol, start_date, end_date, MetaTrader5.COPY_TICKS_ALL)
            if data is None or len(data) == 0:
                print(f"{datetime.now()} Avertissement : No ticks.")
                return ticks
            ticks = pd.DataFrame(data)
            if 'volume' not in ticks.columns or ticks['volume'].sum() == 0:
                ticks = ticks.copy()
                ticks['volume'] = 1
            ticks.set_index('time_msc', drop=True, inplace=True)
            ticks.index = pd.to_datetime(ticks.index, unit='ms')
        except ValueError as e:
            print(f"{datetime.now()} load_data err : {type(e).__name__}: {e}")
            data = None
        mt50.shutdown()
    else:
        ticks = cl.get_ticks_from(symbol, start_date, end_date)
    if ticks is None or len(ticks) == 0:
        raise Exception('None ticks or empty')
    """
    print(ticks.tail(20))
    dfin = ticks.index[-1]
    print(type(dfin), dfin)
    ddeb = dfin - timedelta(seconds=60)
    print(type(ddeb), ddeb)
    dt = ticks.loc[ddeb:dfin]
    print(dt)
    dt = ticks[(ticks.index >= ddeb) & (ticks.index <= dfin)]
    print(dt)
    raise 'fin de test'
    """
    return ticks

import os
import pickle

def load_ticks_from_pickle(filename, symbol, cl, decalage=280):
    #print(filename)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            tick_data = pickle.load(f)
    else:
        ddeb = datetime.now() - timedelta(hours=decalage)
        dfin = datetime.now() + timedelta(hours=3)
        print(ddeb, dfin)
        tick_data = load_ticks(symbol, cl, ddeb, dfin)
        with (open(symbol+'.pkl', 'wb')) as f:
            pickle.dump(tick_data, f)
    #print(len(tick_data), '\n', tick_data.tail(2))
    return tick_data

# Fonction pour charger les données historiques
def load_data(symbol, timeframe, start_date, end_date):
    mt50 = MetaTrader5()
    if not mt50.initialize():
        print(f"0{datetime.now()} Avertissement : Initialisation MT5 échouée.")
        return pd.DataFrame()
    try:
        data = mt50.copy_rates_range(symbol, TF2MT[timeframe], start_date, end_date)
    except ValueError as e:
        print(f"{datetime.now()} load_data err : {type(e).__name__}: {e}")
        data = None
    mt50.shutdown()
    if data is None or len(data) == 0:
        print(f"{datetime.now()} Avertissement : Aucune donnée historique.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.set_index('time', drop=True)
    df.index = pd.to_datetime(df.index, unit='s')
    df = df[['open', 'high', 'low', 'close', 'tick_volume', 'real_volume']].rename(
        columns={'tick_volume': 'volume', 'real_volume': 'openinterest'})
    print(f"{datetime.now()} load_data: tail=\n{df.tail(2)}")

    min_bars = 20
    if len(df) < min_bars:
        print(f"{datetime.now()} Avertissement : Pas assez de barres ({len(df)} < {min_bars}).")
        return pd.DataFrame()

    # Conversion des types de données
    for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Vérification des volumes et openinterest
    if df['volume'].sum() == 0 and df['openinterest'].sum() == 0:
        print(
            f"{datetime.now()} Avertissement : Volumes ou openinterest nuls (volume_sum={df['volume'].sum()}, openinterest_sum={df['openinterest'].sum()}).")
        return pd.DataFrame()
    if df['volume'].isna().all() and df['openinterest'].isna().all():
        print(f"{datetime.now()} Avertissement : Volumes ou openinterest tous NaN.")
        return pd.DataFrame()

    print(
        f"{datetime.now()} Nombre de barres générées : {len(df)}, volume_sum={df['volume'].sum()}, openinterest_sum={df['openinterest'].sum()}")
    return df


def load_config(qui, filename=None, required=False):
    if filename is None:
        path, _ = QFileDialog.getOpenFileName(qui, "Charger config", "", "JSON (*.json)")
    else:
        path = filename
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    elif required:
        raise FileNotFoundError(f"CONFIG OBLIGATOIRE MANQUANTE : {filename}")
    else:
        return {}

def save_config(qui: QWidget, filename, cfg):
    path, _ = QFileDialog.getSaveFileName(parent=None, caption="Sauver config", directory=filename, filter="JSON (*.json)")
    if path:
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        qui.parent.statusBar().showMessage(f"Config sauvegardée : {path}")

def safe_float(value, default='N/A', fmt='.2f'):
    """
    Convertit en float si possible, sinon retourne default.
    """
    if value is None or value == '':
        return default
    try:
        f = float(value)
        return f"{f:{fmt}}"
    except (ValueError, TypeError):
        return default
