import argparse
import platform

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# from PyQt6.QtSql import password
from sklearn.linear_model import LinearRegression
import sys
from mt5linux import MetaTrader5

import os
import pickle

# Codes de couleur de base
ROUGE = '\033[91m'
VERT = '\033[92m'
JAUNE = '\033[93m'
BLEU = '\033[94m'
BLEU_CIEL = '\033[96m'  # Cyan clair
ROSE = '\033[95m'  # Magenta clair
VIOLET = '\033[35m'
ROSE_VIF = '\033[38;5;201m'
VIOLET_CLAIR = '\033[38;5;177m'
BLANC = '\033[97m'  # Blanc brillant
GRAS = '\033[1m'  # Rend n'importe quelle couleur plus intense
RESET = '\033[0m'  # Important pour arrêter la couleur
C = {
    "info": '\033[94m',         # Bleu
    "gain": '\033[92m',         # Vert clair
    "perte": '\033[91m',        # Rouge clair
    "debug": '\033[38;5;201m',  # Rose vif
    "violet": '\033[35m',       # Violet standard
    "violet_clair": '\033[38;5;177m', # Violet clair / Lavande
    "reset": '\033[0m'
}

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

DIRBUYCLOSE = 5
DIRSELLCLOSE = -5
EMABUYCLOSE = 4
EMASELLCLOSE = -4
PROBANEUTRE = 5

BUY_STOP = 3
BUY_CONT = 2
BUY = 1
SELL = -1
SELL_CONT = -2
SELL_STOP = -3
CLOSE = 5
FCLOSE = 6
NCLOSE = 7
NONE = 0
sens_lib = {NCLOSE: 'nclose', FCLOSE: 'fclose', CLOSE: 'close',4: 'extra+',
            BUY_STOP:'buy stop', BUY_CONT:'buy cont', BUY: 'buy', NONE: 'none',
            SELL: 'sell', SELL_CONT: 'sell cont', SELL_STOP: 'sell stop', -4: 'extra-'}
UP = 1
DOWN = -1

Tf2Rs = {
    '1m': '1Min',
    '2m': '2Min',
    '3m': '3Min',
    '4m': '4Min',
    '5m': '5Min',
    '10m': '10Min',
    '15m': '15Min',
    '30m': '30Min',
    '1h': 'h',
    '2h': '2h',
    '3h': '3h',
    '4h': '4h',
    '1d': 'D',
}

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

path_fus="/home/pierre/.wine/drive_c/Program Files/Fusion Markets MetaTrader 5/terminal64.exe"
path_adm='/home/pierre/.wine/drive_c/Program Files/Admiral Markets MT5/terminal64.exe'
path_meta='/home/pierre/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"

# connect to the server
# Détecter le système d'exploitation
os_name = platform.system()
"""
if os_name == 'Windows':
    import MetaTrader5 as mt5
else:  # Linux
"""
def connectMt5(live, host='localhost'):
    # Initialisation MT5
    mt = None
    if os_name == 'Windows':
        pass
        # if not mt5.initialize():
        #    raise Exception("Erreur d'initialisation MT5 (Windows) : " + str(mt5.last_error()))
    else:
        try:
            print(f"initialize Mt5 with {host} port {live.get('port', 18812) if live is not None else 18812}")
            mt = MetaTrader5(host=host, port=live.get("port", 18812) if live is not None else 18812)  # Valeurs par défaut
            login=live.get('mt5_login', None) if live is not None else None
            if login is not None:
                password=live.get('mt5_password', None) if live is not None else None
            else:
                password = None
            print(f"connection login {login} password {password} path {live.get('mt5_path', path_meta) if live is not None else path_meta}")
            if not mt.initialize(path=live.get("mt5_path", path_meta) if live is not None else path_meta, portable=True):  #login=login, password=password,
                raise Exception("Erreur d'initialisation MT5 (Linux) : " + str(mt.last_error()))
            mt.execute("import numpy as np")  # Importe NumPy dans le namespace distant
            if login is None:
                print(f"{datetime.now()} login without account, but terminal account")
            account_info = mt.account_info()
            print("name   = {}".format(account_info.name))
            print("login  = {}".format(account_info.login))
            print("server = {}".format(account_info.server))
        except Exception as e:
            raise Exception(f"Erreur lors de l'initialisation mt5linux : {e}")
    return mt


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

def calculer_stats(df_segment, zone='bid'):
    """
    Calcule 3 indicateurs :
    1. Pente (en $ par tick) ou 'close' pour candles
    2. Volatilité Standard (en $ - racine carrée de la variance)
    3. Volatilité Logarithmique (en % - nervosité relative)
    """
    #if len(df_segment) < 100:        return 0, 0, 0

    # --- 1. PENTE SUR PRIX BRUT ($) ---
    dfn = df_segment.dropna()
    X = np.arange(len(dfn)).reshape(-1, 1)
    y = dfn[zone].values
    model = LinearRegression()
    model.fit(X, y)
    pente = model.coef_[0]
    # --- 2. ÉCART-TYPE STANDARD ($) ---
    # C'est la racine carrée de la variance du prix Bid
    std = df_segment[zone].std()
    # --- 3. VOLATILITÉ RELATIVE (%) ---
    # Écart-type des log-rendements * 100
    log_returns = np.log(df_segment[zone] / df_segment[zone].shift(1)).dropna()
    vol_log_pct = log_returns.std() * 100
    # --- CALCUL DU R2 ---
    # Le R2 est le coefficient de détermination (entre 0 et 1)
    r2 = model.score(X, y)
    # --- CALCUL DE L'ER (Efficiency Ratio) ---
    change = np.abs(y[-1] - y[0])
    vol_abs = np.sum(np.abs(np.diff(y)))
    er = change / vol_abs if vol_abs != 0 else 0
    return pente, std, vol_log_pct, r2, er

def to_number(x, strict=False):
    try:
        return int(x) if x.isdigit() else float(x)
    except:
        return None if strict else x

def safe_format(value, fmt=".2f"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            value = value[0]  # extrait le scalaire si liste singleton
        else:
            value = np.mean(value)  # ou np.median, ou value[0], au choix
    if isinstance(value, (int, float, np.number)):
        return f"{{:{fmt}}}".format(value)
    else:
        return str(value)  # fallback

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

from datetime import timezone # À ajouter en haut du fichier
def load_ticks(live, cl, start_date, end_date):
    ticks = pd.DataFrame()
    if cl is None:
        mt50 = connectMt5(live)
        #mt50 = MetaTrader5()
        #if not mt50.initialize():
        # print(f"0{datetime.now()} Avertissement : Initialisation MT5 échouée.")
        # return pd.DataFrame()
        try:
            #print(start_date, end_date)
            # 1. On s'assure que les dates sont bien en UTC pour mt5linux
            # Cela évite que le wrapper utilise l'horloge système décalée
            #s_date = start_date.replace(tzinfo=timezone.utc)
            #e_date = end_date.replace(tzinfo=timezone.utc)
            #print(f"DEBUG MT5Linux - Appel de {s_date} à {e_date}")
            # 2. On passe les objets datetime avec timezone
            #data = mt50.copy_ticks_range(symbol, s_date, e_date, MetaTrader5.COPY_TICKS_ALL)
            data = mt50.copy_ticks_range(live['symbol'], start_date, end_date, MetaTrader5.COPY_TICKS_ALL)
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
        ticks = cl.get_ticks_from(live["symbol"], start_date, end_date)
    if ticks is None or len(ticks) == 0:
        raise Exception('None ticks or empty')
    return ticks

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

def reload_ticks_from_pickle(filename, live, cl, date_start, date_end):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            tick_data = pickle.load(f)
    else:
        tick_data = load_ticks(live, cl, date_start, date_end)
        #print(f"reload {date_start} {type(date_start)} to {date_end} {type(date_end)}")
        #print(tick_data.head())
        with (open(filename, 'wb')) as f:
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

def clean_numpy_types(data):
    """
    Parcourt récursivement un dictionnaire ou une liste pour convertir
    les types NumPy en types Python natifs.
    """
    if isinstance(data, dict):
        return {key: clean_numpy_types(value) for key, value in data.items()}
    if isinstance(data, list):
        return [clean_numpy_types(item) for item in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def get_extension(path: str) -> str:
    """
    Retourne l'extension d'un fichier (en minuscules, avec le point).
    Exemples :
        "../xxx/fich.txt"      → ".txt"
        "./data/model.keras"   → ".keras"
        "fichier"              → ""
        "archive.tar.gz"       → ".gz"   (prend la dernière extension)
        "/home/user/file."     → "."

    Parameters
    ----------
    path : str
        Chemin complet ou relatif du fichier

    Returns
    -------
    str
        L'extension (avec le point) ou chaîne vide si aucune
    """
    # os.path.splitext sépare juste avant le dernier point
    _, ext = os.path.splitext(path)
    return ext.lower()  # on retourne en minuscules pour comparaison facile

def get_clean_timestamp():
    """Retourne l'heure actuelle sans les nanosecondes/microsecondes."""
    return datetime.now().replace(microsecond=0)

def get_dynamic_sensitivity(base_sensitivity, interval_seconds):
    # Logarithme naturel pour aplatir la courbe d'impact de l'intervalle de temps
    # alpha est un facteur d'échelle à ajuster (ex: 0.2 ou 0.3)
    alpha = 0.25
    multiplier = 1.0 + alpha * np.log(max(interval_seconds, 1))
    return base_sensitivity * multiplier

def get_linear_slope(values):
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope

def get_current_and_previous_candles_seconds(df_candles, current_time_ms):
    """
    Extrait la bougie OHLC (index en secondes) et les 63 précédentes
    à partir d'un tick en millisecondes.
    :param df_candles: DataFrame des bougies dont l'index est en secondes.
    :param current_time_ms: Le timestamp du tick en millisecondes.
    :return: DataFrame contenant la bougie courante + les 63 précédentes (max 64 lignes).
    """
    # 1. Conversion du temps du tick (ms -> secondes) pour aligner avec l'index des bougies
    current_time_sec = current_time_ms / 1000.0
    # 2. Filtrer les bougies antérieures ou égales au tick
    mask = df_candles.index <= current_time_sec
    # 3. Extraire la bougie courante et les 63 précédentes
    window = df_candles.loc[mask].tail(64)
    return window

def get_last_three_chars(texte: str):
    if len(texte) < 3:
        return None
    return [texte[-3], texte[-2], texte[-1]]

