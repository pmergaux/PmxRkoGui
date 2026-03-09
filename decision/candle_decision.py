import pandas as pd
import numpy as np
from ta import trend, volatility, momentum

# ------------------------------------------------------------ calcul compilé
from numba import njit   # prange n'est plus utilisé ici

diff_col = ['diff_close','diff_ema', 'diff_rsi', 'diff_cci', 'diff_macd', 'diff_atr']
signal_col = ['signal_ema', 'signal_rsi', 'signal_macd', 'signal_cci', 'signal_atr']

@njit(cache=True, fastmath=True)  # <-- Retrait de parallel=True
def compute_indicators_pure_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        timestamp_seconds: np.ndarray,
        ema_period: int = 9,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_: int = 9,  # Renommage pour éviter conflit avec le buffer macd_signal
        cci_period: int = 20,
        willr_period: int = 14
) -> tuple:
    n = len(close)

    # === Buffers ===
    rsi = np.full(n, np.nan, dtype=np.float64)
    ema = np.full(n, np.nan, dtype=np.float64)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    macd_signal = np.full(n, np.nan, dtype=np.float64)
    macd_hist = np.full(n, np.nan, dtype=np.float64)
    cci = np.full(n, np.nan, dtype=np.float64)
    willr = np.full(n, np.nan, dtype=np.float64)
    time_vol = np.zeros(n, dtype=np.int32)
    volatility = np.zeros(n, dtype=np.float64)  # Ajout du buffer de volatilité

    # === EMA 14 (Séquentiel) ===
    alpha = 2.0 / (ema_period + 1.0)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]

    # === MACD LINE (Séquentiel) ===
    alpha_f = 2.0 / (macd_fast + 1.0)
    alpha_s = 2.0 / (macd_slow + 1.0)
    ema_fast = close.copy()
    ema_slow = close.copy()
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    for i in range(1, n):
        ema_fast[i] = alpha_f * close[i] + (1 - alpha_f) * ema_fast[i - 1]
        ema_slow[i] = alpha_s * close[i] + (1 - alpha_s) * ema_slow[i - 1]

    for i in range(n):
        macd_line[i] = ema_fast[i] - ema_slow[i]

    # === SIGNAL LINE (Séquentiel) ===
    signal_period = macd_signal_  # Utilisation du paramètre renommé
    alpha_sig = 2.0 / (signal_period + 1.0)

    if n >= signal_period:
        # Initialisation correcte: moyenne des premières valeurs non-NaN (ici, toutes non-NaN)
        first_signal_value = np.mean(macd_line[:signal_period])
        macd_signal[signal_period - 1] = first_signal_value

        # EMA classique à partir du point d'initialisation
        for i in range(signal_period, n):
            macd_signal[i] = alpha_sig * macd_line[i] + (1.0 - alpha_sig) * macd_signal[i - 1]

    # === HISTOGRAMME (Séquentiel) ===
    for i in range(n):
        if not np.isnan(macd_signal[i]):
            macd_hist[i] = macd_line[i] - macd_signal[i]

    # === RSI, CCI, Williams %R, Volatility (Calculs sur fenêtres) ===
    # Ces boucles sont correctes en mode njit séquentiel
    for i in range(n):  # Boucle unique pour les indicateurs de fenêtre et time_vol

        # Volatility (ex: Range) - Mis ici par souci de simplicité
        volatility[i] = (high[i] - low[i]) / close[i]

        # RSI
        if i >= rsi_period:
            gains = np.maximum(np.diff(close[i - rsi_period + 1:i + 1]), 0.0)
            losses = np.maximum(-np.diff(close[i - rsi_period + 1:i + 1]), 0.0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses) + 1e-12
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        # CCI
        if i >= cci_period:
            tp_arr = (high[i - cci_period + 1:i + 1] + low[i - cci_period + 1:i + 1] + close[
                i - cci_period + 1:i + 1]) / 3.0
            ma = np.mean(tp_arr)
            md = np.mean(np.abs(tp_arr - ma))
            tp = (high[i] + low[i] + close[i]) / 3.0
            cci[i] = (tp - ma) / (0.015 * (md + 1e-12))

        # Williams %R
        if i >= willr_period:
            h = np.max(high[i - willr_period + 1:i + 1])
            l = np.min(low[i - willr_period + 1:i + 1])
            willr[i] = -100.0 * (h - close[i]) / (h - l + 1e-12)

        # time_vol
        for i in range(n):
            ts = int(timestamp_seconds[i])
            hour = (ts // 3600) % 24
            minute = (ts // 60) % 60
            time_vol[i] = hour * 100 + minute

    return (
        rsi, ema, macd_line,
        macd_signal, macd_hist,
        cci, willr, time_vol, volatility  # <-- Retourne 9 éléments
    )


def add_indicators(df, param):
    try:
        df = df.copy()
        df = df.reset_index(drop=True)
        # Conversion une fois pour toutes
        close_np = df['close'].to_numpy(dtype=np.float64)
        high_np = df['high'].to_numpy(dtype=np.float64)
        low_np = df['low'].to_numpy(dtype=np.float64)
        df['time'] = pd.to_datetime(df['time'])
        ts_np = df['time'].astype('int64').values // 1_000_000_000
        ema_period = param.get("ema_period", 9)
        rsi_period = param.get('rsi_period', 14)
        macd = param.get('macd', {'macd_fast':12, 'macd_slow':26, "macd_signal":9})
        macd_fast = macd['macd_fast']
        macd_slow = macd['macd_slow']
        macd_signal = macd['macd_signal']
        cci_period = param.get('cci_period', 20)
    except BaseException as e:
        print("err add indic 1", e)
        return None
    # Appel magique (9 variables dépaquetées pour correspondre à la sortie Numba)
    rsi, ema, macd_line, macd_signal, macd_hist, cci, willr, time_vol, vol = compute_indicators_pure_numba(
        close_np, high_np, low_np, ts_np, ema_period, rsi_period, macd_fast, macd_slow, macd_signal, cci_period
    )
    # Remise dans le DataFrame
    df['RSI'] = rsi
    df['EMA'] = ema
    df['MACD_line'] = macd_line
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    df['CCI'] = cci
    df['Williams_R'] = willr
    df['time_vol'] = time_vol
    df['volatility'] = vol  # vol vient maintenant de Numba

    # Les calculs non-Numba restent ici
    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)
    df['time_live'] = df['volatility'] / (df['time_diff'] + 1e-6)
    df['time_live'] = df['time_live'].replace([np.inf, -np.inf], 0).fillna(0)

    return df.dropna().reset_index(drop=True)
# ------------------------------------------------------------------------------
# Fonctions utilitaires
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return [macd, signal_line, macd-signal_line]

#  une moyenne exponentielle d'une colonne df['value'].ewm(span=14).mean()

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md)

def calculate_williams_r(high, low, close, period=14):
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest)

def calculate_stoch_rsi(rsi, period=14):
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min)


def calculate_time_live(df, cfg):
    # 1. Calcul de base
    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)
    # 2. Volatilité "Augmentée" (incluant le slippage réel)
    # abs(close - closer) capture l'impulsion finale au-delà de la brique
    volatilite_reelle = (abs(df['close'] - df['close_renko']) + cfg['parameters']['renko_size']) / df['close']
    # 3. Calcul de la vélocité
    df['time_live'] = volatilite_reelle / (df['time_diff'] + 1e-6)
    # 4. Compression logarithmique (L'astuce du curieux)
    df['time_live'] = np.log1p(df['time_live'])
    # Nettoyage final
    df['time_live'] = df['time_live'].replace([np.inf, -np.inf], 0).fillna(0)
    return df

def calculate_japonais(df: pd.DataFrame):
    df = df.copy()
    df['bb_mavg'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
    df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
    #df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
    #df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']

    df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    open_buy = ((df['direction'] == 1) & (df['close'] < df['bb_mavg']))
    open_sell = ((df['direction'] == -1) & (df['close'] > df['bb_mavg']))
    df['sigo'] = np.where(open_buy, 1, np.where(open_sell, -1, 0))
    close_sell = (df['close']*1.05 < df['bb_lband'])
    close_buy = (df['close']*1.05 > df['bb_hband'])
    df['sigc'] = np.where(close_sell, 1, np.where(close_buy, -1, 0))
    return df

def calculate_indicators(df : pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    #df.reset_index(drop=True)
    features = config['features']
    param = config["parameters"]
    try:
        df['bb_mavg'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
        df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
    #    df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
    #    df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']
        target_col = config["target"]["target_col"]
        if not isinstance(target_col, list):
            target_col = [target_col]
        if 'EMA' in features or "EMA" in target_col or 'diff_ema' in features or 'diff_ema' in target_col:
            df['EMA'] = trend.EMAIndicator(df['close'], window=param.get('ema_period', 9)).ema_indicator()
        if 'RSI' in features or 'RSI' in target_col or 'diff_rsi' in features or 'diff_rsi' in target_col:
            df['RSI'] = momentum.RSIIndicator(df['close'], window=param.get('rsi_period', 14)).rsi()
        if 'MACD_hist' in features or "MACD_hist" in target_col or 'diff_macd' in features or 'diff_macd' in target_col:
            pmacd = param.get('macd', {"macd_fast":12, "macd_slow": 26, "macd_signal": 9})
            macd = trend.MACD(df['close'], window_fast=pmacd["macd_fast"], window_slow=pmacd["macd_slow"], window_sign=pmacd["macd_signal"])
            df['MACD_line'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
        if 'ATR' in features or 'ATR' in target_col or 'diff_atr' in features or 'diff_atr' in target_col:
            df['ATR'] = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=param.get('atr_period', 14)).average_true_range()
        if 'Stoch_RSI' in features:
            df['Stoch_RSI'] = momentum.StochRSIIndicator(df['close'], window=param.get('stochRsi_period', 14)).stochrsi()
        if 'Williams_R' in features:
            df['Williams_R'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=param.get('williamsR_period',14)).williams_r()
        if 'CCI' in features or 'CCI' in target_col or 'diff_cci' in features or 'diff_cci' in target_col:
            df['CCI'] = trend.CCIIndicator(df['high'], df['low'], df['close'], window=param.get('cci_period', 14)).cci()
        if 'time_live' in features:
            df = calculate_time_live(df, config)
        #print(f"Colonnes après direction_openr_closer: {df.columns.tolist()}")
        return df.dropna()   #.reset_index(drop=True)
    except BaseException as e:
        print(f"Erreur dans calculate_indicators: {e}")
        raise "Erreur dans calculate_indicators"

def choix_features(df: pd.DataFrame, cfg: dict):
    features = cfg['features']
    open_rules = cfg['open_rules']
    close_rules = cfg["close_rules"]
    param = cfg["parameters"]
    target = cfg["target"]
    rsi_high = param.get('rsi_high', 70)
    rsi_low = param.get('rsi_low', 30)
    s_rsi_high = param.get('s_rsi_high', 0.8)
    s_rsi_low = param.get('s_rsi_low', 0.2)
    williams_high = param.get('williams_low', -80)
    williams_low = param.get('williams_high', -20)
    cci_high = param.get('cci_high', 100)
    cci_low = param.get('cci_low', -100)
    """
    if 'closer' in df.columns:
        df['direction'] = np.where(df['close_renko'] > df['open_renko'], 1, np.where(df['open_renko'] > df['close_renko'], -1, 0))
    else:
    # distinguer close et close_renko ne sert à rien puisque pour les renko close est défini avec open et close
    """
    df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    df['diff_close'] = (df['close'] - df['close'].shift(1))/df['close'].shift(1)
    buy_cond = (df['direction'] == 1)
    sell_cond = (df['direction'] == -1)
    df['sigc'] = 0
    if "EMA" in df.columns:
        buy_local  = ((df['close'] > df['EMA']) & (df['EMA'] >= df['EMA'].shift(1)))
        sell_local = ((df['close'] < df['EMA']) & (df['EMA'] <= df['EMA'].shift(1)))
        df['signal_ema'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_ema'] = (df['EMA'] - df['close'])/df['close']
    if "RSI" in df.columns:
        buy_local = (df['RSI'] < rsi_low)
        sell_local = (df['RSI'] > rsi_high)
        df['signal_rsi'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_rsi'] = (df['RSI'] - 50)/50
    if "MACD_hist" in df.columns:
        df['signal_macd'] = np.where(df['MACD_hist'] > 0, 1, np.where(df['MACD_hist'] < 0 , -1, 0))
        df['diff_macd'] = df['MACD_hist']
    if 'Stoch_RSI' in df.columns:
        buy_cond &= (df['Stoch_RSI'] < s_rsi_low)
        sell_cond &= (df['Stoch_RSI'] > s_rsi_high)
    if 'ATR' in df.columns:
        buy_local = (df['ATR'] > df['ATR'].mean())
        sell_local = (df['ATR'] < df['ATR'].mean())
        df['signal_atr'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_atr'] = (df['ATR'] - df['ATR'].mean())/df['ATR'].mean()
    if 'Williams_R' in df.columns:
        buy_cond &= (df['Williams_R'] < williams_low)
        sell_cond &= (df['Williams_R'] > williams_high)
    if 'CCI' in df.columns:
        buy_local = (df['CCI'] < cci_low)
        sell_local = (df['CCI'] > cci_high)
        df['signal_cci'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_cci'] = df['CCI'] / 200
    if close_rules.get("close_sens", False):
        #clos_cond = ((df['direction'] != df['direction'].shift(1)))   #la direction actuelle avec la précédente
        #df['sigc'] = np.select([clos_cond & df['sigc']==0], [4*df['direction']], default=df['sigc'])
        clos_cond = (df['sigc'] == 0)  # toujours vraie donc sigc remplace direction
        df['sigc'] = np.select([clos_cond], [df['direction']], default=df['sigc'])
    first = True
    # une valeur par défaut
    opening_cond = (df['direction'] != 0)   # donc toujours vraie
    df['sigo'] = 0
    for col in signal_col:
        if first:
            if not col in df.columns:
                continue
            df['sigo'] = df[col]
            opening_cond = (df[col] != 0)
            first = False
            continue
        if col in df.columns:
            opening_cond &= ((df[col] == 0) | (df[col] == df['sigo']))
    df['sigo'] = np.where(opening_cond, 1 * df['sigo'], 0)
    return df
