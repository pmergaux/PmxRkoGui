import pandas as pd
import numpy as np
from ta import trend, volatility, momentum

# ------------------------------------------------------------ calcul compilé
from numba import njit   # prange n'est plus utilisé ici
@njit(cache=True, fastmath=True)  # <-- Retrait de parallel=True
def compute_indicators_pure_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        timestamp_seconds: np.ndarray,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,  # Renommage pour éviter conflit avec le buffer macd_signal
        cci_period: int = 20,
        willr_period: int = 14
) -> tuple:
    n = len(close)

    # === Buffers ===
    rsi = np.full(n, np.nan, dtype=np.float64)
    ema14 = np.full(n, np.nan, dtype=np.float64)
    macd_line = np.full(n, np.nan, dtype=np.float64)
    macd_signal = np.full(n, np.nan, dtype=np.float64)
    macd_hist = np.full(n, np.nan, dtype=np.float64)
    cci = np.full(n, np.nan, dtype=np.float64)
    willr = np.full(n, np.nan, dtype=np.float64)
    time_vol = np.zeros(n, dtype=np.int32)
    volatility = np.zeros(n, dtype=np.float64)  # Ajout du buffer de volatilité

    # === EMA 14 (Séquentiel) ===
    alpha = 2.0 / (14 + 1.0)
    ema14[0] = close[0]
    for i in range(1, n):
        ema14[i] = alpha * close[i] + (1.0 - alpha) * ema14[i - 1]

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
    signal_period = macd_signal_period  # Utilisation du paramètre renommé
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
        volatility[i] = high[i] - low[i]

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
        rsi, ema14, macd_line,
        macd_signal, macd_hist,
        cci, willr, time_vol, volatility  # <-- Retourne 9 éléments
    )


def add_indicators(df):
    df = df.copy()
    df = df.reset_index(drop=True)
    # Conversion une fois pour toutes
    close_np = df['close'].to_numpy(dtype=np.float64)
    high_np = df['high'].to_numpy(dtype=np.float64)
    low_np = df['low'].to_numpy(dtype=np.float64)
    df['time'] = pd.to_datetime(df['time'])
    ts_np = df['time'].astype('int64').values // 1_000_000_000

    # Appel magique (9 variables dépaquetées pour correspondre à la sortie Numba)
    rsi, ema, macd_line, macd_signal, macd_hist, cci, willr, time_vol, vol = compute_indicators_pure_numba(
        close_np, high_np, low_np, ts_np
    )
    # Remise dans le DataFrame
    df['RSI'] = rsi
    df['EMA'] = ema
    df['MACD'] = macd_line
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

def calculate_time_live(df):
    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['time_live'] = df['volatility'] / (df['time_diff'] + 1e-6)  # évite / par zéro
    # effectuer dans clean_features df['time_live'] = df['time_live'].replace([np.inf, -np.inf], 0).fillna(0)
    return df

def calculate_japonais(df: pd.DataFrame, config:dict):
    df = df.copy()
    param = config["parameters"]
    df['bb_mavg'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    #df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
    #df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
    #df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
    #df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']

    df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    open_buy = ((df['direction'] == 1) & (df['close'] < df['bb_mavg']))
    open_sell = ((df['direction'] == -1) & (df['close'] > df['bb_mavg']))
    df['sigc'] = np.where(open_buy, 1, np.where(open_sell, -1, 0))
    return df

def calculate_indicators(df : pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(drop=True)
    features = config['features']
    param = config["parameters"]

    df['bb_mavg'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
    df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
#    df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
#    df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']

    if 'EMA' in features:
        df['EMA'] = trend.EMAIndicator(df['close'], window=param.get('ema_period', 9)).ema_indicator()
    if 'RSI' in features:
        df['RSI'] = momentum.RSIIndicator(df['close'], window=param.get('rsi_period', 14)).rsi()
    if 'MACD_line' in features or 'MACD_hist' in features:
        pmacd = param.get('macd', {"macd_fast":12, "macd_slow": 26, "macd_signal": 9})
        macd = trend.MACD(df['close'], window_fast=pmacd["macd_fast"], window_slow=pmacd["macd_slow"], window_sign=pmacd["macd_signal"])
        df['MACD_line'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
    if 'ATR' in features:
        df['ATR'] = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=param.get('atr_period', 14)).average_true_range()
    if 'Stoch_RSI' in features:
        df['Stoch_RSI'] = momentum.StochRSIIndicator(df['close'], window=param.get('stochRsi_period', 14)).stochrsi()
    if 'Williams_R' in features:
        df['Williams_R'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=param.get('williamsR_period',14)).williams_r()
    if 'CCI' in features:
        df['CCI'] = trend.CCIIndicator(df['high'], df['low'], df['close'], window=param.get('cci_period', 14)).cci()
    if 'time_live' in features:
        df = calculate_time_live(df)
    #print(f"Colonnes après direction_openr_closer: {df.columns.tolist()}")
    return df.dropna().reset_index(drop=True)

def choix_features(df: pd.DataFrame, cfg: dict):
    features = cfg['features']
    open_rules = cfg['open_rules']
    close_rules = cfg["close_rules"]
    param = cfg["parameters"]
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
    buy_cond = (df['direction'] == 1)
    sell_cond = (df['direction'] == -1)
    if 'EMA' in features and open_rules['rule_ema']:
        buy_cond  &= (df['close'] > df['EMA'].shift(1))
        sell_cond &= (df['close'] < df['EMA'].shift(1))
    if 'RSI' in features and open_rules['rule_rsi']:
        buy_cond &= (df['RSI'] < rsi_high)
        sell_cond &= (df['RSI'] > rsi_low)
    if 'MACD_hist' in features and open_rules['rule_macd']:
        buy_cond &= (df['MACD_line'] > df['MACD_signal'])
        sell_cond &= (df['MACD_line'] < df['MACD_signal'])
    if 'Stoch_RSI' in features and open_rules.get('stk_rsi',False):
        buy_cond &= (df['Stoch_RSI'] < s_rsi_low)
        sell_cond &= (df['Stoch_RSI'] > s_rsi_high)
    if 'ATR' in features and open_rules.get('atr', False):
        buy_cond &= (df['ATR'] > df['ATR'].mean())
        sell_cond &= (df['ATR'] > df['ATR'].mean())
    if 'Williams_R' in features and open_rules.get('wr', False):
        buy_cond &= (df['Williams_R'] < williams_low)
        sell_cond &= (df['Williams_R'] > williams_high)
    if 'CCI' in features and open_rules.get('cci', False):
        buy_cond &= (df['CCI'] < cci_low)
        sell_cond &= (df['CCI'] > cci_high)
    df['sigc'] = 0
    """
    clos_cond = ((df['direction'] == 1) & (df['close'] < df['EMA'])) | \
                ((df['direction'] == -1) & (df['close'] > df['EMA']))
    df['sigc'] = np.where(clos_cond, 5, 0) if close_rules.get("close_ema", False) else 0
    """
    clos_cond = ((df['direction'] != df['direction'].shift(1)) & (df['sigc'] == 0))
    df['sigc'] = np.where(clos_cond, 4, df['sigc']) if close_rules.get("close_sens", False) else 0
    df['sigo'] = np.where(buy_cond, 1, np.where(sell_cond, -1, 0))
    df.loc[df.index[-1], 'sigc'] = 0
    df.loc[df.index[-1], 'sigo'] = 0
    # NOTA le dernière ligne contient des valeurs temporaires
    return df

