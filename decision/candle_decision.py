import pandas as pd
import numpy as np
from ta import trend, volatility, momentum

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

def calculate_japonais(df: pd.DataFrame, config:dict):
    df = df.copy()
    param = config["parameters"]
    df['bb_mavg'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    #df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
    #df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
    df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
    df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']

    df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    clos_buy = ((df['direction'] == 1) & (df['close'] > df['bb_max']))
    clos_sell = ((df['direction'] == -1) & (df['close'] < df['bb_min']))
    df['sigc'] = np.where(clos_buy, 6, np.where(clos_sell, 7, 0))
    return df

def calculate_indicators(df : pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
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
    if 'MACD_line' in features:
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
    if 'time_vol' in features:
        df['time_diff'] = df.index.to_series().diff().dt.total_seconds().fillna(0)
        df['volatility'] = (df['high'] - df['low']) / df['close']
        df['time_vol'] = df['volatility'] / (df['time_diff'] + 1e-6)
        df['time_vol'] = df['time_vol'].replace([np.inf, -np.inf], 0).fillna(0)
    #print(f"Colonnes après direction_openr_closer: {df.columns.tolist()}")
    """
    incohérences = ((df['open'] <= df['close']) & (df['direction'] == -1) |
                    (df['open'] > df['close']) & (df['direction'] == 1)).sum()
    if incohérences > 0:
        print(f"{datetime.now()} {incohérences} briques incohérentes détectées. Signaux potentiellement erronés.")
    """
    return df.fillna(0)

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

