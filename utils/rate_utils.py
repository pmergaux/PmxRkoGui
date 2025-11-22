# Fonctions ticks2rates et ticks22rates modifiées
from datetime import datetime
import pandas as pd
import numpy as np
from future.backports.datetime import timedelta

# Constantes de Timeframe (Identiques à votre définition)
Tf2Rs = {
    '1m': '1Min', '2m': '2Min', '3m': '3Min', '4m': '4Min',
    '5m': '5Min', '10m': '10Min', '15m': '15Min', '30m': '30Min',
    '1h': 'h', '2h': '2h', '3h': '3h', '4h': '4h', '1d': 'D',
}


def ticks2rates(data: pd.DataFrame, period: str, value: str) -> pd.DataFrame:
    """Génère les rates OHLC initiales à partir des ticks."""
    if period not in Tf2Rs:
        raise ValueError(f"Période '{period}' non supportée.")
    if value not in data.columns:
        raise ValueError(f"Colonne de valeur '{value}' manquante dans les données.")

    return data[value].resample(Tf2Rs[period]).ohlc()

def ticks22rates(rates: pd.DataFrame, ticks: pd.DataFrame, timeframe: int, value='bid'):
    per = timedelta(seconds=timeframe)
    rate = rates.iloc[-1]
    drate = rates.index[-1].to_pydatetime()
    lnt = len(ticks)
    rt = 0
    for i in range(lnt):
        dtick = ticks.index[i].to_pydatetime()
        bid = ticks.iloc[i][value]
        if (dtick - drate) > per:
            drate += per
            rate = pd.DataFrame.from_dict({drate: [bid, bid, bid, bid]},
                                          orient='index', columns=['open', 'high', 'low', 'close'])
            rates = pd.concat([rates, rate])
            rt += 1
        else:
            rates.loc[drate, 'close'] = bid
            if rates.iloc[-1]['high'] < bid:
                rates.loc[drate, 'high'] = bid
            elif rates.iloc[-1]['low'] > bid:
                rates.loc[drate, 'low'] = bid
    return rt, rates


def _ticks2rates(ticks: pd.DataFrame, period: str, value: str='bid'):
    #print(f"{datetime.now()} ticks2rates: Début, ticks.shape={ticks.shape}, period={period}, value={value}")
    if value not in ticks.columns:
        raise ValueError(f"Colonne {value} absente dans les ticks")
    if 'volume' not in ticks.columns or ticks['volume'].sum() == 0:
        ticks = ticks.copy()
        ticks['volume'] = 1
        #print(f"{datetime.now()} ticks2rates: Volume forcé à 1")
    #else:
        #print(f"{datetime.now()} ticks2rates: Volume présent, somme={ticks['volume'].sum()}")

    if not isinstance(ticks.index, pd.DatetimeIndex):
        if 'time' in ticks.columns:
            ticks = ticks.set_index('time')
            ticks.index = pd.to_datetime(ticks.index)
        else:
            raise ValueError("Colonne 'time' absente ou index non-datetime")

    df = ticks[[value, 'volume']].resample(Tf2Rs[period]).agg({
        value: ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })

    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    #print(f"{datetime.now()} ticks2rates: Fin, df.shape={df.shape}, volume_somme={df['volume'].sum()}")
    return df

def _ticks22rates(rates: pd.DataFrame, ticks: pd.DataFrame, timeframe: int):
    # print(f"{datetime.now()} ticks22rates: Début, ticks.shape={ticks.shape}, timeframe={timeframe}")
    valeur = 'bid'
    if valeur not in ticks.columns:
        raise ValueError(f"Colonne {valeur} absente dans les ticks")
    if 'volume' not in ticks.columns or ticks['volume'].sum() == 0:
        ticks = ticks.copy()
        ticks['volume'] = 1
        # print(f"{datetime.now()} ticks22rates: Volume forcé à 1")
    # else:
        # print(f"{datetime.now()} ticks22rates: Volume présent, somme={ticks['volume'].sum()}")

    if not isinstance(ticks.index, pd.DatetimeIndex):
        if 'time' in ticks.columns:
            ticks = ticks.set_index('time')
            ticks.index = pd.to_datetime(ticks.index)
        else:
            raise ValueError("Colonne 'time' absente ou index non-datetime")

    if rates is not None and len(rates) > 0:
        if 'volume' not in rates.columns:
            rates['volume'] = 1
            # print(f"{datetime.now()} ticks22rates: Volume forcé à 1 dans rates")
    else:
        rates = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    per = pd.Timedelta(seconds=timeframe)
    rt = 0
    drate = rates.index[-1].to_pydatetime() if not rates.empty else None

    current_bar = None
    current_volume = 0
    #current_spread = 0
    #current_num = 0
    for i in range(len(ticks)):
        dtick = ticks.index[i].to_pydatetime()
        bid = ticks.iloc[i][valeur]
        volume = ticks.iloc[i]['volume']
        #spread = ticks.iloc[i]['spread']

        if drate is None or dtick < drate:
            continue

        if (dtick - drate) > per:
            if current_bar is not None:
                new_row = pd.DataFrame({
                    'open': [current_bar['open']],
                    'high': [current_bar['high']],
                    'low': [current_bar['low']],
                    'close': [current_bar['close']],
                    'volume': [current_volume],
                    #'spread': [current_spread/current_num]
                }, index=[pd.Timestamp(drate)])
                rates = pd.concat([rates, new_row])
                rt += 1

            drate += per
            current_bar = {'open': bid, 'high': bid, 'low': bid, 'close': bid}
            current_volume = volume
            #current_spread = spread
            #current_num = 1
        else:
            if current_bar is not None:
                current_bar['close'] = bid
                current_bar['high'] = max(current_bar['high'], bid)
                current_bar['low'] = min(current_bar['low'], bid)
                current_volume += volume
                #current_spread += spread
                #current_num += 1

    if current_bar is not None:
        new_row = pd.DataFrame({
            'open': [current_bar['open']],
            'high': [current_bar['high']],
            'low': [current_bar['low']],
            'close': [current_bar['close']],
            'volume': [current_volume],
            #'spread': [current_spread/current_num]
        }, index=[pd.Timestamp(drate)])
        rates = pd.concat([rates, new_row])
        rt += 1

    rates.index = pd.to_datetime(rates.index)
    rates.sort_index(inplace=True)
    # print(f"{datetime.now()} ticks22rates: Fin, rates.shape={rates.shape}, volume_somme={rates['volume'].sum()}")
    return rt, rates

# ########## ancienne formule
def ticks222rates(rates: pd.DataFrame, ticks: pd.DataFrame, timeframe: int):
    per = datetime.fromtimestamp(timeframe) - datetime(1970, 1, 1, 1, 0, 0)
    rate = rates.iloc[-1]
    drate = rates.index[-1].to_pydatetime()
    lnt = len(ticks)
    rt = 0
    # spread = rates.iloc[-1]['spread']
    for i in range(lnt):
        dtick = ticks.index[i].to_pydatetime()
        if dtick < drate:
            continue
        bid = ticks.iloc[i]['bid']
        if (dtick - drate) > per:
            # rates.iloc[-1]['close'] = bid
            drate += per
            rate = pd.DataFrame.from_dict({drate: [bid, bid, bid, bid]},  #, 0.0, spread, 0.0]},
                                          orient='index', columns=['open', 'high', 'low', 'close']) #,'tick_volume', 'spread', 'real_volume'])
            rates = pd.concat([rates, rate])
            rt += 1
        else:
            rates.loc[drate, 'close'] = bid
            if rates.iloc[-1]['high'] < bid:
                rates.loc[drate, 'high'] = bid
            elif rates.iloc[-1]['low'] > bid:
                rates.loc[drate, 'low'] = bid
    return rt, rates

