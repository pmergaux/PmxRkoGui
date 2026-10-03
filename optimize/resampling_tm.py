import pandas as pd
import numpy as np
from decision.candle_decision import calculate_r2  # Votre fonction existante


def compare_timeframes(df_raw, tf_list=['15T', '30T', '1H', '2H']):
    results = {}

    for tf in tf_list:
        # 1. Resample les données (OHLC)
        df_tf = df_raw.resample(tf).last().dropna()

        # 2. Calculer le R2 sur ce nouveau timeframe
        # Supposons une fenêtre de 14 périodes pour le R2
        df_tf['r2'] = calculate_r2(df_tf['close'], window=14)

        # 3. Calculer la "Qualité" :
        # Combien de fois le R2 est > 0.85 pendant qu'un mouvement de X% se produit ?
        quality_score = df_tf[df_tf['r2'] > 0.85]['close'].diff().abs().sum()

        results[tf] = {
            'nb_points': len(df_tf),
            'r2_stable_pct': (df_tf['r2'] > 0.85).mean() * 100,
            'volatility_avg': df_tf['close'].pct_change().std()
        }
    return results