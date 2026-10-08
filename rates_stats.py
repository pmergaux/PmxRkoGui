from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

mt5 = MetaTrader5()
# --- CONFIGURATION ---
SYMBOL = "ETHUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
NB_DAYS = 30
ER_WINDOW = 10
Z_WINDOW = 24


def calculer_stats_regression(df_segment):
    """
    Calcule les stats de régression sur le segment donné (similaire à votre utils)
    """
    if len(df_segment) < 5: return 0, 0, 0, 0

    # 1. Pente et R2
    X = np.arange(len(df_segment)).reshape(-1, 1)
    y = df_segment['close'].values
    model = LinearRegression()
    model.fit(X, y)
    pente = model.coef_[0]
    r2 = model.score(X, y)

    # 2. Écart-type Standard ($)
    std_dollars = df_segment['close'].std()

    # 3. Volatilité Log (%)
    log_returns = np.log(df_segment['close'] / df_segment['close'].shift(1)).dropna()
    vol_log_pct = log_returns.std() * 100

    return pente, std_dollars, vol_log_pct, r2


def main():
    print(f"🚀 Connexion à MT5 pour analyse de {SYMBOL}...")
    if not mt5.initialize():
        print("❌ Échec de l'initialisation de MT5")
        return

    # 1. Récupération des données
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NB_DAYS * 24)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("❌ Aucune donnée récupérée")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # 2. Calcul des indicateurs techniques (pour vérifier la cohérence)
    # ER (10)
    direction = abs(df['close'] - df['close'].shift(ER_WINDOW))
    volatilite_somme = df['close'].diff().abs().rolling(window=ER_WINDOW).sum()
    df['ER'] = direction / volatilite_somme

    # Z-Score (24)
    rolling_mean = df['close'].rolling(window=Z_WINDOW).mean()
    rolling_std = df['close'].rolling(window=Z_WINDOW).std()
    df['zscore'] = (df['close'] - rolling_mean) / rolling_std

    resultats = []

    print("📊 Analyse des segments glissants...")
    # On boucle sur le dataframe pour calculer les stats de régression sur la fenêtre du Z-Score (24h)
    for i in range(Z_WINDOW, len(df)):
        # On prend le segment correspondant à la fenêtre du Z-Score
        df_seg = df.iloc[i - Z_WINDOW:i]

        pente, std_dlr, vol_log, r2 = calculer_stats_regression(df_seg)

        resultats.append({
            'timestamp': df.index[i],
            'close': df['close'].iloc[i],
            'ER_10': df['ER'].iloc[i],
            'ZScore_24': df['zscore'].iloc[i],
            'pente_reg': pente,
            'R2': r2,
            'vol_log_pct': vol_log
        })

    # 3. Sauvegarde et Analyse
    res_df = pd.DataFrame(resultats)
    output_file = "analyse_coherence_1H.csv"
    res_df.to_csv(output_file, sep=";", index=False)

    print(f"✅ Analyse terminée. Fichier enregistré : {output_file}")

    # --- TEST DE COHÉRENCE ---
    print("\n--- SYNTHÈSE DE COHÉRENCE ---")

    # Cohérence ER vs R2
    correl_er_r2 = res_df['ER_10'].corr(res_df['R2'])
    print(f"1. Corrélation entre ER(10) et R2(24): {correl_er_r2:.3f}")

    # Efficacité du seuil ER > 0.4
    r2_moyen_er_high = res_df[res_df['ER_10'] > 0.4]['R2'].mean()
    print(f"2. R2 moyen quand ER > 0.4: {r2_moyen_er_high:.3f} (Si > 0.5 = Très cohérent)")

    # Sensibilité du Z-Score
    vol_moyenne_z_high = res_df[res_df['ZScore_24'].abs() > 2]['vol_log_pct'].mean()
    vol_moyenne_totale = res_df['vol_log_pct'].mean()
    print(f"3. Vol Log quand |Z| > 2: {vol_moyenne_z_high:.3f} (vs Moyenne: {vol_moyenne_totale:.3f})")


if __name__ == "__main__":
    main()