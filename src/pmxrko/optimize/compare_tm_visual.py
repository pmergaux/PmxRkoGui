import matplotlib.pyplot as plt
import pandas as pd


def visual_validation(df_raw):
    # Préparation des deux échelles
    df_1h = df_raw.resample('1H').last().dropna()
    df_30m = df_raw.resample('30T').last().dropna()

    # Calcul R2 (avec vos paramètres actuels, ex: 14)
    df_1h['r2'] = calculate_r2(df_1h['close'], window=14)
    df_30m['r2'] = calculate_r2(df_30m['close'], window=14)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Graphique 1h
    ax1.plot(df_1h.index, df_1h['close'], label="Prix 1H", alpha=0.3)
    ax1_r2 = ax1.twinx()
    ax1_r2.plot(df_1h.index, df_1h['r2'], color='red', label="R2 1H")
    ax1_r2.axhline(0.85, color='green', linestyle='--')
    ax1.set_title("Réactivité R2 en 1 Heure")

    # Graphique 30min
    ax2.plot(df_30m.index, df_30m['close'], label="Prix 30min", alpha=0.3)
    ax2_r2 = ax2.twinx()
    ax2_r2.plot(df_30m.index, df_30m['r2'], color='blue', label="R2 30min")
    ax2_r2.axhline(0.85, color='green', linestyle='--')
    ax2.set_title("Réactivité R2 en 30 Minutes")

    plt.show()
