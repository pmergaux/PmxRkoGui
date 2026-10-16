import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Permet d'importer les modules du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def calculate_rolling_stats(prices, window):
    n = len(prices)
    y = prices.values
    x = np.arange(window)
    x_mean = (window - 1) / 2.0
    x_diff = x - x_mean
    ss_x = np.sum(x_diff ** 2)
    
    r2s = np.zeros(n)
    pentes = np.zeros(n)
    
    for i in range(window, n):
        y_seg = y[i-window:i]
        y_mean = np.mean(y_seg)
        y_diff = y_seg - y_mean
        ss_xy = np.sum(x_diff * y_diff)
        pente = ss_xy / ss_x if ss_x != 0 else 0
        
        y_hat = pente * x_diff + y_mean
        ss_res = np.sum((y_seg - y_hat) ** 2)
        ss_tot = np.sum(y_diff ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        r2s[i-1] = r2
        pentes[i-1] = pente
        
    return pentes, r2s

def analyze_r2(df, window_sizes, r2_thresholds, hold_periods):
    print(f"Analyse sur {len(df)} périodes...")
    results = []
    
    for w in window_sizes:
        # Calcul rapide de pente et r2
        pentes, r2s = calculate_rolling_stats(df['close'], w)
        
        df_temp = pd.DataFrame({
            'close': df['close'].values,
            'r2': r2s,
            'pente': pentes
        })
        
        for r2_thresh in r2_thresholds:
            # Signal de retournement/nouvelle tendance: le R2 passe au-dessus du seuil
            signal_mask = (df_temp['r2'] > r2_thresh) & (df_temp['r2'].shift(1) <= r2_thresh)
            
            # Filtrer pour ne garder que les points valides
            signal_indices = df_temp.index[signal_mask].tolist()
            
            if not signal_indices:
                continue
                
            for hold_n in hold_periods:
                profits = []
                wins = 0
                
                for idx in signal_indices:
                    if idx + hold_n < len(df_temp):
                        entry_price = df_temp['close'].iloc[idx]
                        exit_price = df_temp['close'].iloc[idx + hold_n]
                        
                        # Direction définie par la pente au moment du signal
                        pente_signal = df_temp['pente'].iloc[idx]
                        direction = 1 if pente_signal > 0 else -1
                        
                        profit = (exit_price - entry_price) * direction
                        profits.append(profit)
                        if profit > 0:
                            wins += 1
                
                if profits:
                    avg_profit = np.mean(profits)
                    win_rate = wins / len(profits) * 100
                    results.append({
                        'Window': w,
                        'R2_Threshold': r2_thresh,
                        'Hold_Periods': hold_n,
                        'Trades': len(profits),
                        'WinRate(%)': win_rate,
                        'Avg_Profit($)': avg_profit
                    })
                    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Trier par le meilleur profit moyen
        df_results = df_results.sort_values(by='Avg_Profit($)', ascending=False)
        print("\n=== Meilleurs Réglages (Triés par Profit Moyen) ===")
        print(df_results.head(15).to_string(index=False))
        
        # Optionnel : Sauvegarder en CSV
        df_results.to_csv("r2_analysis_results.csv", index=False)
        print("\nRésultats complets sauvegardés dans 'r2_analysis_results.csv'")
    else:
        print("Aucun résultat trouvé.")

if __name__ == "__main__":
    # Prendre les données en chandeliers japonais
    data_path = os.path.expanduser("~/data/df_ETHUSD.pkl")

    if not os.path.exists(data_path):
        print(f"Erreur : Fichier de données introuvable ({data_path}).")
        sys.exit(1)
        
    print(f"Chargement des données depuis : {data_path}")
    df_raw = pd.read_pickle(data_path)
    
    print("Conversion des données en chandeliers 1H...")
    df = df_raw.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Paramètres à tester pour des bougies de 1 Heure
    WINDOWS = [6, 8, 12, 14, 18, 20, 24] # Fenêtres en heures
    R2_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
    HOLD_PERIODS = [2, 3, 5, 8] # Combien d'heures garder la position
    
    # On prend tout l'historique 1H disponible (ici ça fait ~1000-2000 bougies)
    analyze_r2(df, WINDOWS, R2_THRESHOLDS, HOLD_PERIODS)

