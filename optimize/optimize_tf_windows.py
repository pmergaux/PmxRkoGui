import optuna
import pandas as pd

from backtest.strategy_simulator import StrategySimulator


def objective(trial):
    # 1. Variables de l'optimisation
    tf = trial.suggest_categorical("timeframe", ["15T", "30T", "1H", "2H"])
    w_r2 = trial.suggest_int("window_r2", 10, 35)
    w_sar = trial.suggest_int("window_sar", 10, 25)

    # 2. Chargement des données historiques
    # On suppose que vous avez un CSV avec 'close' et 'pred_proba' (RNN fixe)
    df_raw = pd.read_csv("data/history_with_probas.csv", index_col=0, parse_dates=True)

    # Resampling au timeframe choisi
    df_res = df_raw.resample(tf).agg({'close': 'last', 'pred_proba': 'mean'}).dropna()

    # 3. Simulation
    sim = StrategySimulator()
    profit = sim.simulate(df_res, window_r2=w_r2, window_sar=w_sar)

    return profit


# Lancement de la recherche
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Top Params: {study.best_params}")
