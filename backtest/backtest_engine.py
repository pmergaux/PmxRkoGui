from datetime import datetime
import json
import os
import pickle
import time
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.style.core import available

from src.backtest.strategies.lstm_renko_strategy import LSTMRenkoBackStrategy
from src.utils.renko_utils import renko_start_date


def run_backtest(ticks_path, config_path="config_live.json"):
    start_time = time.time()
    num = 1

    with open(config_path, 'r') as f:
        config = json.load(f)

    df = pd.read_pickle(ticks_path)
    #print(df.tail(3))
    df = df[['bid']].copy()
    df = df.reset_index()
    df = df.rename(columns={'time_msc': 'time'})
    df = df.set_index('time', drop=False)
    df.index = pd.to_datetime(df.index, unit="ms")
    # calculer renko_start
    macd = config.get('macd', [12, 26, 8])
    lstm = config.get('lstm', {'seq_len':50, 'units': 100})
    renko_start = renko_start_date(df, config.get('renko_size', 17.1),
                                   max(macd[0] + macd[1], lstm.get('lstm', 50)) + 1,
                                   250)
    #print(renko_start, '\n', df.tail(3))
    # Configurer les  paramètres
    param = {'num': num, 'renko_start':renko_start, 'timeframe':'1m'} | config

    # Exécuter le backtest
    backStrategy = LSTMRenkoBackStrategy(param, df)
    positions, results = backStrategy.exec()
    with open('positions_'+str(num)+'.pkl', 'wb') as f:
        pickle.dump(positions, f)
    print(
        f"{datetime.now()} Backtest n°{num} pour renko_size={config['renko_size']:.2f} terminé en {time.time() - start_time:.2f}s, "
        f"total_profit={results['total_profit']:.2f} in trades={results['total_trades']} win_rate={results['win_rate']:.2f}")
    return {**param, **results}
"""
    total_profit = (final_value - 10000.0) / 10000.0

    report = {
        "total_profit": total_profit,
        "sharpe": strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        "max_drawdown": strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
        "nb_trades": strat.analyzers.trades.get_analysis()['total']['total']
    }

    print(f"BACKTEST TICK-BY-TICK → PROFIT: {total_profit:+.2%} | Sharpe: {report['sharpe']:.3f}")
    return report

    # --- COURBE ---
    fig = cerebro.plot(style='bar', volume=True)[0][0]
    fig.savefig("backtest/results/equity_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # --- SAUVEGARDE ---
    os.makedirs("backtest/results", exist_ok=True)
    with open("backtest/results/report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"BACKTEST OK → PROFIT: {total_profit:+.2%} | Sharpe: {report['sharpe']:.3f}")
    return report
"""