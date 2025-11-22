# optimization/lstm_optimizer.py
import json
import multiprocessing as mp
from multiprocessing import Pool, Queue
import pickle
import os
import logging

import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#from functools import partial

from utils.renko_utils import tick21renko
from decision.candle_decision import calculate_indicators, choix_features
from utils.lstm_utils import create_sequences, build_transformer, clean_features, build_transformer_tunable, \
    generate_param_combinations
import traceback

log = logging.getLogger(__name__)

# optimization/lstm_optimizer.py
import traceback

def _optimize_single(params, tick_path):
    def send_log(msg):
        # Cette fonction n'est PAS dans le processus fils → on ne peut PAS envoyer via queue ici
        # → On retourne simplement None → le log sera dans run_optimization
        pass  # → On supprime les logs internes

    try:
        # === DEBUG : AFFICHER LES PARAMÈTRES ===

        renko_size = params['renko_size']
        filename = f"data/renko_{renko_size:.2f}.pkl"

        # --- CHARGER TICKS ---
        if not os.path.exists(tick_path):
            print('ops file inconnu')
            return None

        # --- GÉNÉRER OU CHARGER RENKO ---
        if not os.path.exists(filename):
            with open(tick_path, 'rb') as f:
                df_ticks = pickle.load(f)
            df_bricks = tick21renko(df_ticks, None, step=renko_size)
            if len(df_bricks) < 100:
                print("ops rko < 100")
                return None
            with open(filename, 'wb') as f:
                pickle.dump(df_bricks, f)
        else:
            with open(filename, 'rb') as f:
                df_bricks = pickle.load(f)

        if len(df_bricks) < 100:
            print("ops rko < 200")
            return None

        # --- INDICATEURS ---
        df_bricks = calculate_indicators(df_bricks, params)
        df_bricks = choix_features(df_bricks, params).iloc[:-1]
        # --- controles ---
        features_cols = params['features']
        ohlc_cols = ['time', 'open_renko','open', 'high', 'low', 'close', 'close_renko']
        target = params.get('target', None)
        if target is None:
            raise "No traget no optimization"
        target_include = True if target.get('include_in_features', 'True') == 'True' else False
        target_cols = target.get('column', 'close')
        if not isinstance(target_cols, list):
            target_cols = [target_cols]
        for col in target_cols:
            if col not in df_bricks.columns or col not in features_cols:
                raise "Target not in data"
        # --- NETTOYAGE ---
        data_cols = [] # --- seront les colonnes à exclure
        data_cols.extend(col for col in ohlc_cols if col not in features_cols)
        all_cols = data_cols + features_cols

        df_clean = df_bricks[data_cols].copy()  # --- on met de côté ---
        cleaned = clean_features(df_bricks, all_cols)  # --- on nettoie tout
        if cleaned.shape[1] < len(features_cols):    # combien a-t-il de colonnes
            print("ops shape < nb cols")
            return None
        # ?df_clean[features_cols] = cleaned[:, len(data_cols):]
        df_clean = df_clean[features_cols]   # --- ne doivent restés que les colonnes utiles au modèle
        # --- TRAIN/TEST ---
        split = int(0.8 * len(df_clean))
        if split < 100:
            print("ops split < 100")
            return None
        train_df = df_clean.iloc[:split]
        test_df = df_clean.iloc[split:]
        # --- SCALER ---
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_df)
        X_test = scaler.transform(test_df)
        # --- SÉQUENCES ---
        lstm = params.get('lstm', {'seq_len': 50, 'units': 100})
        X_train_seq, y_train = create_sequences(X_train,lstm['seq_len'], target, features_cols)
        X_test_seq, y_test = create_sequences(X_test, lstm['seq_len'], target, features_cols)

        if len(X_train_seq) < 50:
            print("ops < 50")
            return None

        # --- MODÈLE ---
        model = build_transformer(seq_len=params['seq_len'], features_len=len(features_cols))
        model.fit(X_train_seq, y_train, epochs=5, batch_size=16, verbose=0)

        # --- PRÉDICTION ---
        proba = model.predict(X_test_seq, verbose=0).flatten()
        signal = np.where(proba > 0.5, 1, -1)

        # --- BACKTEST ---
        df_bt = test_df.iloc[lstm['seq_len']:].copy().reset_index(drop=True)
        if len(df_bt) != len(signal):
            print('ops bt # signal')
            return None

        df_bt['signal'] = signal
        df_bt['next_close'] = df_bt['close'].shift(-1)
        df_bt['return'] = np.where(
            df_bt['signal'] == 1,
            (df_bt['next_close'] - df_bt['close']) / df_bt['close'],
            (df_bt['close'] - df_bt['next_close']) / df_bt['close']
        )
        df_bt['return'] = df_bt['return'].fillna(0) - 0.0005
        df_bt['cum_pnl'] = (1 + df_bt['return']).cumprod() - 1

        sharpe = df_bt['return'].mean() / (df_bt['return'].std() or 1e-8) * np.sqrt(252)

        result = {
            'renko_size': renko_size,
            'seq_len': params['seq_len'],
            'pnl': float(df_bt['cum_pnl'].iloc[-1]),
            'sharpe': float(sharpe),
            'nb_trades': int(len(signal))
        }

        return result

    except Exception as e:
        error_msg = f"CRASH → {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print("Err op Single", error_msg)
        return None

# === FONCTION NOMMÉE (PICKLABLE) ===
def _optimize_single_wrapper(args):
    params, ticks_path = args
    return _optimize_single(params, ticks_path)

def run_optimization(param_grid, ticks_path, queue):
    tasks = [(p, ticks_path) for p in generate_param_combinations(param_grid)]
    results = []

    def send_log(msg):
        queue.put({'type': 'log', 'text': msg})

    def send_progress(pct):
        queue.put({'type': 'progress', 'value': pct})

    send_log(f"DÉMARRAGE → {len(tasks)} tâches avec {ticks_path}")

    with Pool(mp.cpu_count()) as pool:
        # UTILISE LA FONCTION NOMMÉE
        for i, res in enumerate(pool.imap_unordered(_optimize_single_wrapper, tasks)):
            if res:
                results.append(res)
                send_log(f"[{i+1}/{len(tasks)}] OK → Sharpe={res['sharpe']:.3f}")
            else:
                send_log(f"[{i+1}/{len(tasks)}] ÉCHEC")
            send_progress(int((i + 1) / len(tasks) * 100))

    if results:
        best = max(results, key=lambda x: x['sharpe'])
        send_log(f"MEILLEUR → Sharpe={best['sharpe']:.3f}")
        queue.put({'type': 'result', 'data': best})
    else:
        send_log("AUCUN RÉSULTAT VALIDE")
        queue.put({'type': 'result', 'data': {'error': 'Aucun modèle viable'}})
