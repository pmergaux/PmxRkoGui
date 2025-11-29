# optimization/lstm_optimizer.py
import json
import multiprocessing as mp
from multiprocessing import Pool, Queue
import pickle
import os
import logging

import pandas as pd
import tensorflow as tf
import joblib
from reportlab.lib.pagesizes import elevenSeventeen
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#from functools import partial

from utils.renko_utils import tick21renko, colonnesRko
from decision.candle_decision import calculate_indicators, choix_features
from utils.lstm_utils import create_sequences, build_transformer, clean_features, build_transformer_tunable, \
    generate_param_combinations

log = logging.getLogger(__name__)

import traceback

def _optimize_single(grid, config, df_ticks):
    # création d'un param conforme à un param live
    def create_param(grid, config):
        param = {}
        parameters = {}
        lstm = {}
        for name, data in grid.items():
            if name[:4] == 'lstm':
                lstm[name]= data
            else:
                parameters[name]= data
        param["parameters"] = parameters
        param["lstm"] = lstm
        extrait = {cle: config[cle] for cle in ['features', 'target', 'open_rules', 'close_rules', 'live']}
        param.update(extrait)
        return param

    def send_log(msg):
        # Cette fonction n'est PAS dans le processus fils → on ne peut PAS envoyer via queue ici
        # → On retourne simplement None → le log sera dans run_optimization
        pass  # → On supprime les logs internes

    try:
        # === DEBUG : AFFICHER LES PARAMÈTRES ===

        renko_size = grid['renko_size']
        filename = f"data/renko_{renko_size:.2f}.pkl"
        """
        # --- CHARGER TICKS ---
        if not os.path.exists(tick_path):
            print('ops file inconnu')
            return None
        """
        # --- GÉNÉRER OU CHARGER RENKO ---
        if not os.path.exists(filename):
            #with open(tick_path, 'rb') as f:
            #    df_ticks = pickle.load(f)
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

        params = create_param(grid, config)
        # --- INDICATEURS ---
        df_bricks = calculate_indicators(df_bricks, params)
        df_bricks = choix_features(df_bricks, params).iloc[:-1]
        #print("bricks", df_bricks.columns.tolist())
        # --- controles ---
        utils_cols = []
        features_cols = []
        for col in params['features']:
            if col != "":
                if col not in df_bricks.columns:
                    raise f"features column {col} not in inputs"
                features_cols.append(col)
                utils_cols.append(col)
        target = params.get('target', None)
        if target is None:
            raise "No traget no optimization"
        target_include = True if target.get('target_include', 'True') == 'True' else False
        target_cols = target.get('target_col', 'close')
        if not isinstance(target_cols, list):
            target_cols = [target_cols]
        for col in target_cols:
            if col not in df_bricks.columns:   # tout est dans df or col not in features_cols:
                raise f"Target {col} not in inputs"
            if col not in features_cols:
                if target_include:
                    features_cols.append(col)
                utils_cols.append(col)
        # --- NETTOYAGE ---
        exclude_cols = [] # --- seront les colonnes à exclure
        all_cols = df_bricks.columns.tolist()
        exclude_cols.extend(col for col in all_cols if col not in features_cols and col not in target_cols)
        df_clean = df_bricks[exclude_cols].copy()  # --- on met de côté ---
        cleaned = clean_features(df_bricks, all_cols)  # --- on nettoie tout
        if cleaned.shape[1] < len(features_cols):    # combien a-t-il de colonnes
            print("ops shape < nb cols")
            return None
        # ?df_clean[features_cols] = cleaned[:, len(data_cols):]
        df_cleaned = cleaned[utils_cols]   # --- ne doivent restés que les colonnes utiles au modèle
        #print("bricks", df_cleaned.columns.tolist())
        # les indices de colonnes utiles au model
        target_idx = []
        for col in target_cols:
            target_idx.append(df_cleaned.columns.get_loc(col))
        input_idx = []
        for col in features_cols:
            input_idx.append(df_cleaned.columns.get_loc(col))
        # de quoi calculer le hash code
        parameters = {"renko_size": renko_size}
        if "EMA" in features_cols:
            parameters["ema_period"] = params["parameters"]["ema_period"]
        if "RSI" in features_cols:
            parameters['rsi_period'] = params["parameters"]["rsi_period"]
            parameters["rsi_high"] = params["parameters"].get("rsdi_high",70.0)
            parameters["rsi_low"] = params["parameters"].get("rsi_low", 30.0)
        if "MACS_line" in features_cols:
            parameters["macd"] = params["parameters"]["macd"]
        # pour l'instant pas les rules puisque les décisions se font sur lstm uniquement
        to_hash_code = {"parameters": parameters,"features": features_cols, "target": target_cols}
        # --- TRAIN/TEST ---
        split = int(0.8 * len(df_cleaned))
        if split < 100:
            print("ops split < 100")
            return None
        train_df = df_cleaned.iloc[:split]
        test_df = df_cleaned.iloc[split:]
        # --- SCALER ---
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_df)
        X_test = scaler.transform(test_df)
        # --- SÉQUENCES ---
        lstm = params.get('lstm', {'lstm_seq_len': 50, 'lstm_units': 100})
        target_type = target.get("target_type", 0)
        X_train_seq, y_train = create_sequences(X_train, lstm.get("lstm_seq_len"), input_idx, target_idx, target_type, target_include)
        X_test_seq, y_test = create_sequences(X_test, lstm.get("lstm_seq_len"), input_idx, target_idx, target_type, target_include)
        seq_len = lstm['lstm_seq_len']
        if len(X_train_seq) < seq_len:
            print(f"ops < {seq_len}")
            return None
        # --- MODÈLE ---
        model = build_transformer(seq_len=seq_len, features_len=len(features_cols))
        model.fit(X_train_seq, y_train, epochs=5, batch_size=16, verbose=0)
        """
        model.fit(X_train_seq, y_train,
                  epochs=100,
                  validation_split=0.2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
                  verbose=0)
        """
        """
        # === BACKTEST COMPLET ===
        from backtest.engine import full_backtest
        bt_results = full_backtest(model, scaler, test_df, config, seq_len=lstm['lstm_seq_len'])

        # === SAUVEGARDE AVEC HASH ===
        from core.model_manager import ModelManager
        manager = ModelManager()
        config_for_hash = {
            "renko_size": grid['renko_size'],
            "lstm_seq_len": lstm['lstm_seq_len'],
            "features": config['features'],
            "target": config['target']
        }
        hash_code, _ = manager.save(model, scaler, config_for_hash, bt_results)

        bt_results['hash'] = hash_code
        bt_results['renko_size'] = grid['renko_size']
        bt_results['seq_len'] = lstm['lstm_seq_len']

        return bt_results  # ← maintenant c’est un vrai backtest

    except Exception as e:
        return None
        """

        # FORCER LA COMPILATION DU PREDICT UNE SEULE FOIS
        model.make_predict_function()  # ← LÀ, C'EST FINI. Plus jamais de retracing
        # --- PRÉDICTION ---
        proba = model.predict(X_test_seq, verbose=0).flatten()
        signal = np.where(proba > 0.6, 1, np.where(proba < 0.4, -1, 0))

        # --- BACKTEST ---
        df_bt = test_df.iloc[seq_len:].copy()
        bt_col = ''
        if 'close' in target_cols:
            bt_col = 'close'
        elif 'EMA' in target_cols:
            bt_col = 'EMA'
        else:
            df_tmp = df_clean['close'].iloc[-seq_len:]
            bt_col = 'close'
            df_bt = pd.concat([df_bt, df_tmp], axis=1)
        df_bt = df_bt.reset_index(drop=True)
        if len(df_bt) != len(signal):
            print('ops bt # signal')
            return None

        df_bt['signal'] = signal
        # 1. Position = signal décalé (connu à la clôture précédente)
        df_bt['position'] = df_bt['signal'].shift(1).fillna(0)  # ← fillna(0) pour le premier row
        # 2. Retour marché
        df_bt['market_return'] = df_bt[bt_col].pct_change().fillna(0)
        # 3. Retour stratégie
        df_bt['strategy_return'] = df_bt['position'] * df_bt['market_return']
        # 4. Frais seulement quand tu changes de position
        changes = df_bt['position'].diff().abs().fillna(0)
        df_bt['fees'] = changes * 0.0003  # 0.03 % par changement (buy/sell)
        df_bt['strategy_return'] -= df_bt['fees']
        # 5. Si signal == 0 → position == 0 → strategy_return == 0 → fees == 0 (sauf si tu sors d’une position)
        # Equity curve
        df_bt['equity'] = (1 + df_bt['strategy_return']).cumprod()
        df_bt['cum_pnl'] = (1 + df_bt['strategy_return']).cumprod() - 1

        sharpe = df_bt['strategy_return'].mean() / (df_bt['styrategy_return'].std() or 1e-8) * np.sqrt(252)

        result = {
            'renko_size': renko_size,
            'seq_len': seq_len,
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
    params, ticks_data, config = args
    return _optimize_single(params, config, ticks_data)


def run_optimization(param_grid, config, ticks_data, queue=None):
    """
    Retourne TOUS les résultats valides + le meilleur
    Envoie aussi les logs/progress via queue si fournie (pour GUI)
    """
    tasks = [(p, ticks_data, config) for p in generate_param_combinations(param_grid)]
    all_results = []  # ← TOUS les résultats (même les mauvais)
    valid_results = []  # ← seulement ceux qui ont réussi

    def send_log(msg):
        if queue:
            queue.put({'type': 'log', 'text': msg})

    def send_progress(pct):
        if queue:
            queue.put({'type': 'progress', 'value': pct})

    send_log(f"DÉMARRAGE OPTIMISATION → {len(tasks)} combinaisons")

    n_jobs = min(8, mp.cpu_count())  # 8 est le sweet spot sur 99 % des machines
    with Pool(n_jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_optimize_single_wrapper, tasks)):
            if res is not None:
                valid_results.append(res)
                all_results.append(res)
                send_log(f"[{i + 1}/{len(tasks)}] OK → Sharpe={res['sharpe']:.3f} | Renko={res['renko_size']}")
            else:
                all_results.append(None)
                send_log(f"[{i + 1}/{len(tasks)}] ÉCHEC")

            send_progress(int((i + 1) / len(tasks) * 100))

    # === CLASSEMENT & ENVOI FINAL ===
    if valid_results:
        best = max(valid_results, key=lambda x: x['sharpe'])
        top5 = sorted(valid_results, key=lambda x: x['sharpe'], reverse=True)[:5]

        final = {
            'best': best,
            'top5': top5,
            'all_valid': valid_results,  # ← tous les bons
            'total_combinations': len(tasks),
            'success_rate': len(valid_results) / len(tasks)
        }

        send_log(f"TERMINÉ → {len(valid_results)}/{len(tasks)} réussis")
        send_log(f"MEILLEUR → Sharpe={best['sharpe']:.3f} | Renko={best['renko_size']} | Seq={best['seq_len']}")
        if queue:
            queue.put({'type': 'result', 'data': final})
            queue.put({'type': 'done', 'data': None})  # signal de fin
        print("final\n", final)
        return final

    else:
        error = {'error': 'Aucun modèle viable'}
        send_log("ÉCHEC TOTAL – Aucun résultat valide")
        if queue:
            queue.put({'type': 'result', 'data': error})
            queue.put({'type': 'done', 'data': None})
        return error

# FONCTION HORS CLASSE → PICKLABLE À 100%
def run_optimization_in_process(param_grid, config, ticks_data, queue):
    def log(msg):
        queue.put({'type': 'log', 'text': msg})
    def prog(pct):
        queue.put({'type': 'progress', 'value': pct})

    log("Processus fils démarré – optimisation en cours...")

    # Import ici pour éviter les problèmes de pickling
    from optimize.lstm_optimizer import run_optimization

    result = run_optimization(param_grid, config, ticks_data, queue)

    #queue.put({'type': 'result', 'data': result})
    #queue.put({'type': 'done', 'data': None})
    log("Processus fils terminé")