# simple_optimize_pierre2026_COMPLET.py
# Auteur : Pierre (83 ans) — Version finale — Prêt à lancer
# Objectif : 100 essais → trouve le Graal ou meurt
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import random
import json
import os
from datetime import datetime

from pytorch_forecasting import QuantileLoss
VERSIONS = ['SIMPLE', 'LSTM', 'TRIPLE']
VERSION = VERSIONS[0]
# ====================== TES FONCTIONS EXISTANTES (à garder) ======================
from utils.renko_utils import tick21renko
from utils.utils import reload_ticks_from_pickle
from utils.lstm_utils import scale_features_only, assemble_with_targets, create_sequences_numba
from decision.candle_decision import add_indicators  # ← ton add_indicators Numba parfait

# ====================== DONNÉES ======================
fbrick = os.path.join('models/', 'simple_model_*.keras')
nbrick = glob.glob(fbrick)
if nbrick:
    try:
        for fic in nbrick:
            os.remove(fic)
    except OSError as e:
        print(f"err supr fichiers {e}")
fbrick = os.path.join('models/', 'simple_scaler_*.pkl')
nbrick = glob.glob(fbrick)
if nbrick:
    try:
        for fic in nbrick:
            os.remove(fic)
    except OSError as e:
        print(f"err supr fichiers {e}")

start = datetime(2025, 9, 1)
end = datetime(2025, 11, 20, 23, 59, 59)
base_name = f"../data/ETHUSD_{start.strftime('%Y_%m_%d_%H_%M_%S')}_{end.strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
df = reload_ticks_from_pickle(base_name, 'ETHUSD', None, start, end)
if df is None or df.empty:
    print("Pas de données → exit")
    exit()
df['time'] = pd.to_datetime(df['time'])
# =========================================================== Les paramètres
features_cols = ["EMA", "RSI", "MACD_hist", "time_live"]
params = {"ema_period": 9, "rsi_period": 14, "rsi_high": 70, "rsi_low": 30, "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}}
# ================== TARGET SIMPLE ==================
target_col = "target_sign_mean"
target_cols = [target_col]
# ====================== TARGETS SIMPLES ======================
def prepare_targets_simple(df, horizon=5):
    df = df.copy()
    df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
    df[target_col] = np.sign(df['future_return'])
    df[target_col] = df[target_col].replace(-1, 0)  # pour n'avoir que 0 ou 1
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df

# ==================================================================
# 3. MODÈLES RENFORCÉS
# ==================================================================

# ---------- LSTM AMÉLIORÉ (le seul qui marche vraiment en trading) ----------
def lstm_train_model(X_train, y_train, X_val, y_val, units, seq_len, features_len, dropout=0.2):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_len, features_len)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(units // 2),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')  # ← tanh mieux que sigmoid pour direction
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)
    return model
# ================== LSTM ULTRA-SIMPLE MAIS QUI GAGNE ==================
def lstm_train(X_tr, y_tr, X_va, y_va, seq, feats):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq, feats)),
        tf.keras.layers.LSTM(96, return_sequences=True),
        tf.keras.layers.LSTM(48),
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='huber')
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=35, batch_size=64, verbose=0)
    return model
# ---------- TEMPORAL FUSION TRANSFORMER (le roi 2025) ----------
def tft_train_predict(train_df, val_df, test_df, feature_cols, target_col='target_sign_mean', max_epochs=20):
    try:
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping

        for d in [train_df, val_df, test_df]:
            d["time_idx"] = np.arange(len(d))
            d["group"] = 0

        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target_col,
            group_ids=["group"],
            max_encoder_length=120,
            max_prediction_length=1,
            static_reals=[],
            time_varying_known_reals=feature_cols,
            time_varying_unknown_reals=[target_col],
            target_normalizer=GroupNormalizer(groups=["group"]),
        )

        val_dataset = TimeSeriesDataSet.from_dataset(training, pd.concat([train_df, val_df]), predict=True)
        train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=3e-4,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            loss=QuantileLoss(),
            log_interval=10,
        )

        trainer = Trainer(max_epochs=max_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                          enable_model_summary=False, callbacks=[EarlyStopping(monitor="val_loss", patience=5)])
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred = tft.predict(val_dataset.to_dataloader(train=False, batch_size=64), mode="quantiles")
        return pred.numpy().flatten()
    except:
        return None

# ---------- N-BEATS (excellent sur signaux financiers) ----------
def nbeats_train_predict(train_series, val_series, test_series):
    try:
        from darts import TimeSeries
        from darts.models import NBEATSModel

        model = NBEATSModel(input_chunk_length=120, output_chunk_length=5, n_neurons=256,
                            num_blocks=4, num_layers=4, layer_widths=512, loss_fn=torch.nn.MSELoss())
        model.fit(train_series, val_series=val_series, epochs=30, verbose=False)
        pred = model.predict(n=1, series=val_series)
        return pred.values().flatten()
    except:
        return None

# ==================================================================
# 4. BACKTEST RÉALISTE
# ==================================================================
# commence par un backtest simple
def backtest(df_test, proba, buy_thr=0.62, sell_thr=0.38, stop_loss_pct=0.018, spread_dollar=0.025):
    """
    df_test : DataFrame avec colonne 'close' (vrai prix du brick)
    proba   : np.array de même longueur que df_test
    → Tout est vrai. Rien n’est inventé.
    """
    pos = 0
    pnl = []
    entry_price = 0.0

    for i in range(len(df_test)):
        close = df_test['close'].iloc[i]
        p = proba[i]
        # === GESTION POSITION ===
        if p > buy_thr and pos == 0:
            pos = 1
            entry_price = close + spread_dollar   # on paye le spread à l'entrée
            continue
        elif p < sell_thr and pos == 0:
            pos = -1
            entry_price = close - spread_dollar
            continue
        elif 0.44 <= p <= 0.56 and pos != 0:
            # sortie forcée si signal neutre
            pnl.append(pos * (close - entry_price))
            pos = 0
            continue
        # === STOP LOSS EN % ===
        if pos != 0:
            unrealized = pos * (close - entry_price)
            if unrealized / entry_price < -stop_loss_pct:
                pnl.append(unrealized)  # on paye encore à la sortie
                pos = 0
                continue
        # === PNL DU BRICK (seulement si en position) ===
        if pos != 0:
            brick_pnl = pos * (close - entry_price)
            pnl.append(brick_pnl)
            entry_price = close  # mise à jour pour le prochain brick

    # === Si on sort à la fin ===
    if pos != 0:
        final_close = df_test['close'].iloc[-1]
        final_pnl = pos * (final_close - entry_price) - spread_dollar/2
        pnl.append(final_pnl)
    a = np.array(pnl)
    if len(a) == 0 or a.std() == 0:
        return -999999
    sharpe = a.mean() / a.std() * np.sqrt(365 * 390)
    return sharpe * 1000

# ================== LSTM ULTRA-SIMPLE MAIS QUI GAGNE ==================
"""
Règle,Pourquoi c’est obligatoire
Position unique,Tu ne peux pas être long ET short en même temps
Stop Loss 2%,Sinon tu exploses sur un gros brick
Sortie si signal neutre,Évite de rester en position sur du bruit
Spread à chaque trade,C’est la réalité du marché
Sharpe annualisé,Comparaison juste entre stratégies
Max Drawdown pénalisé,Un DD de -40% = tu es viré
"""
def realistic_backtest(returns, proba, thresh_buy=0.6, thresh_sell=0.4, spread=2.5):
    position = 0
    equity = 0.0
    daily_pnl = []

    for r, p in zip(returns, proba):
        cost = spread / 10000 if position == 0 else 0

        if p > thresh_buy and position <= 0:
            position = 1
        elif p < thresh_sell and position >= 0:
            position = -1
        elif 0.45 <= p <= 0.55 and position != 0:
            position = 0
            cost += spread / 10000

        if position != 0 and position * r < -0.02:  # Stop Loss 2%
            equity += position * r
            daily_pnl.append(position * r)
            position = 0
            continue

        pnl = position * r - cost
        equity += pnl
        daily_pnl.append(pnl)

    daily_pnl = np.array(daily_pnl)
    if len(daily_pnl) == 0 or daily_pnl.std() == 0:
        return float('-inf')

    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365 * 390)
    pf = daily_pnl[daily_pnl > 0].sum() / abs(daily_pnl[daily_pnl < 0].sum()) if daily_pnl[daily_pnl < 0].sum() < 0 else 10
    equity_curve = np.cumsum(daily_pnl)
    max_dd = abs((equity_curve - np.maximum.accumulate(equity_curve)).min())

    score = sharpe * 1000 + pf * 200 - max_dd * 5000
    return score

def realistic_backtest_t(returns, proba, threshold_buy=0.6, threshold_sell=0.4, spread_pips=2.5):
    position = 0
    equity = 0
    trades = []
    for r, p in zip(returns, proba):
        if p > threshold_buy and position <= 0:
            position = 1
            entry_cost = spread_pips
        elif p < threshold_sell and position >= 0:
            position = -1
            entry_cost = spread_pips
        elif position != 0 and abs(p - 0.5) < 0.05:  # sort si neutre
            position = 0

        if position != 0:
            equity += position * r
            if position * r < -0.02:  # stop loss 2%
                equity -= 0.02
                position = 0

        if position != 0 and len(trades) > 0 and trades[-1][0] != position:
            trades.append([position, r, p])

    returns_series = pd.Series([position * r for r, p in zip(returns, proba)])
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252 * 390) if returns_series.std() > 0 else 0
    profit_factor = returns_series[returns_series > 0].sum() / abs(returns_series[returns_series < 0].sum()) if returns_series[returns_series < 0].sum() < 0 else 10
    calmar = returns_series.cumsum().max() / returns_series.cumsum().min() if returns_series.cumsum().min() < 0 else 10

    return sharpe * 1000 + profit_factor * 100 + calmar * 50

# ==================================================================
# 5. ÉVALUATION FINALE — LA VÉRITÉ
# ==================================================================
def evaluate_config(config):
    try:
        # 0. standardiser la config par exemple si on doit utiliser le hcode
        renko_size = config['renko_size']
        seq_len = config['seq_len']
        units = config['lstm_units']
        thresh_buy = config['thresh_buy']
        thresh_sell = config['thresh_sell']

        # 1. Renko + indicateurs
        df_bricks = tick21renko(df, None, renko_size, 'bid')
        df_renko = add_indicators(df_bricks)
        df_renko = prepare_targets_simple(df_renko, horizon=5)
        if len(df_renko) < 1000:
            return float('-inf')

        # 2. Features
        # mettre ici les déterminations des colonnes features et targets
        # prepare_target est déjà faite
        # 3. Split
        train_len = int(len(df_renko) * 0.65)
        val_len = int(len(df_renko) * 0.15)
        train_df = df_renko.iloc[:train_len]
        val_df = df_renko.iloc[train_len:train_len + val_len]
        test_df = df_renko.iloc[train_len + val_len:]
        # 4. Scale
        scaler_path = f"models/scaler_temp_{renko_size:.1f}.pkl"
        X_train, X_val, X_test = scale_features_only(train_df, val_df, test_df, features_cols, scaler_path)
        train_r, val_r, test_r = assemble_with_targets(X_train, X_val, X_test, train_df, val_df, test_df, target_cols)
        # 5. séquences
        X_train_seq, y_train_seq = create_sequences_numba(train_r, seq_len, len(features_cols))
        X_val_seq, y_val_seq = create_sequences_numba(val_r, seq_len, len(features_cols))
        X_test_seq, _ = create_sequences_numba(test_r, seq_len, len(features_cols))

        if len(X_train_seq) < 50:
            return float('-inf')

        # 6. Returns pour backtest
        returns = np.diff(test_df['close'].values[-len(X_test_seq):])

        # 5. Train + predict
        if VERSION==VERSIONS[0]:
            model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
            pred = model.predict(X_test_seq, verbose=0).flatten()
            proba = (pred + 1) / 2
            score = backtest(test_df.iloc[-len(proba):], proba, thresh_buy, thresh_sell)
        elif VERSION==VERSIONS[1]:
            model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
            proba_lstm = model.predict(X_test_seq, verbose=0).flatten()
            proba_lstm = (proba_lstm + 1) / 2  # tanh → [0,1]
            score = realistic_backtest(returns, proba_lstm, thresh_buy, thresh_sell)
        else:
            # === LSTM ===
            model = lstm_train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, units, seq_len, len(features_cols))
            proba_lstm = model.predict(X_test_seq, verbose=0).flatten()
            proba_lstm = (proba_lstm + 1) / 2  # tanh → [0,1]
            # === TFT ===
            proba_tft = tft_train_predict(train_df, val_df, test_df, features_cols, target_col)
            if proba_tft is None: proba_tft = np.full(len(proba_lstm), 0.5)
            # === N-BEATS ===
            try:
                from darts import TimeSeries
                train_ts = TimeSeries.from_dataframe(train_df, value_cols=target_cols)
                val_ts = TimeSeries.from_dataframe(val_df, value_cols=target_cols)
                proba_nbeats = nbeats_train_predict(train_ts, val_ts, None)
                if proba_nbeats is None or len(proba_nbeats) != len(proba_lstm):
                    proba_nbeats = np.full(len(proba_lstm), 0.5)
            except:
                proba_nbeats = np.full(len(proba_lstm), 0.5)
            # === ENSEMBLE ===
            proba = (proba_lstm * 0.5 + proba_tft * 0.3 + proba_nbeats * 0.2)
            proba = np.clip(proba, 0.01, 0.99)
            # 7. Backtest réaliste
            score = realistic_backtest(returns, proba, thresh_buy, thresh_sell)

        print(f"{renko_size:5.1f} | {seq_len:3d} | {units:3d} | {thresh_buy:.3f}/{thresh_sell:.3f} → Score {score:8.1f}")
        return score

    except Exception as e:
        print("ERREUR →", e)
        return float('-inf')

# ====================== MAIN — 100 ESSAIS — LE GRAAL ======================
if __name__ == "__main__":
    print("DÉBUT OPTIMISATION PIERRE 2026 — 100 ESSAIS")

    best_score = float('-inf')
    best_config = None
    results = []

    for i in range(100):
        config = {
            'renko_size': round(random.uniform(12, 45), 1),
            'seq_len': random.randint(20, 120),
            'lstm_units': random.randint(64, 256),
            'thresh_buy': round(random.uniform(0.55, 0.75), 3),
            'thresh_sell': round(random.uniform(0.25, 0.45), 3),
        }

        score = evaluate_config(config)
        results.append((score, config))

        if score > best_score:
            best_score = score
            best_config = config.copy()
            print(f"NOUVEAU RECORD → {score:.1f}")

    # ==================== SAUVEGARDE ====================
    os.makedirs("models/simple_opt", exist_ok=True)
    with open("models/simple_opt/best_pierre2026.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print("\nMEILLEURE CONFIG TROUVÉE:")
    print(json.dumps(best_config, indent=2))
    print(f"Score final: {best_score:.1f}")

    # Top 5
    for score, cfg in sorted(results,key=lambda x: x[0], reverse=True)[:5]:
        print(f"{score:8.1f} → {cfg['renko_size']:.1f} | {cfg['seq_len']} | {cfg['lstm_units']} | {cfg['thresh_buy']}/{cfg['thresh_sell']}")

