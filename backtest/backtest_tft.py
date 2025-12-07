import os
import shutil
import torch
import gc
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
import numpy as np


def to_renko(df, brick_size):
    """Convertit ticks (time, close) en Renko OHLC dataframe basique (sans volume, comme tes features)."""
    df = df[['time', 'close']].copy().dropna().reset_index(drop=True)

    renko = []
    start_price = df['close'][0]
    current_brick_close = start_price
    brick_time = df['time'][0]

    for i in range(1, len(df)):
        price = df['close'][i]
        time = df['time'][i]

        diff = price - current_brick_close
        if abs(diff) < brick_size:
            continue  # pas assez de mouvement

        num_bricks = int(abs(diff) / brick_size)
        direction = np.sign(diff)

        for _ in range(num_bricks):
            brick_open = current_brick_close
            brick_close = current_brick_close + direction * brick_size
            brick_high = max(brick_open, brick_close)
            brick_low = min(brick_open, brick_close)

            renko.append({
                'time': brick_time,
                'open': brick_open,
                'high': brick_high,
                'low': brick_low,
                'close': brick_close,
            })

            current_brick_close = brick_close
            brick_time = time

    # Ajout du dernier brick si besoin
    if renko:
        renko_df = pd.DataFrame(renko)
        renko_df['time_idx'] = range(len(renko_df))  # pour TFT
        renko_df['time_max'] = renko_df['time'].max()  # pour cutoff
        renko_df['symbol'] = 'ETHUSD'  # groupe fixe
        renko_df['time_live'] = (renko_df['time'] - renko_df['time'].min()).dt.total_seconds() / 3600  # ex. en heures

        return renko_df
    else:
        raise ValueError("Pas assez de mouvement pour créer des Renko.")


def run_backtest(config):
    """
    Version corrigée : ticks → Renko → features → TFT.
    Retourne: (score, chemin_du_meilleur_checkpoint.ckpt ou None)
    """
    df = config['data']['data'] if isinstance(config['data'], dict) else config['data']
    df = df.copy()  # sécurité

    try:
        # ==================================================================
        # 1. Création des Renko à partir des ticks
        # ==================================================================
        renko_df = to_renko(df, config['renko_size'])

        # Ajout des features (EMA, RSI, MACD_hist) sur Renko
        # (utilise pandas_ta ou ta-lib si installé ; ici pandas simple pour EMA ex.)
        renko_df['EMA'] = renko_df['close'].ewm(span=config['ema_period'], adjust=False).mean()
        renko_df['delta'] = renko_df['close'].diff()
        renko_df['up'] = np.where(renko_df['delta'] > 0, renko_df['delta'], 0)
        renko_df['down'] = np.where(renko_df['delta'] < 0, -renko_df['delta'], 0)
        renko_df['avg_up'] = renko_df['up'].ewm(alpha=1 / config['rsi_period']).mean()
        renko_df['avg_down'] = renko_df['down'].ewm(alpha=1 / config['rsi_period']).mean()
        renko_df['RSI'] = 100 - 100 / (1 + renko_df['avg_up'] / renko_df['avg_down'])

        # MACD_hist (ex. simple)
        ema12 = renko_df['close'].ewm(span=12, adjust=False).mean()
        ema26 = renko_df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        renko_df['MACD_hist'] = macd - signal

        # Target sign mean ou autre (adapte si besoin)
        renko_df['target_sign_mean'] = np.sign(renko_df['close'].shift(-1) - renko_df['close']).rolling(
            3).mean().fillna(0)

        # ==================================================================
        # 2. Création du TimeSeriesDataSet
        # ==================================================================
        max_encoder_length = config.get('seq_len', 20)
        max_prediction_length = 1  # ex. prédiction directionnelle

        training_cutoff = renko_df["time_idx"].max() - int(0.2 * len(renko_df))  # 20% val

        training = TimeSeriesDataSet(
            renko_df[renko_df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target=config['target_col'],
            group_ids=["symbol"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["symbol"],
            time_varying_known_reals=["time_idx", "time_live"],
            time_varying_unknown_reals=config['features_base'],  # EMA, RSI, etc. (sans volume)
            target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, renko_df, predict=True, stop_randomization=True)

        # Workers activés comme tu veux (4 pour accélérer création dataset)
        train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=4, persistent_workers=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=4, persistent_workers=True)

        # ==================================================================
        # 3. Modèle TFT + Trainer
        # ==================================================================
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=config.get('lstm_units', 50),
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,  # quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        tft.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # ==================================================================
        # 4. Checkpoint + Fit
        # ==================================================================
        run_id = f"tft_{int(datetime.now().timestamp() * 1000)}_{os.getpid()}"
        checkpoint_dir = f"models/simple_opt/checkpoints_temp/{run_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-tft",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        )

        trainer = Trainer(
            max_epochs=config.get('max_epochs', 30),
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=True,
        )

        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # ==================================================================
        # 5. Score + Cleanup
        # ==================================================================
        best_path = checkpoint_callback.best_model_path
        if not best_path or not os.path.exists(best_path):
            return -999.0, None

        score = -checkpoint_callback.best_model_score.item()  # ou ton score custom (backtest sur val)

        del tft, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return score, best_path

    except Exception as e:
        print(f"[run_backtest] CRASH → {e}")
        return -999.0, None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


import os
import shutil
import torch
import gc
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
import numpy as np


def to_renko(df, brick_size):
    """Convertit ticks (time, close) en Renko OHLC dataframe basique (sans volume, comme tes features)."""
    df = df[['time', 'close']].copy().dropna().reset_index(drop=True)

    renko = []
    start_price = df['close'][0]
    current_brick_close = start_price
    brick_time = df['time'][0]

    for i in range(1, len(df)):
        price = df['close'][i]
        time = df['time'][i]

        diff = price - current_brick_close
        if abs(diff) < brick_size:
            continue  # pas assez de mouvement

        num_bricks = int(abs(diff) / brick_size)
        direction = np.sign(diff)

        for _ in range(num_bricks):
            brick_open = current_brick_close
            brick_close = current_brick_close + direction * brick_size
            brick_high = max(brick_open, brick_close)
            brick_low = min(brick_open, brick_close)

            renko.append({
                'time': brick_time,
                'open': brick_open,
                'high': brick_high,
                'low': brick_low,
                'close': brick_close,
            })

            current_brick_close = brick_close
            brick_time = time

    # Ajout du dernier brick si besoin
    if renko:
        renko_df = pd.DataFrame(renko)
        renko_df['time_idx'] = range(len(renko_df))  # pour TFT
        renko_df['time_max'] = renko_df['time'].max()  # pour cutoff
        renko_df['symbol'] = 'ETHUSD'  # groupe fixe
        renko_df['time_live'] = (renko_df['time'] - renko_df['time'].min()).dt.total_seconds() / 3600  # ex. en heures

        return renko_df
    else:
        raise ValueError("Pas assez de mouvement pour créer des Renko.")


def run_backtest(config):
    """
    Version corrigée : ticks → Renko → features → TFT.
    Retourne: (score, chemin_du_meilleur_checkpoint.ckpt ou None)
    """
    df = config['data']['data'] if isinstance(config['data'], dict) else config['data']
    df = df.copy()  # sécurité

    try:
        # ==================================================================
        # 1. Création des Renko à partir des ticks
        # ==================================================================
        renko_df = to_renko(df, config['renko_size'])

        # Ajout des features (EMA, RSI, MACD_hist) sur Renko
        # (utilise pandas_ta ou ta-lib si installé ; ici pandas simple pour EMA ex.)
        renko_df['EMA'] = renko_df['close'].ewm(span=config['ema_period'], adjust=False).mean()
        renko_df['delta'] = renko_df['close'].diff()
        renko_df['up'] = np.where(renko_df['delta'] > 0, renko_df['delta'], 0)
        renko_df['down'] = np.where(renko_df['delta'] < 0, -renko_df['delta'], 0)
        renko_df['avg_up'] = renko_df['up'].ewm(alpha=1 / config['rsi_period']).mean()
        renko_df['avg_down'] = renko_df['down'].ewm(alpha=1 / config['rsi_period']).mean()
        renko_df['RSI'] = 100 - 100 / (1 + renko_df['avg_up'] / renko_df['avg_down'])

        # MACD_hist (ex. simple)
        ema12 = renko_df['close'].ewm(span=12, adjust=False).mean()
        ema26 = renko_df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        renko_df['MACD_hist'] = macd - signal

        # Target sign mean ou autre (adapte si besoin)
        renko_df['target_sign_mean'] = np.sign(renko_df['close'].shift(-1) - renko_df['close']).rolling(
            3).mean().fillna(0)

        # ==================================================================
        # 2. Création du TimeSeriesDataSet
        # ==================================================================
        max_encoder_length = config.get('seq_len', 20)
        max_prediction_length = 1  # ex. prédiction directionnelle

        training_cutoff = renko_df["time_idx"].max() - int(0.2 * len(renko_df))  # 20% val

        training = TimeSeriesDataSet(
            renko_df[renko_df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target=config['target_col'],
            group_ids=["symbol"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["symbol"],
            time_varying_known_reals=["time_idx", "time_live"],
            time_varying_unknown_reals=config['features_base'],  # EMA, RSI, etc. (sans volume)
            target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(training, renko_df, predict=True, stop_randomization=True)

        # Workers activés comme tu veux (4 pour accélérer création dataset)
        train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=4, persistent_workers=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=4, persistent_workers=True)

        # ==================================================================
        # 3. Modèle TFT + Trainer
        # ==================================================================
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=config.get('lstm_units', 50),
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,  # quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        tft.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # ==================================================================
        # 4. Checkpoint + Fit
        # ==================================================================
        run_id = f"tft_{int(datetime.now().timestamp() * 1000)}_{os.getpid()}"
        checkpoint_dir = f"models/simple_opt/checkpoints_temp/{run_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-tft",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        )

        trainer = Trainer(
            max_epochs=config.get('max_epochs', 30),
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=True,
        )

        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # ==================================================================
        # 5. Score + Cleanup
        # ==================================================================
        best_path = checkpoint_callback.best_model_path
        if not best_path or not os.path.exists(best_path):
            return -999.0, None

        score = -checkpoint_callback.best_model_score.item()  # ou ton score custom (backtest sur val)

        del tft, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return score, best_path

    except Exception as e:
        print(f"[run_backtest] CRASH → {e}")
        return -999.0, None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()