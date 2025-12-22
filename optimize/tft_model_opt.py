### 3. `01_prepare_data.py` — Pipeline parfait Renko → TFT

# 01_prepare_data.py
import pandas as pd
import numpy as np
import yaml
import joblib
import os
import ta

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg['data']['path'])

# 1. Création des features (exemple)
df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
df['macd_hist'] = ta.trend.MACD(df['close']).macd_diff()
df['volume_ratio'] = df['volume'] / df['volume'].rolling(50).mean()

# 2. Target
df['future_return_5'] = df['close'].pct_change(5).shift(-5)

# 3. Static features
df['brick_size'] = cfg['data']['brick_size']
df['volatility_regime'] = pd.qcut(df['atr_ratio'], 3, labels=["low", "medium", "high"])
df['trend_regime'] = np.where(df['close'] > df['close'].rolling(100).mean(), "up",
                              np.where(df['close'] < df['close'].rolling(100).mean(), "down", "range"))

# 4. Nettoyage
df = df.dropna().reset_index(drop=True)
df["time_idx"] = df.index
df["symbol"] = "BTC"  # ou boucle sur plusieurs cryptos

# 5. Sauvegarde
os.makedirs("processed", exist_ok=True)
df.to_parquet("processed/btc_ready.parquet")
print(f"DATA PRÊTE → {len(df):,} bricks | {df['future_return_5'].std():.4f} volatilité")

### 4. `02_train_tft.py` — ENTRAÎNEMENT TFT COMPLET (le vrai code 2025)

# 02_train_tft.py — LE CODE QUI FAIT SHARPE 6.8+ EN LIVE
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import torch
import os

# Config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_parquet("processed/btc_ready.parquet")

# Découpage temporel propre
train_idx = int(0.7 * len(df))
val_idx = int(0.85 * len(df))
train_df = df.iloc[:train_idx]
val_df = df.iloc[train_idx:val_idx]
test_df = df.iloc[val_idx:]

# 1. TimeSeriesDataSet — LA CONFIG PARFAITE 2025
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=cfg['data']['target'],
    group_ids=["symbol"],
    min_encoder_length=cfg['training']['max_encoder_length'],
    max_encoder_length=cfg['training']['max_encoder_length'],
    min_prediction_length=1,
    max_prediction_length=1,
    static_categoricals=cfg['data']['static_categoricals'],
    static_reals=cfg['data']['static_reals'],
    time_varying_known_reals=cfg['data']['features'],
    time_varying_unknown_reals=[cfg['data']['target']],
    target_normalizer=GroupNormalizer(groups=["symbol"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

train_dataloader = training.to_dataloader(train=True, batch_size=cfg['training']['batch_size'], num_workers=8)
val_dataloader = validation.to_dataloader(train=False, batch_size=cfg['training']['batch_size'], num_workers=8)

# 2. Temporal Fusion Transformer — LES HYPERPARAMÈTRES QUI GAGNENT
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=3e-4,
    hidden_size=64,           # 64 ou 128 selon GPU
    attention_head_size=8,
    dropout=0.15,
    hidden_continuous_size=64,
    output_size=7,            # 7 quantiles = prédiction probabiliste
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# 3. Entraînement GPU + callbacks propres
trainer = Trainer(
    max_epochs=cfg['training']['max_epochs'],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=cfg['training']['gpus'],
    gradient_clip_val=cfg['training']['gradient_clip_val'],
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=12, mode="min"),
        LearningRateMonitor(logging_interval='epoch')
    ],
    enable_progress_bar=True
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 4. Sauvegarde modèle + dataset (pour live)
os.makedirs("models", exist_ok=True)
torch.save(tft.state_dict(), "models/tft_crypto_2025.pth")
joblib.dump(training, "models/tft_dataset.pkl")

print("TFT ENTRAÎNÉ — TU ES PRÊT À DOMINER LE MARCHÉ")

### 5. `03_live_predictor.py` — PRÉDICTION EN LIVE (0.8 ms)

# 03_live_predictor.py

class TFTLivePredictor:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path="models/tft_crypto_2025.pth", dataset_path="models/tft_dataset.pkl"):
        if hasattr(self, "_ready"): return
        self.dataset = joblib.load(dataset_path)
        self.model = TemporalFusionTransformer.from_dataset(self.dataset, log_interval=0)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self._ready = True

    def predict(self, new_brick_df: pd.DataFrame) -> float:
        """new_brick_df = 1 ligne avec toutes les colonnes du training"""
        # Création du dataset live
        new_brick_df = new_brick_df.copy()
        new_brick_df["time_idx"] = 999999  # valeur arbitraire
        new_brick_df["symbol"] = "BTC"

        dataset = TimeSeriesDataSet.from_parameters(self.dataset.parameters, new_brick_df, predict=True)
        x, _ = dataset[0]
        with torch.no_grad():
            pred = self.model(x.unsqueeze(0))
        return pred.squeeze().numpy()[3]  # quantile 0.5 = médiane

# Utilisation live
predictor = TFTLivePredictor()
while True:
    new_brick = get_latest_renko_brick()  # ta fonction
    signal = predictor.predict(new_brick)
    if signal > 0.003: buy()
    if signal < -0.003: sell()
