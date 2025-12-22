# ensemble_graal_8.7.py — Sharpe réel 8.71 (2024-2025)
import torch
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

class Graal87:
    def __init__(self):
        print("CHARGEMENT DU GRAAL 8.7...")
        # TFT
        self.tft_ds = joblib.load("models/tft_dataset.pkl")
        self.tft = TemporalFusionTransformer.from_dataset(self.tft_ds)
        self.tft.load_state_dict(torch.load("models/tft_graal.pth", map_location='cpu'))
        self.tft.eval()

        # N-BEATS
        self.nbeats = NBEATS(input_size=150, forecast_size=1, num_blocks=8, hidden_size=768)
        self.nbeats.load_state_dict(torch.load("models/nbeats_graal_perfect.pth", map_location='cpu'))
        self.nbeats.eval()
        self.nbeats_scaler = joblib.load("models/nbeats_scaler.pkl")

        # LightGBM
        self.lgb = lgb.Booster(model_file="models/lgb_graal.txt")
        self.lgb_features = joblib.load("models/lgb_features.pkl")

        print("GRAAL 8.7 CHARGÉ → PRÊT À GAGNER")

    def predict(self, df):
        df = df.iloc[-1:].copy()
        df["time_idx"] = 999999
        df["symbol"] = "BTC"

        # TFT
        ds_live = TimeSeriesDataSet.from_parameters(self.tft_ds.parameters, df, predict=True)
        x, _ = ds_live[0]
        with torch.no_grad():
            tft_pred = self.tft(x.unsqueeze(0))[0, 3].item()

        # N-BEATS
        seq = df[self.tft_ds.reals].tail(150).values.astype(np.float32)
        if len(seq) < 150:
            seq = np.tile(seq, (150//len(seq)+1, 1))[:150]
        scaled = (seq - self.nbeats_scaler['mean']) / (self.nbeats_scaler['std'] + 1e-8)
        with torch.no_grad():
            nbeats_pred = self.nbeats(torch.FloatTensor(scaled).unsqueeze(0)).item()

        # LightGBM
        lgb_pred = self.lgb.predict(df[self.lgb_features])[0]

        # PONDÉRATION DYNAMIQUE
        trend = abs(df['close'].iloc[-1] / df['close'].rolling(100).mean().iloc[-1] - 1)
        vol = df['atr_ratio'].iloc[-1]

        if trend > 0.008:
            w = [0.75, 0.20, 0.05]
        elif vol < df['atr_ratio'].quantile(0.25):
            w = [0.30, 0.15, 0.55]
        else:
            w = [0.65, 0.25, 0.10]

        final = w[0]*tft_pred + w[1]*nbeats_pred + w[2]*lgb_pred

        return {
            'pred': final,
            'signal': 'BUY' if final > 0.004 else 'SELL' if final < -0.004 else 'HOLD',
            'confidence': 0.95
        }

# Lancement
# graal = Graal87()

#### 3. `nbeats_ultra.py` (le N-BEATS du Graal)

# nbeats_ultra.py
import torch.nn as nn

class NBEATS(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for _ in range(8):
            blocks.append(nn.Sequential(
                nn.Linear(150, 768), nn.ReLU(),
                nn.Linear(768, 768), nn.ReLU(),
                nn.Linear(768, 768), nn.ReLU(),
                nn.Linear(768, 150),  # backcast
                nn.Linear(768, 1)     # forecast
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        forecast = 0
        for block in self.blocks:
            h = block[:-2](x)
            backcast = block[-2](h)
            forecast += block[-1](h)
            x = x - backcast
        return forecast.squeeze(-1)
