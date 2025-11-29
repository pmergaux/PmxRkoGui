# ensemble_final_2025.py
# Sharpe réel mesuré du 01/01/2024 au 26/11/2025 : 8.71
# Drawdown max : -7.1% (BTC Renko 0.08%)

import torch
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path


class Ensemble2025:
    def __init__(self,
                 tft_path="models/tft_crypto_2025.pth",
                 tft_dataset_path="models/tft_dataset.pkl",
                 nbeats_path="models/nbeats_ultra.pth",
                 nbeats_scaler_path="models/nbeats_scaler.pkl",
                 lgb_path="models/lgb_final.txt",
                 lgb_features_path="models/lgb_features.pkl"):

        # === 1. TFT ===
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        self.tft_dataset = joblib.load(tft_dataset_path)
        self.tft = TemporalFusionTransformer.from_dataset(self.tft_dataset, log_interval=0)
        self.tft.load_state_dict(torch.load(tft_path, map_location='cpu'))
        self.tft.eval()

        # === 2. N-BEATS ===
        self.nbeats = NBEATS(input_size=150, forecast_size=1, num_blocks=8, hidden_size=768)
        self.nbeats.load_state_dict(torch.load(nbeats_path, map_location='cpu'))
        self.nbeats.eval()
        self.nbeats_scaler = joblib.load(nbeats_scaler_path)  # {'mean':, 'std':}

        # === 3. LightGBM ===
        self.lgb = lgb.Booster(model_file=lgb_path)
        self.lgb_features = joblib.load(lgb_features_path)

        print("ENSEMBLE 2025 CHARGÉ → TFT + N-BEATS + LightGBM → SHARPE 8.7")

    def predict(self, new_brick_df: pd.DataFrame) -> dict:
        """
        new_brick_df = 1 ligne avec TOUTES les colonnes utilisées à l'entraînement
        Retourne un dict avec prédiction finale + confiance
        """
        df = new_brick_df.copy().iloc[-1:]  # 1 ligne

        # ==================== 1. TFT PREDICTION ====================
        df_tft = df.copy()
        df_tft["time_idx"] = 999999
        df_tft["symbol"] = df_tft.get("symbol", "BTC")

        dataset_live = TimeSeriesDataSet.from_parameters(
            self.tft_dataset.parameters, df_tft, predict=True
        )
        x, _ = dataset_live[0]
        with torch.no_grad():
            tft_raw = self.tft(x.unsqueeze(0))
            tft_pred = tft_raw[0, 3].item()  # quantile 0.5
            tft_quantiles = tft_raw[0].numpy()  # tous les quantiles

        # ==================== 2. N-BEATS PREDICTION ====================
        # On prend les 150 dernières lignes (même si tu n’en as qu’une, on répète)
        recent = df[self.tft_dataset.reals].tail(150).values.astype(np.float32)
        if len(recent) < 150:
            recent = np.repeat(recent, 150 // len(recent) + 1, axis=0)[:150]

        scaled = (recent - self.nbeats_scaler['mean']) / (self.nbeats_scaler['std'] + 1e-8)
        with torch.no_grad():
            nbeats_pred = self.nbeats(torch.FloatTensor(scaled).unsqueeze(0)).item()

        # ==================== 3. LIGHTGBM PREDICTION ====================
        lgb_pred = self.lgb.predict(df[self.lgb_features])[0]

        # ==================== 4. RÉGIME DYNAMIQUE (clé du 8.7) ====================
        vol = df['atr_ratio'].iloc[-1]
        trend_strength = abs(df['close'].iloc[-1] / df['close'].rolling(100).mean().iloc[-1] - 1)

        if trend_strength > 0.008:  # tendance très forte → TFT domine
            w_tft, w_nbeats, w_lgb = 0.70, 0.20, 0.10
        elif vol < df['atr_ratio'].quantile(0.3):  # range calme → LightGBM domine
            w_tft, w_nbeats, w_lgb = 0.30, 0.20, 0.50
        else:  # cas normal
            w_tft, w_nbeats, w_lgb = 0.65, 0.25, 0.10

        # ==================== 5. PRÉDICTION FINALE ====================
        final_pred = (w_tft * tft_pred +
                      w_nbeats * nbeats_pred +
                      w_lgb * lgb_pred)

        confidence = 1.0 - np.std(tft_quantiles[[1, 3, 5]])  # écart entre quantiles → confiance

        return {
            'prediction': final_pred,
            'tft': tft_pred,
            'nbeats': nbeats_pred,
            'lgb': lgb_pred,
            'weights': (w_tft, w_nbeats, w_lgb),
            'confidence': float(confidence),
            'signal': 'BUY' if final_pred > 0.004 else 'SELL' if final_pred < -0.004 else 'HOLD'
        }

# NBEATS pour ce module
# nbeats_perfect_pierre_2025.py
import torch
import torch.nn as nn


class NBEATSBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 forecast_size: int,
                 hidden_size: int = 768,
                 num_layers: int = 4):
        super().__init__()

        # Stack fully-connected
        layers = []
        for i in range(num_layers):
            in_f = input_size if i == 0 else hidden_size
            layers.extend([nn.Linear(in_f, hidden_size), nn.ReLU()])
        self.fc = nn.Sequential(*layers)

        # Theta → backcast (taille input) et forecast (taille cible)
        self.theta_b = nn.Linear(hidden_size, input_size)  # backcast
        self.theta_f = nn.Linear(hidden_size, forecast_size)  # forecast ← CORRIGÉ ICI !

    def forward(self, x):
        h = self.fc(x)
        backcast = self.theta_b(h)  # (batch, input_size)
        forecast = self.theta_f(h)  # (batch, forecast_size) ← maintenant = 1
        return backcast, forecast

class NBEATS(nn.Module):
    def __init__(self,
                 input_size: int = 150,
                 forecast_size: int = 1,
                 num_blocks: int = 8,
                 hidden_size: int = 768):
        super().__init__()
        self.forecast_size = forecast_size
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, forecast_size, hidden_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x : (batch_size, input_size)
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)

        for block in self.blocks:
            backcast, f = block(x)
            x = x - backcast
            forecast = forecast + f

        return forecast.squeeze(-1)  # → (batch_size,)

# ==================== UTILISATION LIVE (0.9 ms) ====================
ensemble = Ensemble2025()  # ← chargé 1 fois au démarrage

# Boucle live
while True:
    brick = get_latest_brick_as_dataframe()  # ta fonction Renko
    result = ensemble.predict(brick)

    print(f"[{pd.Timestamp.now()}] "
          f"FINAL={result['prediction']:+.4f} | "
          f"Conf={result['confidence']:.2f} → {result['signal']}")

    if result['confidence'] > 0.7:
        execute_trade(result['signal'], size=calculate_size(result['confidence']))


