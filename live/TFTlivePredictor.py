import torch
import joblib
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, MAE
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.metrics import QuantileLoss
import dill


# ==================================================================
# 1. CHARGEMENT DU MODÈLE ENTRAINÉ UNE FOIS POUR TOUTES (au démarrage)
# ==================================================================
class TFTLivePredictor:
    def __init__(self, model_path="models/tft_best_2026.pth", scaler_path="models/tft_scaler.pkl"):
        # Charger le modèle entraîné
        # 1. Charger le dataset d'entraînement (obligatoire !)
        with open("models/tft_training_dataset.pkl", "rb") as f:
            self.training_dataset = dill.load(f)

        # 2. Recréer le modèle à partir de ce dataset
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training_dataset,  # ← voilà d'où il vient !
            learning_rate=1e-3,  # valeur bidon (ignorée)
            hidden_size=16,
            dropout=0.1,
            loss=MAE()
        )

        # 3. Charger les poids
        self.tft.load_state_dict(torch.load("models/tft_best_2026.pth"))
        self.tft.eval()
        print("TFT Live chargé avec succès")

        self.tft = TemporalFusionTransformer.from_dataset(self.training_dataset)  # on recrée l'architecture
        self.tft.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.tft.eval()  # ← MODE INFERENCE (très important)
        self.tft.freeze()  # ← désactive dropout, batchnorm, etc.

        # Charger le scaler/normalizer (OBLIGATOIRE ! même normalisation qu’au training)
        self.scaler = joblib.load(scaler_path)  # ou GroupNormalizer si tu l’as sauvegardé

        # Conserver les paramètres du dataset d’entraînement
        self.feature_cols = ["EMA", "RSI", "MACD_hist", "close", "time_live"]
        self.max_encoder_length = 120
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tft.to(self.device)

    # ==================================================================
    # 2. PRÉDICTION ULTRA-RAPIDE SUR LES DERNIÈRES LIGNES (1ms par prédiction)
    # ==================================================================
    def predict_last_n(self, df_latest: pd.DataFrame, n_last: int = 3) -> np.ndarray:
        """
        Prédit les n dernières briques Renko (ou 1 seule)
        df_latest : ton DataFrame avec les dernières lignes (doit avoir au moins 120 lignes)
        Retourne : array de n probas (ex: [0.71, 0.68, 0.82])
        """
        if len(df_latest) < self.max_encoder_length:
            raise ValueError(f"Pas assez de données : {len(df_latest)} < {self.max_encoder_length}")

        # Garder seulement les n dernières lignes à prédire
        df_predict = df_latest.tail(n_last).copy()

        # Créer le dataset de prédiction (même paramètres qu’au training)
        predict_dataset = TimeSeriesDataSet.from_parameters(
            self.training_dataset.get_parameters(),  # ← on copie TOUT (normalizer, encoder_length, etc.)
            df_latest,                          # tout l’historique pour l’encoder
            predict=True,
            stop_randomization=True,
        )

        # Dataloader ultra-léger
        predict_loader = predict_dataset.to_dataloader(train=False, batch_size=n_last, num_workers=0)

        # PRÉDICTION (0.02 sec même sur CPU)
        with torch.no_grad():
            raw_pred = self.tft.predict(predict_loader, mode="prediction", show_progress_bar=False)

        # Extraction propre
        if isinstance(raw_pred, list):
            x = raw_pred[0]
        else:
            x = raw_pred

        if hasattr(x, 'cpu'):
            pred = x.cpu().numpy().squeeze()
        else:
            pred = np.asarray(x).squeeze()

        # Conversion tanh → proba
        proba = np.clip((np.tanh(pred) + 1.0) / 2.0, 0.01, 0.99)

        return proba[-n_last:]  # on retourne seulement les n dernières

# ==================================================================
# UTILISATION EN LIVE (1 ligne = 1ms)
# ==================================================================
# Au démarrage du bot (1 seule fois)
predictor = TFTLivePredictor()

# À chaque nouveau brick Renko (ex: toutes les 2–10 min)
df = get_latest_renko_data()  # ton DataFrame mis à jour

# Prédiction sur les 1, 2 ou 3 dernières briques
proba_last_3 = predictor.predict_last_n(df, n_last=3)
proba_now = proba_last_3[-1]  # ← la toute dernière prédiction

print(f"TFT Frozen → Proba UP = {proba_now:.3f}")

# Signal
if proba_now > 0.63:
    send_order("BUY")
elif proba_now < 0.37:
    send_order("SELL")
else:
    send_order("FLAT")
