# decision/neural_decision.py
from utils.lstm_utils import (
    load_model, load_scaler, prepare_sequence, predict_sequence, get_feature_columns
)
from utils.utils import BUY, SELL, NONE  # ← CONSTANTES CENTRALES
import pandas as pd
from typing import List, Optional

class NeuralDecision:
    """
    Retourne :
        BUY  (1)  → Achat
        SELL (-1) → Vente
        NONE (0)  → Neutre (HOLD)
        None      → Pas de signal (erreur, données insuffisantes)
    """
    def __init__(self, model_path: str = "model/lstm_model.h5", scaler_path: str = "model/scaler.pkl"):
        self.model = load_model(model_path)
        self.scaler = load_scaler(scaler_path)

    def predict(self, df: pd.DataFrame, seq_len: int = 30, feature_cols: List[str] = None) -> Optional[int]:
        """
        Retourne un signal via constantes centralisées.
        """
        if len(df) < seq_len:
            return None  # Données insuffisantes

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        try:
            seq = prepare_sequence(df, feature_cols, seq_len)
            pred = predict_sequence(self.model, self.scaler, seq)
        except Exception as e:
            print(f"Erreur prédiction LSTM : {e}")
            return None  # Erreur → pas de signal

        if pred > 0.6:
            return BUY      # 1
        elif pred < 0.4:
            return SELL     # -1
        else:
            return NONE     # 0