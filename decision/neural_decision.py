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
    def __init__(self, param: dict, model_path: str, scaler_path: str):
        self.param = param
        self.model = load_model(model_path)
        self.scaler = load_scaler(scaler_path)

    def predict(self, df: pd.DataFrame) -> Optional[int]:
        """
        Retourne un signal via constantes centralisées.
        """
        lstm = self.param.get('lstm')
        features = self.param.get('features')
        target = self.param.get('target')
        if lstm is None or features is None or target is None:
            raise "revoir paramètrage"
        seq_len = lstm.get('seq_len')
        target_col = target.get('column', None)
        if len(df) < seq_len or target_col is None:
            raise "Données insuffisantes ou target missing"
        if target.get('include', False) and target_col not in features:
            features.append(target_col)
        try:
            seq = prepare_sequence(df, features, seq_len)
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