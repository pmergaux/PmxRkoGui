# backtest/strategies/lstm_renko_backtrader.py
import joblib
import pandas as pd
import tensorflow as tf
from backtest.BackTrader import BackTrader
from strategy.pmxRko import PmxRkoStrategy
from decision.candle_decision import calculate_indicators, choix_features
from strategy.pmxRko import PmxRkoStrategy
from utils.lstm_utils import create_sequences
from utils.renko_utils import tick21renko
from utils.utils import BUY, SELL, CLOSE


class LSTMRenkoBackStrategy(BackTrader, PmxRkoStrategy):
    def __init__(self, param, data):
        BackTrader.__init__(self, param, data)
        PmxRkoStrategy.__init__(self, param)

        self.model = tf.keras.models.load_model(param.get('model_path', 'models/transformer_best.keras'))
        self.scaler = joblib.load(param.get('scaler_path', 'models/scaler.pkl'))
        self.features = param['features']
        self.seq_len = param['lstm']['seq_len']

        self.bricks = pd.DataFrame()
        self.last_processed_time = None  # ← temps de la dernière brique close traitée

    def next(self):
        # --- 1. CONSTRUIRE BRIQUES ---
        self.bricks = tick21renko(self.ti, self.bricks, self.p['renko_size'], 'bid')

        if self.bricks is None or len(self.bricks) < 2:
            return

        # --- 2. NOUVELLE BRIQUE CLOSE ? ---
        last_closed_time = self.bricks.index[-2]  # ← dernière brique close
        if self.last_processed_time is not None and self.last_processed_time >= last_closed_time:
            return
        self.last_processed_time = last_closed_time

        # --- 3. ASSEZ DE BRIQUES CLOSES ? ---
        if len(self.bricks) < self.seq_len + 21:
            return

        # --- 4. CALCUL SUR TOUTES LES BRIQUES ---
        df_ind = calculate_indicators(self.bricks, self.p)
        df_feat = choix_features(df_ind, self.p)

        # --- 5. EXCLURE LA DERNIÈRE LIGNE (en devenir) ---
        df_feat_closed = df_feat.iloc[:-1]

        # --- 6. SÉQUENCES SUR BRIQUES CLOSES UNIQUEMENT ---
        X_scaled = self.scaler.transform(df_feat_closed[self.features].tail(self.seq_len + 20))
        X_seq, _ = create_sequences(X_scaled, self.seq_len)

        if len(X_seq) == 0:
            return

        # --- 7. PRÉDICTION SUR LA DERNIÈRE SÉQUENCE CLOSE ---
        proba = self.model.predict(X_seq[-1:], verbose=0)[0][0]

        # --- 8. SIGNAL ---
        signal = 1 if proba > 0.6 else -1 if proba < 0.4 else 0

        # --- 9. GESTION POSITION ---
        if signal == 1 and self.position is None:
            self.open_back(BUY)
        elif signal == -1 and self.position is None:
            self.open_back(SELL)
        elif signal == 0 and self.position is not None:
            self.close_back(CLOSE, 'lstm_exit')
