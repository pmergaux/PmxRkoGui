# lstm/lstm_model.py
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class LSTMTrader(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # BUY, SELL, HOLD

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LiveLSTM:
    def __init__(self, model_path=None):
        self.model = LSTMTrader()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.buffer = deque(maxlen=60)

    def add_tick(self, features):
        self.buffer.append(features)
        if len(self.buffer) < 60:
            return None
        x = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).numpy()[0]
        return {"buy": float(probs[0]), "sell": float(probs[1]), "hold": float(probs[2])}
