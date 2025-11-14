# core/trader.py
from live.mt5_bridge import MT5Bridge
from lstm.lstm_model import LiveLSTM
from risk.kelly import kelly_fraction, dynamic_lot_size
import json
import zmq
from datetime import datetime

class IndustrialTrader:
    def __init__(self, config, signal_callback):
        self.config = config
        self.signal_callback = signal_callback
        self.lstm = LiveLSTM("models/lstm_best.pth")
        self.bridge = MT5Bridge(self.on_tick)
        self.equity = 10000
        self.position = None

    def start(self):
        self.bridge.start()

    def stop(self):
        self.bridge.stop()

    def on_tick(self, tick):
        # === FEATURES ===
        features = [
            tick["close"], tick["ema"], tick["rsi"], tick["macd_line"],
            tick["macd_signal"], tick["cci"], tick["renko_up"], tick["renko_down"],
            tick["volume"], tick["spread"]
        ]

        # === LSTM PREDICTION ===
        pred = self.lstm.add_tick(features)
        if not pred:
            return

        # === RÈGLES MANUELLES ===
        rsi_ok = 30 < tick["rsi"] < 70 if self.config.get("rule_rsi") else True
        ema_cross = tick["close"] > tick["ema"] if self.config.get("rule_ema") else True
        macd_cross = tick["macd_line"] > tick["macd_signal"] if self.config.get("rule_macd") else True

        # === DÉCISION FINALE ===
        signal = "HOLD"
        proba = pred["hold"]

        if pred["buy"] > self.config["threshold_buy"] and rsi_ok and ema_cross and macd_cross:
            signal, proba = "BUY", pred["buy"]
        elif pred["sell"] > self.config["threshold_sell"]:
            signal, proba = "SELL", pred["sell"]

        # === KELLY + LOT SIZE ===
        kelly_f = kelly_fraction(proba, odds=1.5)
        lot = dynamic_lot_size(self.equity, risk_percent=2, kelly_f=kelly_f)

        # === ENVOI SIGNAL À UI + MT5 ===
        msg = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": tick["symbol"],
            "signal": signal,
            "proba": proba,
            "lot": round(lot, 2),
            "tp": tick["close"] + 50 * tick["point"] if signal == "BUY" else tick["close"] - 50 * tick["point"],
            "sl": tick["close"] - 30 * tick["point"] if signal == "BUY" else tick["close"] + 30 * tick["point"],
            "source": "LSTM+Rules"
        }
        self.signal_callback(msg)

        # === ENVOI À MT5 VIA ZEROMQ (PUB) ===
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://127.0.0.1:5557")
        socket.send_string(json.dumps(msg))
