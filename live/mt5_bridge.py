# live/mt5_bridge.py
import zmq
import json
import threading
from datetime import datetime

class MT5Bridge:
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://127.0.0.1:5556")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("MT5 Bridge → Connecté à ZeroMQ 5556")
        while self.running:
            try:
                msg = socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg)
                data["recv_time"] = datetime.now().isoformat()
                self.callback(data)
            except zmq.Again:
                pass
            except Exception as e:
                print(f"MT5 Bridge error: {e}")
