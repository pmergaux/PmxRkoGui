# core/model_manager.py
import os
import json
import joblib
from utils.config_utils import config_to_hash
import tensorflow as tf


class ModelManager:
    def __init__(self, models_dir="models"):
        self.dir = models_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, scaler, config: dict, backtest_results: dict = None):
        hash_code = config_to_hash(config)

        paths = {
            "model": os.path.join(self.dir, f"model_{hash_code}.h5"),
            "scaler": os.path.join(self.dir, f"scaler_{hash_code}.pkl"),
            "config": os.path.join(self.dir, f"config_{hash_code}.json"),
            "results": os.path.join(self.dir, f"results_{hash_code}.json")
        }

        model.save(paths["model"])
        joblib.dump(scaler, paths["scaler"])

        with open(paths["config"], 'w') as f:
            json.dump(config, f, indent=4, default=str)

        if backtest_results:
            with open(paths["results"], 'w') as f:
                json.dump(backtest_results, f, indent=4)

        print(f"MODÈLE SAUVEGARDÉ → {hash_code}")
        return hash_code, paths

    def load_best(self):
        results = [f for f in os.listdir(self.dir) if f.startswith("results_")]
        if not results:
            return None, None, None, None

        best_file = max(results, key=lambda x: json.load(open(os.path.join(self.dir, x)))['sharpe'])
        hash_code = best_file[8:-5]  # results_XXXX.json → XXXX

        model = tf.keras.models.load_model(f"{self.dir}/model_{hash_code}.h5")
        scaler = joblib.load(f"{self.dir}/scaler_{hash_code}.pkl")
        config = json.load(open(f"{self.dir}/config_{hash_code}.json"))
        results = json.load(open(f"{self.dir}/results_{hash_code}.json"))

        print(f"MEILLEUR MODÈLE CHARGÉ → {hash_code} | Sharpe: {results['sharpe']:.3f}")
        return model, scaler, config, results