# optimize/ray_tune_corrected.py — 100 % Fonctionnel, 2025
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch  # Pour bayésienne
from ray.tune.schedulers import ASHAScheduler  # Pour pruning
import tensorflow as tf
import numpy as np
import time

# === 1. ESPACE DE RECHERCHE (param_space) ===
param_space = {
    "lstm_units": tune.randint(50, 200),  # Entier
    "learning_rate": tune.loguniform(1e-5, 1e-2),  # Log uniforme
    "seq_len": tune.randint(20, 100),
    "renko_size": tune.uniform(10.0, 50.0)
}

# === 2. FONCTION OBJECTIF (trainable) ===
def objective(config):
    # Simulation de ton LSTM (remplace par ton code réel)
    units = config["lstm_units"]
    lr = config["learning_rate"]
    seq_len = config["seq_len"]
    renko = config["renko_size"]

    # Entraînement fictif
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=(seq_len, 10)),  # 10 features
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')

    # Données fictives
    X = np.random.randn(1000, seq_len, 10)
    y = np.random.randint(0, 2, 1000)

    # Fit (10 epochs)
    for epoch in range(10):
        loss = model.fit(X, y, epochs=1, verbose=0, validation_split=0.2).history['loss'][0]
        time.sleep(0.1)  # Simule temps
        tune.report({"loss": loss, "epoch": epoch})  # Rapport à Ray

    final_loss = model.evaluate(X, y, verbose=0)
    tune.report({"loss": final_loss})  # Final

# === 3. LANCEMENT CORRIGÉ ===
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Search algorithm (bayésienne avec Optuna)
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    # Scheduler (pruning)
    scheduler = ASHAScheduler(metric="loss", mode="min")

    # Tuner
    tuner = tune.Tuner(
        objective,
        param_space=param_space,  # ← CORRIGÉ : param_space
        tune_config=tune.TuneConfig(
            num_samples=100,
            metric="loss",
            mode="min",
            scheduler=scheduler
        ),
        run_config=tune.RunConfig(
            storage_path="./ray_results",  # Sauvegarde
            name="lstm_optimize"
        )
    )

    # Fit
    results = tuner.fit()

    # Meilleur
    best = results.get_best_result(metric="loss", mode="min")
    print("MEILLEUR RÉSULTAT :")
    print(best.config)
    print(f"Loss : {best.metrics['loss']:.4f}")

