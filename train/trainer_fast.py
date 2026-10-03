# refresh_model.py
import os
import json
import pandas as pd
from train.trainer import train_model


def fast_refresh():
    # 1. Charger la configuration actuelle
    config_path = "../config_live.json"  # <--- Vérifiez le nom de votre fichier
    with open(config_path) as json_file:
        cfg = json.load(json_file)

    # 2. INJECTER VOS MEILLEURS PARAMÈTRES (Issus de votre Optuna de 20h)
    # Remplacez les valeurs ci-dessous par les résultats de votre optimisation
    cfg['lstm']['lstm_units'] = 128  # Votre valeur Optuna
    cfg['lstm']['lstm_seq_len'] = 24  # Votre valeur Optuna
    cfg['lstm']['learning_rate'] = 0.001  # Votre valeur Optuna

    # 3. RÉGLER LA FENÊTRE SUR 30 JOURS
    # On force le chargement des données récentes uniquement
    cfg['train_data']['days_history'] = 30
    cfg['train_data']['test_size'] = 0.15  # 15% pour la validation

    # 4. RÉGLER LE NOMBRE D'ÉPOCHS
    # Pas besoin de 1000 epochs, on veut juste une adaptation
    cfg['lstm']['epochs'] = 50
    cfg['lstm']['patience'] = 10

    print(f"--- Début du rafraîchissement (30 jours) ---")
    print(f"Utilisation des paramètres optimisés : Units={cfg['lstm']['lstm_units']}")

    # 5. LANCER L'ENTRAÎNEMENT
    # Cette fonction va charger les données, créer le nouveau scaler et entraîner le modèle
    model, scaler = train_model(cfg)

    # 6. SAUVEGARDER AVEC UN NOUVEAU NOM
    model_name = "models/pmx_rnn_refreshed_30d.keras"
    scaler_name = "models/scaler_refreshed_30d.pkl"

    model.save(model_name)
    joblib.dump(scaler, scaler_name)

    print(f"--- Rafraîchissement terminé ---")
    print(f"Modèle sauvegardé : {model_name}")
    print(f"Désormais, modifiez votre config live pour pointer vers ce modèle.")


if __name__ == "__main__":
    fast_refresh()
"""
model_path: models/pmx_rnn_refreshed_30d.keras
scaler_path: models/scaler_refreshed_30d.pkl"""