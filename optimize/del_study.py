import optuna

storage_url = "postgresql+pg8000://pierre:axa8Garp@localhost/optuna_db"
study_name = "trading_diff_postg"

try:
    optuna.delete_study(study_name=study_name, storage=storage_url)
    print(f"✅ L'étude '{study_name}' a été supprimée de Postgres.")
except KeyError:
    print(f"⚠️ L'étude '{study_name}' n'existe pas dans la base.")

# Optionnel : Supprimer aussi le fichier modèle sur le disque
"""
import os
if os.path.exists("best_model_FINAL.h5"):
    os.remove("best_model_FINAL.h5")
    print("✅ Ancien modèle disque supprimé.")
"""