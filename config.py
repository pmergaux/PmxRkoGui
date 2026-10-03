import os
from pathlib import Path

# Définition des dossiers de base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path.home() / "data"  # Pointe vers ~/data

# Chemins spécifiques pour PmxRkoGui
STRATEGY_PATH = BASE_DIR / "strategy"
OPTIMIZE_PATH = BASE_DIR / "optimize"
BACKTEST_PATH = BASE_DIR / "backtest"

# Configuration PostgreSQL
DB_USER = "pierre"
DB_PASSWORD = "axa8Garp"
DB_HOST = "localhost" # ou l'IP de votre serveur
DB_NAME = "optuna_db"
# storage_url = "postgresql+pg8000://pierre:axa8Garp@localhost/optuna_db"

# URL de connexion pour Optuna
SQLALCHEMY_DATABASE_URL = f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
# pour la suivre : optuna-dashboard postgresql://pierre:axa8Garp@localhost:5432/optuna_db
# pour lancer les optimisations python3 /media/pierre/datad/dvlp/MyProjects/TRADING/MT5/PmxRkoGui/run_optimizatio.py
