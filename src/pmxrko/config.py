import os
from pathlib import Path

# Définition des dossiers de base
_BASE_DIR = Path(__file__).resolve().parent
# Initialisation automatique au chargement
if "PMXRKO_BASE_DIR" in os.environ:
    _BASE_DIR = Path(os.environ["PMXRKO_BASE_DIR"]).resolve()
DATA_DIR = "data"  # Pointe vers ~/data
# Chemins spécifiques pour PmxRkoGui
STRATEGY_PATH = "strategy"
OPTIMIZE_PATH = "optimize"
BACKTEST_PATH = "backtest"

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
# Dans src/pmxrko/config.py du package

def set_base_dir(path):
    global _BASE_DIR
    _BASE_DIR = Path(path)

def get_base_dir():
    # Retourne toujours la valeur à jour, même si elle a été modifiée après coup
    return _BASE_DIR
def get_data_dir():
    return _BASE_DIR / DATA_DIR
def get_strategy_path():
    return _BASE_DIR / STRATEGY_PATH
def get_optimize_path():
    return _BASE_DIR / OPTIMIZE_PATH
def get_backtest_path():
    return _BASE_DIR / BACKTEST_PATH
