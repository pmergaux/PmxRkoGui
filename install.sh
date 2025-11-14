#!/bin/bash
# GrokLstmRenkoTrader — Installation automatique (Linux)
echo "Pierre, tu as 83 ans. Tu as gagné. Installation en cours..."

# Venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Dépendances
pip install PyQt6 numpy pandas tensorflow numba ray[tune] optuna scikit-optimize matplotlib pyqtgraph MetaTrader5 pyzmq

# Raccourci
echo '#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python src/main.py
' > run.sh
chmod +x run.sh

echo "Installation terminée ! Lance avec : ./run.sh"
