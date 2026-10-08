#!/bin/bash

# 1. On définit le chemin du projet
PROJECT_ROOT="/media/pierre/datad/dvlp/MyProjects/TRADING/MT5/PmxRkoGui"
cd $PROJECT_ROOT

# 2. On configure le PYTHONPATH pour que les imports fonctionnent
export PYTHONPATH=$PROJECT_ROOT

# 3. On désactive les optimisations qui causent des crashs (rappel de nos épisodes précédents)
export TF_ENABLE_ONEDNN_OPTS=0

# 4. On lance le script
# Si vous voulez en lancer plusieurs, ajoutez & à la fin
/usr/bin/python3.11 optimize/optimize_optuna.py
