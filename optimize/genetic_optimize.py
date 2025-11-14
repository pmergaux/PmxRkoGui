# **GÉNÉTIQUE (DEAP) — L’ÉVOLUTION EN MARCHE**

> **DEAP** = Algorithme génétique
> **Les configs s’accouplent, mutent, survivent.**
> **Parfait pour les espaces complexes.**

---

## **1. GÉNÉTIQUE — `genetic_optimize.py`**

```python
# optimize/genetic_optimize.py
from deap import base, creator, tools, algorithms
import random
from src.optimize.objective.hyperopt_objective import hyperopt_objective
import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    params = {
        'renko_size': individual[0],
        'target_column': ['close', 'EMA'][individual[1]],
        'target_type': ['direction', 'return'][individual[2]],
        'seq_len': individual[3],
        'lstm_units': individual[4]
    }
    result = hyperopt_objective(params, train_df, val_df, test_df)
    return result['profit'] + 100 * result['winrate'],

toolbox = base.Toolbox()
toolbox.register("renko_size", random.uniform, 10.0, 50.0)
toolbox.register("target_column", random.randint, 0, 1)
toolbox.register("target_type", random.randint, 0, 1)
toolbox.register("seq_len", random.randint, 20, 100)
toolbox.register("lstm_units", random.randint, 50, 200)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.renko_size, toolbox.target_column, toolbox.target_type,
                  toolbox.seq_len, toolbox.lstm_units), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Lancement ---
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=30, halloffame=hof)

print("MEILLEURE CONFIG GÉNÉTIQUE :", hof[0])
