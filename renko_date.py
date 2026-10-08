import numpy as np
import pandas as pd
import pickle

# Chargez votre fichier généré
TICKS_NPY_PATH = f"/media/pierre/datad/data/renko_cache/renko_22.1.pkl" # Adaptez le nom
df = pd.read_pickle(TICKS_NPY_PATH)
print(df.head(), '\n', len(df))
