import numpy as np
import pandas as pd
import pickle

# Chargez votre fichier généré
TICKS_NPY_PATH = f"/media/pierre/datad/data/ETHUSD_120.csv" # Adaptez le nom
df_ticks = pd.read_csv(TICKS_NPY_PATH, sep=";")
df_ticks['time_msc'] = pd.to_datetime(df_ticks['time_msc'], unit="ms")
df_ticks = df_ticks.set_index('time_msc', drop=False).sort_index()
print(df_ticks.head())
# Si le chiffre commence par 17... (ex: 1742589345000), c'est des millisecondes.
# Si le chiffre commence par 17... et est beaucoup plus long, c'est des nanosecondes.

