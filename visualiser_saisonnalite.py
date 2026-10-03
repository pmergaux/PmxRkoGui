import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement et préparation
df = pd.read_csv("stats_pentes_volatilites.csv", sep=";")
# On ne garde que les jours individuels pour la superposition
df_jours = df[df['type'] == 'Jour'].copy()
df_jours['date'] = pd.to_datetime(df_jours['date'])

# Création d'une colonne "Numéro de Semaine" pour grouper les courbes
# On part de la date la plus récente
dates_uniques = sorted(df_jours['date'].unique(), reverse=True)
df_jours['semaine_id'] = df_jours['date'].apply(lambda x: (dates_uniques[0] - x).days // 7)

# Ordre des jours pour l'axe X
ordre_jours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 2. Configuration des graphiques
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
plt.subplots_adjust(hspace=0.3)

metrics = [
    ('coef_pente', 'Pente ($/tick)', 'Directionnalité du prix'),
    ('std_dollars', 'Écart-type Std ($)', 'Amplitude absolue'),
    ('vol_log_pct', 'Volatilité Log (%)', 'Nervosité relative / Bruit')
]

for i, (col, yield_name, title) in enumerate(metrics):
    ax = axes[i]
    # On trace une ligne par semaine
    sns.lineplot(
        data=df_jours,
        x='label',
        y=col,
        hue='semaine_id',
        palette='viridis',
        ax=ax,
        sort=False,
        marker='o',
        alpha=0.7
    )

    ax.set_title(f"Superposition 10 semaines : {title}", fontsize=14, fontweight='bold')
    ax.set_ylabel(yield_name)
    ax.set_xlabel("")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Semaine (0=Récente)', bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajuster l'ordre des jours sur l'axe X
plt.xticks(range(7), ordre_jours)

plt.tight_layout()
plt.show()