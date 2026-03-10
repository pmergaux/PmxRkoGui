# analyze_study.py — comprendre où chercher
import optuna
storage_url = "postgresql+pg8000://pierre:axa8Garp@localhost/optuna_db"
study = optuna.load_study(study_name="trading_diff_postg", storage=storage_url)

df = study.trials_dataframe()
top = df[df['value'] > df['value'].quantile(0.9)]  # top 10%
print(top[[c for c in top.columns if c.startswith('params_')]].describe())
