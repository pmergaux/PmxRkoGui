import gc
import os
import json
import optuna
import torch


# ==================================================================
# TA FONCTION D'ÉVALUATION (doit être en haut du fichier ou importée)
# ==================================================================
def evaluate_config(config):
    """Ta fonction actuelle qui renvoie un score (float)"""
    # ... tout ton code actuel qui était dans evaluate_config ...
    # elle doit être autonome (pas d'accès à des variables globales non-picklables)
    # → retourne juste le score (float)
    try:
        # Exemple fictif (remplace par ton vrai code)
        from backtest.backtest_module import run_backtest  # ← importe ici si besoin
        score = run_backtest(config)
        return float(score)
    except Exception as e:
        print(f"ERREUR config {config.get('renko_size')} / {config.get('target_col')} → {e}")
        return -999.0  # pénalité forte si crash


# ==================================================================
# OBJECTIF OPTUNA
# ==================================================================
def objective(trial):
    # --- Hyperparamètres à optimiser ---
    renko_size = trial.suggest_float("renko_size", 15.0, 15.9, step=0.05)  # plus fin que 0.1
    ema_period = trial.suggest_int("ema_period", 7, 14)
    rsi_period = trial.suggest_int("rsi_period", 10, 20)
    target_col = trial.suggest_categorical("target_col", ["close", "EMA", "target_sign_mean"])
    threshold_buy = trial.suggest_float("threshold_buy", 0.50, 0.80, step=0.025)
    threshold_sell = trial.suggest_float("threshold_sell", 0.15, 0.50, step=0.025)

    # Optionnel : tu peux aussi tuner seq_len, units, etc.
    seq_len = trial.suggest_categorical("seq_len", [15, 20, 25, 30])
    lstm_units = trial.suggest_categorical("lstm_units", [32, 50, 64, 100])

    config = {
        'renko_size': round(renko_size, 3),
        'ema_period': int(ema_period),
        'rsi_period': int(rsi_period),
        'target_col': target_col,
        'target_type': 'direction',
        'seq_len': int(seq_len),
        'lstm_units': int(lstm_units),
        'threshold_buy': round(threshold_buy, 3),
        'threshold_sell': round(threshold_sell, 3),
        'features_base': ["EMA", "RSI", "MACD_hist", "close", "time_live", "TFT"],
        'VERSION': ['TFT'],
        'hcode': f"optuna_trial_{trial.number}"
    }
    try:
        # TON ÉVALUATION ULTRA-LIGHT (pas de TFT complet à chaque trial !)
        score = evaluate_config(config)  # ← ta fonction rapide (LSTM 50u ou TFT frozen)
        # OU si tu veux vraiment du TFT → utilise le modèle FROZEN
        # score = tft_frozen_predict_and_score(config)

        # Nettoyage mémoire forcé
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score
    except Exception as e:
        return -999.0  # pénalité

# ==================================================================
# LANCEMENT OPTUNA (la magie en 3 lignes)
# ==================================================================
def run_optuna_search(n_trials=500, timeout_minutes=30):
    sampler = optuna.samplers.TPESampler(seed=42)  # reproductible
    pruner = optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=20)

    # LANCEMENT OPTUNA ANTI-CRASH
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
        storage="sqlite:///optuna_light.db",
        load_if_exists=True
    )

    print(f"OPTUNA → {n_trials} trials | timeout {timeout_minutes} min | {os.cpu_count()} cœurs")
    study.optimize(
        objective,
        n_trials=300,
        timeout=3600,  # 1 heure max
        n_jobs=1,  # ← 1 SEUL PROCESSUS À LA FOIS (évite OOM)
        gc_after_trial=True,  # ← nettoie la RAM après chaque trial
        show_progress_bar=True
    )

    print("\n" + "=" * 70)
    print("MEILLEURE CONFIG TROUVÉE PAR OPTUNA")
    print("=" * 70)
    best = study.best_trial
    print(f"Score: {best.value:.4f} | Trial #{best.number}")
    print(json.dumps(best.params, indent=2))

    # Sauvegarde
    os.makedirs("models/simple_opt", exist_ok=True)
    final_config = {
        **best.params,
        'target_type': 'direction',
        'features_base': ["EMA", "RSI", "MACD_hist", "close", "time_live", "TFT"],
        'VERSION': ['TFT'],
        'hcode': 'optuna_best_2026',
        'optuna_trial': best.number,
        'optuna_score': best.value
    }

    with open("models/simple_opt/best_pierre2026_optuna.json", "w") as f:
        json.dump(final_config, f, indent=2)

    # Top 10
    print("\nTOP 10 CONFIGS:")
    for i, trial in enumerate(study.best_trials[:10], 1):
        print(
            f"{i}. Score {trial.value:.4f} → renko={trial.params['renko_size']:.2f} | target={trial.params['target_col']} | tb={trial.params['threshold_buy']:.2f}")

    return study, final_config

# ==================================================================
# VISUALISATION OPTUNA – LES PLUS BELLES COURBES DU MONDE
# ==================================================================
def plot_optuna_visualizations(study):
    import optuna.visualization as vis

    print("\nGénération des visualisations Optuna... (ça vaut le coup d'attendre 10 sec)")

    # 1. Évolution du meilleur score
    fig1 = vis.plot_optimization_history(study)
    fig1.update_layout(title="Évolution du Best Score (plus haut = mieux)", height=600)
    fig1.show()

    # 2. Importance des hyperparamètres (QUELLE VARIABLE COMPTE LE PLUS ?!)
    fig2 = vis.plot_param_importances(study)
    fig2.update_layout(title="Importance des Hyperparamètres (Hyperopt Impact)")
    fig2.show()

    # 3. Slice plot – comment chaque paramètre influence le score
    fig3 = vis.plot_slice(study, params=["renko_size", "threshold_buy", "threshold_sell", "ema_period", "target_col"])
    fig3.update_layout(title="Impact de chaque paramètre sur le score", height=800)
    fig3.show()

    # 4. Parallel coordinate – les meilleures configs en 3D interactif
    fig4 = vis.plot_parallel_coordinate(study, params=["renko_size", "threshold_buy", "target_col", "ema_period"])
    fig4.update_layout(title="Top configs – Parallel Coordinate Plot")
    fig4.show()

    # 5. Contour plot – interaction entre renko_size et threshold_buy (magique)
    fig5 = vis.plot_contour(study, params=["renko_size", "threshold_buy"])
    fig5.update_layout(title="Interaction renko_size × threshold_buy")
    fig5.show()

    # 6. EDF – Distribution des scores (combien de configs > 100, > 150, etc.)
    fig6 = vis.plot_edf(study)
    fig6.update_layout(title="Distribution Empirique des Scores (EDF)")
    fig6.show()

    # 7. Top 10 trials en tableau
    print("\nTOP 10 MEILLEURES CONFIGS:")
    df = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    df = df.sort_values("value", ascending=False).head(10)
    df["value"] = df["value"].round(4)
    print(
        df[["number", "value", "params_renko_size", "params_threshold_buy", "params_target_col", "duration"]].to_string(
            index=False))

    # 8. Sauvegarde HTML des meilleures visualisations
    os.makedirs("models/simple_opt/viz", exist_ok=True)
    fig1.write_html("models/simple_opt/viz/history.html")
    fig2.write_html("models/simple_opt/viz/importance.html")
    fig4.write_html("models/simple_opt/viz/parallel.html")
    fig5.write_html("models/simple_opt/viz/contour_renko_tb.html")
    print("Toutes les visualisations sauvegardées dans models/simple_opt/viz/")

# ==================================================================
# LANCE ÇA sans visu !
# ==================================================================
"""
if __name__ == "__main__":
    study, best_config = run_optuna_search(
        n_trials=600,  # 600 trials ≈ 10–18 min sur 16 cœurs
        timeout_minutes=40  # sécurité
    )
"""
# ==================================================================
# ou LANCEMENT FINAL + VISU
# ==================================================================
if __name__ == "__main__":
    study, best_config = run_optuna_search(
        n_trials=600,
        timeout_minutes=40
    )

    # LA MAGIE FINALE
    plot_optuna_visualizations(study)

    print("\nOPTUNA + VISUALISATION")
    print("Ouvre les fichiers .html dans ton navigateur → tu vas halluciner")
"""
dans models/simple_opt/viz/
### Bonus ultime : Dashboard live PENDANT l’optimisation

Ouvre un **deuxième terminal** et lance :

```bash
optuna-dashboard sqlite:///optuna_tft.db --port 8080
```

→ http://localhost:8080  
→ Tu vois **en temps réel** toutes les courbes qui se dessinent pendant que ça tourne !

### Résultat final

Tu lances → tu vas boire un café → tu reviens → tu as :

- Le meilleur modèle jamais trouvé
- Des graphiques de ouf qui expliquent **pourquoi**
- Des fichiers HTML à envoyer à ton boss / tes potes ("regarde comme je suis fort")

"""
