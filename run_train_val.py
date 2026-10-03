import subprocess
import time
import sys


def run_optimization():
    import os
    # Détection automatique du répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, os.path.join(script_dir, "backtest/pmxrko_train_val.py")]

    # Configure l'environnement d'exécution pour que les imports fonctionnent
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir

    while True:
        print("\n🚀 Lancement d'une session d'optimisation (Lot RAM limité)...")
        try:
            # On lance le processus avec notre variable d'environnement PYTHONPATH
            process = subprocess.Popen(command, env=env)
            process.wait()  # On attend qu'il finisse ou recycle la RAM

            if process.returncode == 0:
                print("✅ Optimisation globale terminée de manière normale ! FIN.")
                break
            elif process.returncode == 10:
                print(
                    "♻️ Recyclage de la RAM effectué avec succès. Relancement de la session suivante dans 2 secondes...")
                time.sleep(2)
            else:
                print(
                    f"⚠️ Le processus s'est arrêté (Code ou Crash : {process.returncode}). Relancement sécurisé dans 5s...")
                time.sleep(5)

        except KeyboardInterrupt:
            print("🛑 Arrêt manuel demandé par l'utilisateur.")
            break


if __name__ == "__main__":
    run_optimization()
