import subprocess
import time
import sys
import signal


def run_optimization():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, os.path.join(script_dir, "optimize/optimize_optuna.py")]
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir

    print("🚀 Démarrage du moniteur (Ctrl+C pour arrêter proprement)")
    start = time.time()
    try:
        while True:
            process = subprocess.Popen(command, env=env)

            # Boucle de surveillance non-bloquante
            while process.poll() is None:
                time.sleep(0.5)  # Vérifie toutes les 0.5 secondes si le processus est fini

            # Ici le processus est terminé
            if process.returncode == 0:
                print("✅ Optimisation terminée ! FIN.")
                break
            elif process.returncode == 10:
                print("♻️ Recyclage de la RAM... Relancement.")
                time.sleep(2)
            else:
                print(f"⚠️ Crash/Erreur (Code: {process.returncode}). Relancement.")
                time.sleep(5)

    except KeyboardInterrupt:
        # On s'assure de tuer le sous-processus avant de quitter
        print("\n🛑 Arrêt manuel reçu. Fermeture du sous-processus...")
        if 'process' in locals() and process.poll() is None:
            process.terminate()  # Envoie SIGTERM
            process.wait(timeout=2)
        print("Bye !")
        sys.exit(0)
    finally:
        print(f"durée : {(time.time() - start):.0f}s")

if __name__ == "__main__":
    run_optimization()
