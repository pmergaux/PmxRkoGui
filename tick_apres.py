import os
import pandas as pd
from datetime import datetime, timezone
from mt5linux import MetaTrader5
import time

# --- CONFIGURATION ---
SYMBOL = "ETHUSD"
FILENAME = f"/media/pierre/datad/data/{SYMBOL}_origin.csv"
CHUNK_DAYS = 2
PATH_META = '/home/pierre/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'


def update_ticks_file(symbol, filename):
    mt5 = MetaTrader5()
    if not mt5.initialize(path=PATH_META):
        print(f"❌ Erreur MT5: {mt5.last_error()}")
        return

    # 1. Déterminer le point de départ
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"📖 Lecture du dernier tick de : {filename}")
        # On lit la fin du fichier sans charger les 4.6 Go en RAM
        with open(filename, 'rb') as f:
            try:
                f.seek(-1024, os.SEEK_END) # On recule de 1ko
            except OSError:
                f.seek(0) # Si le fichier est plus petit que 1ko
            
            lines = f.readlines()
            if len(lines) > 1:
                last_line_str = lines[-1].decode('utf-8')
                # Format supposé : time;bid;ask;last;volume;time_msc;flags
                # On utilise split pour extraire time_msc (index 5)
                parts = last_line_str.strip().split(';')
                
                # On essaie de trouver l'index de time_msc dynamiquement si possible 
                # ou on part du principe que c'est l'avant dernier ou dernier index numérique
                try:
                    # Dans le format standard MT5 : time, bid, ask, last, volume, time_msc, flags
                    # Donc time_msc est à l'index 5
                    last_msc = int(parts[5])
                    start_dt = datetime.fromtimestamp(last_msc / 1000.0, tz=timezone.utc)
                    print(f"⏳ Dernier tick trouvé le : {start_dt} (msc: {last_msc})")
                except (IndexError, ValueError):
                    print("⚠️ Impossible de parser la dernière ligne, démarrage par défaut.")
                    start_dt = datetime(2025, 5, 1, tzinfo=timezone.utc)
                    last_msc = 0
            else:
                start_dt = datetime(2025, 5, 1, tzinfo=timezone.utc)
                last_msc = 0
    else:
        print("🆕 Fichier inexistant. Création d'une nouvelle base.")
        start_dt = datetime(2025, 5, 1, tzinfo=timezone.utc)
        last_msc = 0

    # 2. Récupération incrémentale
    current_start = start_dt
    final_now = datetime.now(timezone.utc)
    total_added = 0

    while current_start < final_now:
        segment_end = current_start + pd.Timedelta(days=CHUNK_DAYS)
        if segment_end > final_now: segment_end = final_now

        print(f"📡 Requête : {current_start.strftime('%Y-%m-%d %H:%M')} -> {segment_end.strftime('%Y-%m-%d %H:%M')}")

        ticks = mt5.copy_ticks_range(symbol, current_start, segment_end, mt5.COPY_TICKS_ALL)

        if ticks is not None and len(ticks) > 0:
            df_new = pd.DataFrame(ticks)

            # FILTRE DE SÉCURITÉ : On ne garde que ce qui est strictement après le dernier tick
            df_new = df_new[df_new['time_msc'] > last_msc]

            if not df_new.empty:
                # Écriture immédiate (append) pour ne pas saturer la RAM
                file_exists = os.path.isfile(filename)
                df_new.to_csv(filename, sep=";", index=False, mode='a', header=not file_exists)

                last_msc = int(df_new['time_msc'].iloc[-1])
                total_added += len(df_new)
                print(f"  ✅ +{len(df_new)} ticks ajoutés (Total session: {total_added})")

        current_start = segment_end
        time.sleep(0.1)  # Respect du pont RPC

    print(f"🏁 Mise à jour terminée. {total_added} ticks ajoutés au total.")
    mt5.shutdown()


if __name__ == "__main__":
    update_ticks_file(SYMBOL, FILENAME)
