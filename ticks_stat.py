import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import os

from utils.utils import calculer_stats


def extraire_analyse_saisonniere_complete(input_csv, output_csv):
    print(f"🚀 Analyse complète de {input_csv}...")

    # Optimisation : lecture de la fin du fichier (800 Mo)
    taille_totale = os.path.getsize(input_csv)
    offset = max(0, taille_totale - 25 * 1024 * 1024)

    df = pd.read_csv(input_csv, sep=";", usecols=['time_msc', 'bid'],
                     skiprows=range(1, 1000) if offset > 0 else None)

    df['date'] = pd.to_datetime(df['time_msc'], unit='ms')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Calage Temps Serveur
    dernier_tick = df.index.max()
    jours_a_retirer = (dernier_tick.weekday() + 1) % 7
    dernier_dimanche = (dernier_tick - timedelta(days=jours_a_retirer)).replace(hour=23, minute=59, second=59)

    resultats = []

    # Boucle sur 10 semaines
    for s in range(10):
        fin_sem = dernier_dimanche - timedelta(weeks=s)
        debut_sem = (fin_sem - timedelta(days=6)).replace(hour=0, minute=0, second=0)
        df_semaine = df.loc[debut_sem:fin_sem]

        if df_semaine.empty: continue

        print(f"⏳ Semaine du {debut_sem.date()} au {fin_sem.date()}")

        # Jours individuels
        jours_presents = df_semaine.index.normalize().unique()
        for jour_ts in jours_presents:
            df_jour = df_semaine[df_semaine.index.normalize() == jour_ts]
            pente, std_dlr, vol_log, r2, er = calculer_stats(df_jour)

            resultats.append({
                'label': jour_ts.strftime('%A'),
                'date': jour_ts.date(),
                'coef_pente': pente,
                'std_dollars': std_dlr,
                'vol_log_pct': vol_log,
                'R2': r2,
                'type': 'Jour'
            })

        # Groupes
        for label, mask in [('Lundi-Vendredi', df_semaine.index.weekday <= 4),
                            ('Samedi-Dimanche', df_semaine.index.weekday >= 5)]:
            df_grp = df_semaine[mask]
            p, s_d, v_l, r2, er = calculer_stats(df_grp)
            resultats.append({
                'label': label,
                'date': f"{debut_sem.date()}",
                'coef_pente': p,
                'std_dollars': s_d,
                'vol_log_pct': v_l,
                'R2':r2,
                'type': 'Groupe'
            })

    # Sauvegarde
    final_df = pd.DataFrame(resultats)
    final_df = final_df[['label', 'date', 'coef_pente', 'std_dollars', 'vol_log_pct', 'R2', 'type']]
    final_df.to_csv(output_csv, index=False, sep=";")
    print(f"✅ Terminé ! Fichier : {output_csv}")


if __name__ == "__main__":
    extraire_analyse_saisonniere_complete("/home/pierre/data/ETHUSD_origin.csv", "stats_pentes_volatilites.csv")
