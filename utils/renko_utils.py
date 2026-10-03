import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# Constantes
RTIME, ROPENR, ROPEN, RHIGH, RLOW, RCLOSE, RCLOSER = range(7)
colonnesRko = ['time', 'open_renko', 'open', 'high', 'low', 'close', 'close_renko']

# Fonctions utilitaires
def _update_brick_list(brick: list, price: float) -> None:
    """Met à jour une brique dans une liste."""
    brick[RHIGH] = max(brick[RHIGH], price)
    brick[RLOW] = min(brick[RLOW], price)
    brick[RCLOSE] = price

def _update_brick(df_bricks: pd.DataFrame, price: float) -> None:
    """Met à jour la dernière brique dans le DataFrame."""
    df_bricks.iat[-1, RHIGH] = max(df_bricks.iat[-1, RHIGH], price)
    df_bricks.iat[-1, RLOW] = min(df_bricks.iat[-1, RLOW], price)
    df_bricks.iat[-1, RCLOSE] = price

"""
# Exemple de logique de construction Renko avec volume
def build_renko_with_volume(ticks, brick_size):
    renko_data = []
    current_volume = 0
    # ... votre logique de prix ...
    
    for tick in ticks:
        current_volume += tick.volume # On accumule le volume
        
        # Si le prix dépasse la brique
        if abs(tick.last - last_brick_price) >= brick_size:
            renko_data.append({
                'open': ...,
                'close': ...,
                'volume': current_volume # Le volume total consommé pour créer cette brique
            })
            current_volume = 0 # On réinitialise pour la brique suivante
"""

def _correct_closer(df_bricks: pd.DataFrame, step: float) -> None:
    """Corrige la colonne 'closer' pour toutes les briques."""
    df_bricks['close_renko'] = np.where(df_bricks['open'] < df_bricks['close'], df_bricks['open_renko'] + step,
        np.where(df_bricks['open'] > df_bricks['close'], df_bricks['open_renko'] - step,
                 np.where(df_bricks['open_renko'] > df_bricks['close'], df_bricks['open_renko'] - step, df_bricks['open_renko'] + step)))

def put_index_renko(bricks):
    bricks = bricks.set_index('time', drop=False)
    bricks.index = pd.to_datetime(bricks.index, unit="ms")
    return bricks

def tick21renko(df: pd.DataFrame, bricks: pd.DataFrame, step: float, value: str = 'bid') -> pd.DataFrame:
    def tick2renko(df: pd.DataFrame, step: float = 10.0, value: str = 'bid'):
        if df.empty or value not in df.columns:
            print(f"Erreur: DataFrame vide ou colonne '{value}' manquante.")
            return pd.DataFrame(columns=colonnesRko)
        price = df.iat[0, df.columns.get_loc(value)]
        time = df.index[0]
        pprice = math.floor(price / step) * step
        bricks = pd.DataFrame([[time, pprice, price, price, price, price, 0.0]],
                              columns=colonnesRko)
        bricks = bricks.set_index('time', drop=False)
        bricks.index = pd.to_datetime(bricks.index, unit='ms')
        return bricks
    start = 0
    if bricks is None or bricks.empty:
        bricks = tick2renko(df, step, value)
        start = 1
    prices = df[value].values[start:]
    times = df.index.values[start:]  # Timestamps en millisecondes
    pprice = bricks.iat[-1, ROPENR]
    # Initialiser la liste avec la dernière brique non clôturée
    #print('input bricks\n', bricks.tail(2))
    current_brick = [
        bricks.index[-1],
        bricks.iat[-1, ROPENR],
        bricks.iat[-1, ROPEN],
        bricks.iat[-1, RHIGH],
        bricks.iat[-1, RLOW],
        bricks.iat[-1, RCLOSE],
        bricks.iat[-1, RCLOSER]
    ]
    #old_direction = 1 if current_brick[ROPEN] > current_brick[RCLOSE] else -1
    all_new_bricks = [current_brick]  # Liste pour accumuler toutes les briques
    local_bricks = bricks[:-1]
    for price, time in zip(prices, times):
        price_diff = price - pprice
        direction = 1 if price_diff > 0 else -1
        abs_diff = abs(price_diff)
        if abs_diff <= step:      #or (old_direction != direction and abs_diff < 2 * step):
            _update_brick_list(all_new_bricks[-1], price)
            continue
        mult = math.floor(abs_diff / step)
        cprice = pprice + step * direction
        if mult == 1:
            #all_new_bricks[-1][RCLOSER] = cprice
            _update_brick_list(all_new_bricks[-1], price)
        nprice = pprice + direction * mult * step
        mult -= 1
        if mult > 1:
            timeopen = all_new_bricks[-1][RTIME]
            stt = (time - timeopen) / mult
            ptt = (price - cprice) / mult
            iprice = pprice + step * direction
            for t in range(1, mult):
                #all_new_bricks[-1][RCLOSER] = iprice
                all_new_bricks.append(
                    [pd.Timestamp(timeopen + stt * t),
                     iprice, cprice, max(cprice, cprice + ptt),
                     min(cprice, cprice + ptt),
                     cprice + ptt,
                     0.0])
                iprice += step * direction
                cprice += ptt
        all_new_bricks.append(
            [time, nprice, price,
             price, price, price, 0])
        #old_direction = direction
        pprice = nprice
    # Concaténer les briques restantes, en évitant le warning
    new_df = pd.DataFrame(all_new_bricks, columns=colonnesRko)
    # Vérifier que bricks.iloc[:-1] et new_df ne sont pas vides
    if not bricks.empty:
        bricks = pd.concat([local_bricks, new_df], axis=0, ignore_index=True)
    else:
        bricks = new_df
    _correct_closer(bricks, step)
    bricks = bricks.set_index('time', drop=False)
    bricks.index = pd.to_datetime(bricks.index, unit="ms")
    return bricks

def tick2renko(df: pd.DataFrame, bricks: pd.DataFrame, step: float, value: str = 'bid', mode=False) -> pd.DataFrame:
    def tick0renko(df: pd.DataFrame, step: float = 10.0, value: str = 'bid'):
        if df.empty or value not in df.columns:
            print(f"Erreur: DataFrame vide ou colonne '{value}' manquante.")
            return pd.DataFrame(columns=colonnesRko)
        price = df.iat[0, df.columns.get_loc(value)]
        time = df.index[0]
        pprice = math.floor(price / step) * step
        bricks = pd.DataFrame([[time, pprice, price, price, price, price, 0.0]],
                              columns=colonnesRko)
        bricks = bricks.set_index('time', drop=False)
        bricks.index = pd.to_datetime(bricks.index, unit='ms')
        return bricks
    start = 0
    if bricks is None or bricks.empty:
        bricks = tick0renko(df, step, value)
        start = 1
    prices = df[value].values[start:]
    times = df.index.values[start:]  # Timestamps en millisecondes
    pprice = bricks.iat[-1, ROPENR]
    # Initialiser la liste avec la dernière brique non clôturée
    #print('input bricks\n', bricks.tail(2))
    current_brick = [
        bricks.index[-1],
        bricks.iat[-1, ROPENR],
        bricks.iat[-1, ROPEN],
        bricks.iat[-1, RHIGH],
        bricks.iat[-1, RLOW],
        bricks.iat[-1, RCLOSE],
        bricks.iat[-1, RCLOSER]
    ]
    old_direction = 1 if current_brick[ROPEN] > current_brick[RCLOSE] else -1
    all_new_bricks = [current_brick]  # Liste pour accumuler toutes les briques
    for i, (price, time) in enumerate(zip(prices, times)):
        price_diff = price - pprice
        direction = 1 if price_diff > 0 else -1
        abs_diff = abs(price_diff)
        if abs_diff < step or (old_direction != direction and abs_diff < 2 * step):
            _update_brick_list(all_new_bricks[-1], price)
            continue
        mult = math.floor(abs_diff / step)
        if old_direction == direction:
            cprice = price if mult == 1 else pprice + step * direction
        else:
            cprice = price if mult == 2 else pprice + 2 * step * direction
            pprice += step * direction
            all_new_bricks[-1][ROPENR] = pprice
            mult -= 1
        _update_brick_list(all_new_bricks[-1], cprice)
        nprice = pprice + direction * mult * step
        if mult > 1:
            timeopen = all_new_bricks[-1][RTIME]
            stt = (time - timeopen) / mult
            ptt = (price - cprice) / mult
            iprice = pprice + step * direction
            for t in range(1, mult):
                #all_new_bricks[-1][RCLOSER] = iprice
                all_new_bricks.append(
                    [pd.Timestamp(timeopen + stt * t),
                     iprice, cprice, max(cprice, cprice + ptt),
                     min(cprice, cprice + ptt),
                     cprice + ptt,
                     0.0])
                iprice += step * direction
                cprice += ptt
        #all_new_bricks[-1][RCLOSER] = nprice
        all_new_bricks.append(
            [time, nprice, price,
             price, price, price, 0])
        old_direction = direction
        pprice = nprice
    # Mettre à jour la dernière brique de bricks avec la première de all_new_bricks
    if all_new_bricks:
        #print('colonnes :', bricks.columns, ' bricks 0\n',bricks.tail(),'\nlist', all_new_bricks[0])
        bricks.iat[-1, ROPENR] = all_new_bricks[0][ROPENR]
        bricks.iat[-1, ROPEN] = all_new_bricks[0][ROPEN]
        bricks.iat[-1, RHIGH] = all_new_bricks[0][RHIGH]
        bricks.iat[-1, RLOW] = all_new_bricks[0][RLOW]
        bricks.iat[-1, RCLOSE] = all_new_bricks[0][RCLOSE]
        bricks.iat[-1, RCLOSER] = all_new_bricks[0][RCLOSER]
        #print('bricks 0\n',bricks.tail())
        # Concaténer les briques restantes, en évitant le warning
        if len(all_new_bricks) > 1:
            new_df = pd.DataFrame(all_new_bricks[1:], columns=colonnesRko)
            # Vérifier que bricks.iloc[:-1] et new_df ne sont pas vides
            if not bricks.iloc[:-1].empty and not new_df.empty:
                bricks = pd.concat([bricks.iloc[:-1][bricks.iloc[:-1].notna().any(axis=1)],
                                    new_df[new_df.notna().any(axis=1)]],
                                   ignore_index=True)
            elif not new_df.empty:
                bricks = pd.concat([bricks.iloc[:0], new_df[new_df.notna().any(axis=1)]], ignore_index=True)
            # Si new_df est vide, bricks reste inchangé
        # Si all_new_bricks contient seulement la brique courante, bricks est déjà mis à jour
    else:
        # Cas improbable : aucune brique n'a été générée
        pass

    _correct_closer(bricks, step)
    bricks = bricks.set_index('time', drop=False)
    bricks.index = pd.to_datetime(bricks.index, unit="ms")
    """
    timestamps = [brick[RTIME] for brick in bricks]
    if len(timestamps) != len(set(timestamps)):
        print(f"Attention: {len(timestamps) - len(set(timestamps))} timestamps dupliqués détectés dans les briques.")
    """
    return bricks


def renko_start_date(ticks, step, min, decal, interval=1, func=None):
    ddeb = ticks.index[0]
    dfin = ddeb + timedelta(hours=decal)
    dti = ticks.loc[ddeb:dfin]
    rko = tick21renko(dti, None, step=step, value='bid')
    if rko is None:
        raise 'impossible création Renko'
    while len(rko) < min:
        ddeb = dfin
        dfin = ddeb + timedelta(hours=interval)
        dti = ticks.loc[ddeb:dfin]
        while len(dti) > 0 and dti.index[0] <= ddeb:
            dti.drop(dti.index[0], inplace=True)
        if len(dti) > 0:
            rko = tick21renko(dti, rko, step=step, value='bid')
    return dfin

# =================================================================== Gemini solution
def update_renko_bricks(
    df_renko_existing: pd.DataFrame | None,
    df_new_ticks: pd.DataFrame,
    price_col: str,
    brick_size: float,
    mode_gapped: bool = False # False = Progressive, True = Gapped
) -> pd.DataFrame:

    if df_new_ticks is None or df_new_ticks.empty or price_col not in df_new_ticks.columns:
        print(f"Erreur: DataFrame vide ou colonne '{price_col}' manquante.")
        return pd.DataFrame(columns=colonnesRko)
    prices = df_new_ticks[price_col].values
    times_ns = df_new_ticks.index
    # --- 1. INITIALISATION DES ANCRES (VARIABLES D'ÉTAT SCALAIRES) 🎯 ---
    if prices.size == 0:
        return df_renko_existing if df_renko_existing is not None else pd.DataFrame()
    is_initial_run = df_renko_existing is None or df_renko_existing.empty
    if is_initial_run:
        open_renko_iter = math.floor(prices[0] / brick_size) * brick_size
        open_price_source_iter = prices[0]
        open_time_iter_ns = times_ns[0]
        last_brick = [open_time_iter_ns, open_renko_iter, open_price_source_iter, 0.0, open_price_source_iter, 0.0, 0.0]
        df_renko_closed_history = pd.DataFrame()
    else:
        open_time_source_iter_ns = df_renko_existing.index[-1]
        last_brick = [open_time_source_iter_ns, df_renko_existing.iat[-1, ROPENR],
                      df_renko_existing.iat[-1, ROPEN], 0.0, df_renko_existing.iat[-1, RLOW], 0.0, 0.0]
        open_renko_iter = last_brick[ROPENR]
        open_price_source_iter = last_brick[ROPEN]
        df_renko_closed_history = df_renko_existing.iloc[:-1].copy()
    renko_new_list = []
    # --- 2. BOUCLE ITERATIVE TICK-PAR-TICK (Maximized Speed) ---
    for current_price, current_time_ns in zip(prices, times_ns):
        price_diff = current_price - open_renko_iter
        if abs(price_diff) <= brick_size:
            _update_brick_list(last_brick, current_price)
            continue
        abs_total_bricks = math.floor(np.abs(price_diff) / brick_size)
        direction = np.sign(price_diff)
        if not mode_gapped:
            if abs_total_bricks > 1:
                time_difference_ns = current_time_ns - open_time_iter_ns
                total_source_jump = current_price - open_price_source_iter
                price_step = total_source_jump / abs_total_bricks
                time_step_ns = time_difference_ns / abs_total_bricks
                # Préparation pour le mode PROGRESSIVE (mode_gapped = False)
                for k in range(abs_total_bricks -1):
                    # Mise à jour des variables de travail pour la brique suivante (N+1)
                    open_price_source_iter = open_price_source_iter + price_step * direction
                    open_renko_iter = open_renko_iter + brick_size * direction
                    last_brick[RCLOSER] = open_renko_iter
                    last_brick[RCLOSE] = open_price_source_iter
                    _update_brick_list(last_brick, open_price_source_iter)
                    renko_new_list.append(last_brick)
                    last_brick = [0] * len(colonnesRko)
                    open_time_iter_ns = open_time_iter_ns + time_step_ns
                    last_brick[RTIME] = open_time_iter_ns
                    last_brick[ROPENR] = open_renko_iter
                    last_brick[ROPEN] = open_price_source_iter
                    last_brick[RLOW] = open_price_source_iter
                    abs_total_bricks -= 1
        # GAPPED or NOT
        open_renko_iter = last_brick[ROPENR] + brick_size * direction * abs_total_bricks
        last_brick[RCLOSER] = open_renko_iter
        last_brick[RCLOSE] = current_price
        _update_brick_list(last_brick, current_price)
        renko_new_list.append(last_brick)
        last_brick = [0] * len(colonnesRko)
        open_time_iter_ns = current_time_ns
        last_brick[RTIME] = open_time_iter_ns
        last_brick[ROPENR] = open_renko_iter
        last_brick[ROPEN] = current_price
        last_brick[RLOW] = current_price
    renko_new_list.append(last_brick)
    # --- 4. CONCATÉNATION FINALE ---
    if renko_new_list:
        df_new_bricks = pd.DataFrame(
            renko_new_list, columns=colonnesRko)
        df_new_bricks['time'] = pd.to_datetime(df_new_bricks['time'], unit='ns')
        df_new_bricks = df_new_bricks.set_index('time', drop=False)
    else:
        df_new_bricks = pd.DataFrame()
    return pd.concat([df_renko_closed_history, df_new_bricks])
# ================================================================ pour rko pro et Zmq

def renko_from_ticks(ticks, brick_size=0.00010):
    """Transforme ticks en briques Renko."""
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df['close'] = (df['bid'] + df['ask']) / 2

    # Renko
    df['diff'] = df['close'].diff()
    df['direction'] = np.sign(df['diff'])
    df['cumsum'] = (df['diff'].abs() / brick_size).cumsum()
    df['brick'] = df['cumsum'].apply(lambda x: int(x) * brick_size * df['direction'].iloc[0])
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['close_renko'] = df['open'] + df['brick']
    df['up'] = df['close_renko'] > df['open']

    # Briques finales
    bricks = []
    current_open = df['close'].iloc[0]
    for _, row in df.iterrows():
        if abs(row['close_renko'] - current_open) >= brick_size:
            bricks.append({
                'time': row.name,
                'open': current_open,
                'close': row['close_renko'],
                'up': row['up']
            })
            current_open = row['close_renko']
    return bricks

def analyze_renko_speed(df_bricks):
    """
    Analyse la durée des briques Renko pour détecter un emballement.
    La durée d'une brique = temps de sa création jusqu'à la création de la suivante.
    """
    if len(df_bricks) < 4:
        return None
    # On extrait les temps
    times = df_bricks['time'].to_numpy()
    # Calcul des durées de vie des briques (différence entre la suivante et la courante)
    # Pour un tableau de N éléments, diff(1) donne time[i] - time[i-1].
    # Pour avoir time[i+1] - time[i], on fait un diff inversé ou on décale :
    durations = np.diff(times)  # durées de toutes les briques passées
    # La dernière brique de df_bricks est en cours, sa durée complète n'est pas figée.
    # On regarde donc les briques déjà clôturées (les 3 dernières de notre liste de durées)
    last_completed_durations = durations[-3:]  # [N-3, N-2, N-1]
    # Identifier la plus courte parmi ces dernières briques
    min_duration = np.min(last_completed_durations)
    is_current_shortest = (last_completed_durations[-1] == min_duration)
    return {
        "durations": last_completed_durations,  # Les 3 dernières durées [N-2, N-1, Courante clôturée]
        "min_duration": min_duration,
        "is_shortest": is_current_shortest  # True si la dernière brique clôturée est la plus rapide des 3
    }

def analyse_renko_moy(all_bricks):
    # On extrait les temps
    # 1. Convertir les objets datetime en timestamps bruts en millisecondes (float ou int)
    # .astype('int64') / 1_000_000 convertit les nanosecondes internes de Pandas/Numpy en millisecondes
    times_ms = all_bricks['time'].to_numpy().astype('int64') / 1_000_000
    # 2. Calculer les durées (différences entre la suivante et la courante)
    durations = np.diff(times_ms) / 1000
    # 3. Calculs statistiques
    # 2. Trouver la valeur minimale et son indice exact
    min_index = np.argmin(durations)
    min_duration = durations[min_index]
    moy_duration = np.mean(durations)
    std_duration = np.std(durations)
    z_duration = (min_duration - moy_duration) / std_duration
    # 3. Récupérer le 'time' associé à ce minimum
    # min_index correspond au début de la brique (ou min_index + 1 pour la fin de la brique)
    min_time_raw = all_bricks['time'].iloc[min_index]
    print(f"⚡ Plus courte : {min_duration:.3f} s (le {min_time_raw})\n"
          f"📊 Moyenne : {moy_duration:.1f} s, std {std_duration:.1f} s, z {z_duration:.2f}")

def analyse_renko_iqr(all_bricks):
    # 1. Calcul des durées en secondes (ou millisecondes)
    times_ms = all_bricks['time'].to_numpy().astype('int64') / 1_000_000
    durations = np.diff(times_ms)/1000
    # 2. Calcul de la Médiane et des Quartiles (Q1 à 25%, Q3 à 75%)
    median_duration = np.median(durations)
    """
    q25, q75 = np.percentile(durations, [25, 75])
    iqr = q75 - q25
    # 3. Définir un seuil d'anomalie basé sur l'IQR
    # En statistique, on qualifie souvent de "valeur aberrante sévère" tout ce qui est en dessous de : Q1 - 1.5 * IQR
    # Pour un emballement (vitesse extrême), on regarde les valeurs très basses :
    lower_bound = q25 - 1.5 * iqr
    # Alternative : un ratio strict par rapport à la médiane (ex: 5% ou 10% de la médiane)
    # threshold = median_duration * 0.05
    # 4. Analyser la brique la plus courte et sa robustesse
    min_duration = np.min(durations)
    min_index = np.argmin(durations)
    min_time_raw = all_bricks['time'].iloc[min_index]
    print(f"✅ Analyse IQR réussie :\n"
          f"⚡ Plus courte : {min_duration:.1f} s (le {min_time_raw})\n"
          f"📊 Médiane : {median_duration:.1f} s | IQR : {iqr:.1f} s\n"
          f"🎯 Seuil d'alerte (Q1 - 1.5*IQR) : {lower_bound:.1f} s")
    # 5. Exemple d'alerte sur la toute dernière brique (la courante)
    current_duration = min_duration #durations[-1]
    if current_duration < lower_bound:
        print(f"🚨 ALERTE EMBALLEMENT : Brique actuelle de {current_duration:.1f}s (anormalement rapide !)")
    """
    # 1. Calculs de base

    # 2. Définition de plusieurs seuils d'emballement potentiels
    # Option A : Pourcentage de la médiane (ex: 5% ou 10% du temps médian)
    #threshold_5_percent = median_duration * 0.05
    #threshold_10_percent = median_duration * 0.10
    # Option B : Temps fixe en secondes (ex: moins de 60 secondes ou moins de 300 secondes)
    threshold_10s = 5.0
    threshold_60s = 60.0
    threshold_300s = 300.0
    # 3. Compter combien de briques tombent sous ces seuils
    #count_5_pct = np.sum(durations < threshold_5_percent)
    #count_10_pct = np.sum(durations < threshold_10_percent)
    maxi = 15
    mini = 5

    count_10s = np.sum((durations < maxi) & (durations > mini))
    #count_60s = np.sum(durations < threshold_60s)
    #count_300s = np.sum(durations < threshold_300s)

    total_bricks = len(durations)

    # 4. Affichage des résultats pour choisir le meilleur réglage
    print(f"📊 Analyse des seuils d'emballement (sur {total_bricks} briques) :")
    print(f"   - Médiane globale : {median_duration:.1f} s (~{median_duration / 60:.1f} min)")
    print(f"   + Briques urgentes {count_10s} entre {mini} et {maxi} secondes")
    #print(f"   ⚡ Briques < 5% de la médiane (< {threshold_5_percent:.1f}s) : {count_5_pct} briques")
    #print(f"   ⚡ Briques < 10% de la médiane (< {threshold_10_percent:.1f}s) : {count_10_pct} briques")
    #print(f"   ⚡ Briques < 5 secondes (ultra-rapide) : {count_10s} briques")
    #print(f"   ⚡ Briques < 60 secondes (ultra-rapide) : {count_60s} briques")
    #print(f"   ⚡ Briques < 5 minutes (rapide) : {count_300s} briques")
