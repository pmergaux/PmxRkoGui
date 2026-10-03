import pandas as pd
import numpy as np
#from debugpy._vendored.pydevd.pydevd_attach_to_process.winappdbg import debug
from ta import trend, volatility, momentum
import warnings
from numba import njit
from numba import config
config.DISABLE_JIT = False  # S'assurer que ce n'est pas désactivé
config.NUMBA_DISABLE_JIT = False

from utils.utils import calculer_stats

# Placer ceci au tout début de votre script
warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

diff_col = ['diff_close','diff_ema', 'diff_rsi', 'diff_cci', 'diff_macd', 'diff_atr']
signal_col = ['signal_ema', 'signal_rsi', 'signal_macd', 'signal_cci', 'signal_atr']
indic_col = ['EMA', 'CCI', 'ATR', 'RSI', 'bb_mavg', 'bb_hband', 'bb_lband']

# ------------------------------------------------------------------------------
# Fonctions utilitaires
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return [macd, signal_line, macd-signal_line]

#  une moyenne exponentielle d'une colonne df['value'].ewm(span=14).mean()

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md)

def calculate_williams_r(high, low, close, period=14):
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest)

def calculate_stoch_rsi(rsi, period=14):
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min)

def calculate_time_live(df, cfg):
    # 1. Calcul de base
    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)
    # 2. Volatilité "Augmentée" (incluant le slippage réel)
    # abs(close - closer) capture l'impulsion finale au-delà de la brique
    volatilite_reelle = (abs(df['close'] - df['close_renko']) + cfg['parameters']['renko_size']) / df['close']
    # 3. Calcul de la vélocité
    df['time_live'] = volatilite_reelle / (df['time_diff'] + 1e-6)
    # 4. Compression logarithmique (L'astuce du curieux)
    df['time_live'] = np.log1p(df['time_live'])
    # Nettoyage final
    df['time_live'] = df['time_live'].replace([np.inf, -np.inf], 0).fillna(0)
    return df

def calculate_japonais(df: pd.DataFrame):
    df = df.copy()
    df['bb_mavg'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
    df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
    #df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
    #df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']

    df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    open_buy = ((df['close'] < df['bb_mavg'])) # & (df['direction'] == 1))
    open_sell = ((df['close'] > df['bb_mavg'])) # & (df['direction'] == -1))
    df['sigo'] = np.where(open_buy, 1, np.where(open_sell, -1, 0))
    close_sell = (df['low']*1.002 < df['bb_lband'])
    close_buy = (df['high']*1.002 > df['bb_hband'])
    df['sigc'] = np.where(close_sell, 1, np.where(close_buy, -1, 0))
    return df

def calculate_stochastic(df, window=5, slow_period=3, signal_period=3):
    # 1. Calcul du Stochastique "Fast"
    stoch_gen = momentum.StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=window,
        smooth_window=1  # On désactive le lissage interne pour tout contrôler ici
    )
    k_fast = stoch_gen.stoch()
    # 2. On crée le "Slow K" (le vrai oscillateur)
    # C'est cette ligne que vous devez utiliser pour le signal !
    k_slow = k_fast.rolling(window=slow_period).mean()
    # 3. On crée le "Signal" (le %D)
    # C'est la moyenne mobile du Slow K
    signal = k_slow.rolling(window=signal_period).mean()
    # 4. LOGIQUE DE SIGNAL AVANCÉE (Seuils Dynamiques et Momentum) :
    k_diff = k_slow.diff()
    
    # --- Création des seuils dynamiques (Type Bandes de Bollinger sur Stochastique) ---
    # Au lieu d'utiliser 80/20 de manière rigide, on calcule la moyenne et l'écart-type 
    # de l'oscillateur sur les 50 dernières périodes.
    lookback = 50
    stoch_mean = k_slow.rolling(window=lookback).mean()
    stoch_std = k_slow.rolling(window=lookback).std()
    
    # On définit l'Upper et le Lower bound dynamiques (bornés pour éviter l'absurde)
    # Dans un marché plat, les seuils se resserrent (ex: 70/30). Dans une forte tendance, ils s'écartent (ex: 90/10).
    upper_bound = np.clip(stoch_mean + 1.5 * stoch_std, 65, 95)
    lower_bound = np.clip(stoch_mean - 1.5 * stoch_std, 5, 35)
    
    # A. Signal de base par croisement
    stoch_val = np.where(k_slow > signal, 1, np.where(k_slow < signal, -1, 0))
    
    # B. Épuisement (Sécurité avec seuils dynamiques)
    # On bloque les signaux d'Achat si on est déjà au-dessus de notre limite dynamique haute
    stoch_val = np.where((k_slow > upper_bound) & (stoch_val == 1), 0, stoch_val)
    # On bloque les signaux de Vente si on est déjà en-dessous de notre limite dynamique basse
    stoch_val = np.where((k_slow < lower_bound) & (stoch_val == -1), 0, stoch_val)
    
    # C. Anticipation de retournement (Impulsion depuis l'extrême dynamique)
    # Si on franchit la bande haute et que ça pique vers le bas -> VENTE anticipée
    stoch_val = np.where((k_slow > upper_bound) & (k_diff < 0), -1, stoch_val)
    # Si on franchit la bande basse et que ça rebondit -> ACHAT anticipé
    stoch_val = np.where((k_slow < lower_bound) & (k_diff > 0), 1, stoch_val)

    df['stoch'] = stoch_val
    # Optionnel : ajout des colonnes pour debug
    # df['stoch_k'] = k_slow
    # df['stoch_d'] = signal
    return df

def calculate_sar(df, pas=0.02, maxi=0.2):
    # 1. PSAR Standard (Classique) - idéal pour les marchés calmes, filtre le bruit
    psar_calm = trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step=pas, max_step=maxi).psar()
    
    # 2. PSAR Agressif (Rapide) - colle au prix, idéal pour sécuriser vite lors des explosions
    psar_explo = trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step=pas*2.5, max_step=maxi*2).psar()
    
    # 3. Évaluation dynamique de la volatilité
    # On compare l'écart-type court terme (20) à la moyenne de l'écart-type moyen terme (50)
    std_court = df['close'].rolling(20).std()
    std_moyen = std_court.rolling(50).mean()
    
    # On bascule en mode agressif si la volatilité explose (ex: +50% par rapport à la moyenne)
    is_volatile = std_court > (std_moyen * 1.5)
    
    # 4. Mixage Dynamique
    psar_dynamique = np.where(is_volatile, psar_explo, psar_calm)
    
    df['psar'] = np.where(psar_dynamique > df['high'], -1, np.where(psar_dynamique < df['low'], 1, 0))
    return df

def fast_calculer_stats_vector(df, window=14, column='close'):
    """
    Calcule Pente, Std, VolLog et R2 de manière ultra-rapide.
    Version optimisée pour les backtests et Optuna.
    """
    y = df[column].values
    n = len(y)
    x = np.arange(window)

    # Initialisation des tableaux de résultats
    pentes = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    vols_log = np.full(n, np.nan)
    r2s = np.full(n, np.nan)
    ers = np.full(n, np.nan)  # Ajout de l'ER

    # Constantes pour la régression (ne dépendent que de la fenêtre)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)

    # Boucle glissante optimisée
    for i in range(window - 1, n):
        y_seg = y[i - window + 1: i + 1]
        y_mean = np.mean(y_seg)

        # 1. Pente (Slope) - Version mathématique directe
        slope = np.sum((x - x_mean) * (y_seg - y_mean)) / ss_x
        pentes[i] = slope

        # 2. Écart-type Standard
        stds[i] = np.std(y_seg)

        # 3. Volatilité Logarithmique (%)
        # On calcule les log-rendements sur le segment
        log_returns = np.diff(np.log(y_seg))
        vols_log[i] = np.std(log_returns) * 100

        # 4. R2 (Coefficient de détermination)
        y_hat = slope * (x - x_mean) + y_mean
        ss_res = np.sum((y_seg - y_hat) ** 2)
        ss_tot = np.sum((y_seg - y_mean) ** 2)
        r2s[i] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 5. ER (Efficiency Ratio de Kaufman)
        # On calcule le mouvement net sur la fenêtre vs le bruit total
        change = np.abs(y_seg[-1] - y_seg[0])
        vol_abs = np.sum(np.abs(np.diff(y_seg)))
        ers[i] = change / vol_abs if vol_abs != 0 else 0

    return pentes, stds, vols_log, r2s, ers


def is_volatility_too_high(df, ind_cfg, threshold_multiplier=1.8):
    """
    Filtre de volatilité basé sur time_live + volatility
    Retourne True si on doit SKIP le trade (trop risqué)
    """
    if len(df) < 30:
        return False

    recent = df.tail(30)

    # time_live = volatility / time_diff → très bon indicateur de régime
    avg_time_live = recent['time_live'].mean()
    max_time_live = recent['time_live'].max()

    # volatility brute
    avg_vol = recent['volatility'].mean()

    # Seuil dynamique selon la config
    base_threshold = ind_cfg.get('veto', {}).get('z_threshold', 2.5)

    if (max_time_live > base_threshold * 1.6) or \
            (avg_time_live > base_threshold * 1.35) or \
            (avg_vol > recent['volatility'].quantile(0.85) * 1.7):
        print(f"⛔ Volatilité trop élevée - time_live: {avg_time_live:.2f} | vol: {avg_vol:.1f}")
        return True

    return False

@njit(fastmath=True, cache=True)
def fast_stats_numba(y_seg):
    n = y_seg.size
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x_mean = (n - 1) / 2.0

    # Calcul manuel de la moyenne pour éviter les overheads de np.mean sur petites fenêtres
    sum_y = 0.0
    for val in y_seg:
        sum_y += val
    y_mean = sum_y / n

    ss_x = 0.0
    ss_xy = 0.0
    ss_tot = 0.0

    for i in range(n):
        x_diff = i - x_mean
        y_diff = y_seg[i] - y_mean
        ss_x += x_diff ** 2
        ss_xy += x_diff * y_diff
        ss_tot += y_diff ** 2

    # 3. PENTE
    pente = ss_xy / ss_x if ss_x > 1e-12 else 0.0

    # 4. ÉCART-TYPE
    variance = ss_tot / n
    std = np.sqrt(variance)

    # 5. VOLATILITÉ LOGARITHMIQUE
    # On calcule manuellement le log pour éviter np.log(0) et errstate
    vol_log_pct = 0.0
    log_ret_sum = 0.0
    log_ret_sq_sum = 0.0
    count = 0

    for i in range(1, n):
        # On s'assure que y_seg > 0 avant le log
        if y_seg[i - 1] > 1e-12 and y_seg[i] > 1e-12:
            lr = np.log(y_seg[i] / y_seg[i - 1])
            log_ret_sum += lr
            log_ret_sq_sum += lr * lr
            count += 1

    if count > 1:
        lr_mean = log_ret_sum / count
        vol_log_pct = np.sqrt(max(0, (log_ret_sq_sum / count) - (lr_mean ** 2))) * 100

    # 6. R2
    # ss_res = sum((y_seg[i] - (pente * (i - x_mean) + y_mean))**2)
    # Plus rapide : ss_res = ss_tot - (ss_xy^2 / ss_x)
    ss_res = ss_tot - (ss_xy ** 2 / ss_x) if ss_x > 1e-12 else ss_tot
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # 7. ER (Efficiency Ratio)
    vol_abs = 0.0
    for i in range(1, n):
        vol_abs += abs(y_seg[i] - y_seg[i - 1])

    er = abs(y_seg[-1] - y_seg[0]) / vol_abs if vol_abs > 1e-12 else 0.0

    return pente, std, vol_log_pct, r2, er

def fast_stats_single(y_seg):
    """
    Calcule Pente, Std, VolLog et R2 pour un seul segment de données.
    y_seg : array-like (les 'n' dernières valeurs de 'close')
    """
    n = len(y_seg)
    x = np.arange(n)

    # 1. Moyennes
    x_mean = (n - 1) / 2.0  # Moyenne de np.arange(n)
    y_mean = np.mean(y_seg)

    # 2. Composantes pour la régression (Moindres Carrés)
    x_diff = x - x_mean
    y_diff = y_seg - y_mean

    ss_x = np.sum(x_diff ** 2)
    ss_xy = np.sum(x_diff * y_diff)

    # 3. PENTE ($/brique)
    pente = ss_xy / ss_x if ss_x != 0 else 0

    # 4. ÉCART-TYPE ($)
    std = np.std(y_seg)

    # 5. VOLATILITÉ LOGARITHMIQUE (%)
    # On évite le log(0) ou valeurs négatives au cas où
    with np.errstate(divide='ignore', invalid='ignore'):
        log_returns = np.diff(np.log(y_seg))
        vol_log_pct = np.std(log_returns) * 100 if len(log_returns) > 0 else 0

    # 6. R2 (Coefficient de détermination)
    y_hat = pente * x_diff + y_mean
    ss_res = np.sum((y_seg - y_hat) ** 2)
    ss_tot = np.sum(y_diff ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # 7. ER (Efficiency Ratio)
    change = np.abs(y_seg[-1] - y_seg[0])
    vol_abs = np.sum(np.abs(np.diff(y_seg)))
    er = change / vol_abs if vol_abs != 0 else 0

    return pente, std, vol_log_pct, r2, er

def fast_r2(df, window=14):
    """
    Calcul vectorisé du R2 (Coefficient de détermination)
    Identique à calculer_stats mais 100x plus rapide pour Optuna.
    """
    y = df['close'].values
    x = np.arange(window)

    def get_r2(y_seg):
        if len(y_seg) < window: return 0
        coeffs = np.polyfit(x, y_seg, 1)
        p = np.poly1d(coeffs)
        y_hat = p(x)
        y_bar = np.mean(y_seg)
        ss_res = np.sum((y_seg - y_hat) ** 2)
        ss_tot = np.sum((y_seg - y_bar) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Utilisation de rolling.apply pour la rapidité
    return df['close'].rolling(window=window).apply(get_r2)

def calculate_r2(df, window=14, zone='close'):
    """
    Applique calculer_stats sur une fenêtre glissante pour obtenir le R2.
    """
    # On initialise la colonne avec une valeur neutre
    r2_values = np.zeros(len(df))
    vol_log_pcts = np.zeros(len(df))
    # On commence le calcul seulement quand on a assez de données pour la fenêtre (inclus la barre actuelle)
    for i in range(window - 1, len(df)):
        # Extraction du segment incluant la barre actuelle à l'index i
        df_segment = df.iloc[i - window + 1 : i + 1]

        # Appel à votre fonction existante
        pente, std, vol_log_pct, r2, er = calculer_stats(df_segment, zone=zone)

        r2_values[i] = r2
        vol_log_pcts[i] = vol_log_pct

    df['r2'] = r2_values
    df['vol_log_pct'] = vol_log_pcts
    return df

def calculate_atr(df, window=14):
    df_clean = df.dropna(subset=['high', 'low', 'close'])
    return volatility.AverageTrueRange(df_clean['high'], df_clean['low'], df_clean['close'], window).average_true_range()

def calculate_atr_4sl(df, multiplier=2.0, window=14):
    """
    import pandas_ta as ta  # ou la librairie 'ta' selon votre préférence
    Calcule la distance de stop optimale basée sur la volatilité.
    multiplier : 1.5 (agressif/serré), 2.0 (standard), 3.0 (large/tendance)
    atr_ser = ta.atr(df['high'], df['low'], df['close'], length=window)
    """
    # Calcul de l'ATR (Volatility)
    # Si vous utilisez la librairie 'ta' :
    # from ta.volatility import AverageTrueRange
    # 2. Nettoyage des données (Sécurité)
    atr_ser = calculate_atr(df, window)
    current_atr = atr_ser.iloc[-1]
    # print(f"current_atr: {atr_ser[-3:]}")
    # La distance de sécurité est un multiple de la volatilité moyenne
    return current_atr * multiplier

def calculate_efficiency_ratio(df, window=10):
    """
    Calcule l'Efficience Ratio.
    Ajout d'une sécurité pour la division par zéro et nettoyage.
    """
    # 1. Calcul de la variation nette sur la période
    net_change = (df['close'] - df['close'].shift(window)).abs()
    # 2. Somme des variations absolues (bruit) sur la même fenêtre
    total_noise = df['close'].diff().abs().rolling(window=window).sum()
    # 3. Calcul de l'ER avec sécurité division par zéro
    # Si le bruit est nul (marché plat), l'ER est 0
    er = np.where(total_noise != 0, net_change / total_noise, 0.0)
    # 4. Affectation au DataFrame
    df['er'] = er
    return df

def calculate_vwap(df, window=24):
    # 1. Calcul du VWAP classique
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap = (tp * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap

def calculate_vwap_zscore(df, window=24):
    # print(f"DEBUG: Taille DF = {len(df)}")
    vwap = calculate_vwap(df, window)
    # 2. Calcul de l'écart brut (Distance)
    distance = df['close'] - vwap
    # 3. Calcul du Z-Score de la distance
    # On regarde si l'écart actuel est supérieur à la moyenne des écarts récents
    mean_dist = distance.rolling(window=window).mean()
    std_dist = distance.rolling(window=window).std()
    result = (distance - mean_dist) / std_dist
    # print(f"DEBUG: Taille Résultat = {len(result)}")  # Doit être égal à Taille DF
    df['vwap_z'] = result.fillna(0)
    return df

def calculate_vwap_score(df, window=24):
    vwap = calculate_vwap(df, window)
    # On retourne la distance relative (en %) car c'est ce qui aide le RNN
    # 0.001 = le prix est 0.1% au-dessus du VWAP
    df['vwap'] = (df['close'] - vwap) / vwap
    return df

def is_market_safe(df, pos, cfg):
    """
    Retourne True si le marché est considéré comme 'trader-friendly'.
    """
    window = cfg.get('er', {}).get('window', 10)
    er_min = cfg.get('veto', {}).get('er_min', 0.38)
    z_threshold = cfg.get('veto', {}).get('z_threshold', 2.0)
    # 1. Sécurité sur la longueur
    if pos < window:
        return True
    # 2. Récupération de l'ER (soit pré-calculé, soit calculé à la volée)
    if 'er' in df.columns:
        # On prend la valeur à l'index actuel 'pos'
        er = df.at[df.index[pos], 'er']
    else:
        # Calcul à la volée sur la fenêtre glissante
        df_slice = df.iloc[pos - window: pos + 1]
        move = abs(df_slice['close'].iloc[-1] - df_slice['close'].iloc[0])
        noise = df_slice['close'].diff().abs().sum()
        er = move / noise if noise != 0 else 0
    """
    # 3. Volatilité (ROC)
    # Note : on utilise df.iloc pour calculer le ROC sur l'index 'pos'
    close_val = df['close'].iloc[pos]
    prev_close = df['close'].iloc[pos - 3]
    current_roc = abs((close_val - prev_close) / prev_close)
    # Calcul de la moyenne sur les 20 dernières briques
    avg_roc = df['close'].pct_change(3).abs().rolling(window=window).mean().iloc[pos]
    # 4. Filtres
    # Si ER < 0.2 (trop de bruit) OU si la volatilité actuelle est 3x supérieure à la moyenne
    # Utilisation d'une petite sécurité pour éviter avg_roc=0
    if er < 0.2 or current_roc > (avg_roc * 3 if avg_roc > 0 else 1.0):
        return False
    """
    # 3. utilisation du zscore
    # Plus robuste statistiquement que le ROC
    move = df['close'].diff().abs()
    mean_move = move.rolling(window).mean().iloc[pos]
    std_move = move.rolling(window).std().iloc[pos]

    current_move = abs(df['close'].iloc[pos] - df['close'].iloc[pos - 1])
    z_score = (current_move - mean_move) / std_move if std_move > 0 else 0

    # 3. Filtre unifié
    if er < er_min or z_score > z_threshold:
        return False
    return True

def is_market_exploding(df, cfg):
    window = cfg.get('z_score', {}).get('window', 10)
    # On calcule la moyenne et l'écart-type des variations de prix
    df_clean = df.dropna(subset=['high', 'low', 'close'])
    df_clean['explode'] = df_clean['close'].diff().abs()
    mean_move = df_clean['explode'].rolling(window).mean()
    std_move = df_clean['explode'].rolling(window).std()
    # Z-Score : à quel point le mouvement actuel est-il anormal ?
    current_move = abs(df_clean['close'].iloc[-1] - df_clean['close'].iloc[-2])
    z_score = (current_move - mean_move.iloc[-1]) / std_move.iloc[-1]
    return z_score > cfg.get('veto').get('z_score', 2.5)  # True si le mouvement est 2x plus fort que d'habitude

# ====================== INDICATEURS DE BASE (OPTIMISÉ) ======================
@njit(fastmath=True, cache=True)
def compute_all_indicators(close, high, low, ts, ema_p, rsi_p, macd_f, macd_s, macd_sig, cci_p, atr_p):
    n = len(close)

    rsi = np.full(n, np.nan, dtype=np.float64)
    ema = np.full(n, np.nan, dtype=np.float64)
    macd_hist = np.full(n, np.nan, dtype=np.float64)
    macd_line = np.zeros(n, dtype=np.float64)
    macd_signal = np.zeros(n, dtype=np.float64)
    cci = np.full(n, np.nan, dtype=np.float64)
    bb_mavg = np.full(n, np.nan, dtype=np.float64)
    bb_hband = np.full(n, np.nan, dtype=np.float64)
    bb_lband = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    volatility = np.zeros(n, dtype=np.float64)
    time_vol = np.zeros(n, dtype=np.int32)

    # EMA
    alpha = 2.0 / (ema_p + 1.0)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]

    # MACD
    ema_f = np.zeros(n, dtype=np.float64)
    ema_s = np.zeros(n, dtype=np.float64)
    ema_f[0] = ema_s[0] = close[0]
    for i in range(1, n):
        ema_f[i] = (2.0 / (macd_f + 1)) * close[i] + (1 - 2.0 / (macd_f + 1)) * ema_f[i - 1]
        ema_s[i] = (2.0 / (macd_s + 1)) * close[i] + (1 - 2.0 / (macd_s + 1)) * ema_s[i - 1]
    macd_line = ema_f - ema_s
    if n >= macd_sig:
        macd_signal[macd_sig - 1] = np.mean(macd_line[:macd_sig])
        for i in range(macd_sig, n):
            macd_signal[i] = (2.0 / (macd_sig + 1)) * macd_line[i] + (1 - 2.0 / (macd_sig + 1)) * macd_signal[i - 1]
        macd_hist[:] = macd_line - macd_signal

    # Boucle principale
    for i in range(n):
        time_vol[i] = ((ts[i] // 3600) % 24) * 100 + ((ts[i] // 60) % 60)
        volatility[i] = (high[i] - low[i]) / close[i]

        # Bollinger
        if i >= 19:
            ma = 0.0
            for k in range(i-19, i+1):
                ma += close[k]
            ma /= 20
            var = 0.0
            for k in range(i-19, i+1):
                var += (close[k] - ma)**2
            std = np.sqrt(var / 20)
            bb_mavg[i] = ma
            bb_hband[i] = ma + 2 * std
            bb_lband[i] = ma - 2 * std

        # ATR
        if i >= atr_p:
            tr_sum = 0.0
            for j in range(i - atr_p + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr_sum += tr
            atr[i] = tr_sum / atr_p

        # RSI
        if i >= rsi_p:
            gains = losses = 0.0
            for j in range(i - rsi_p + 1, i + 1):
                diff = close[j] - close[j-1]
                if diff > 0:
                    gains += diff
                else:
                    losses -= diff
            rs = (gains / rsi_p) / ((losses / rsi_p) + 1e-12)
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        # CCI
        if i >= cci_p:
            tp_sum = 0.0
            for j in range(i - cci_p + 1, i + 1):
                tp_sum += (high[j] + low[j] + close[j])
            ma = tp_sum / cci_p
            md = 0.0
            for j in range(i - cci_p + 1, i + 1):
                md += abs((high[j] + low[j] + close[j])/3 - ma)
            md /= cci_p
            cci[i] = ((high[i] + low[i] + close[i])/3 - ma) / (0.015 * (md + 1e-12))

    return rsi, ema, macd_line, macd_signal, macd_hist, cci, time_vol, volatility, bb_mavg, bb_hband, bb_lband, atr


def add_indicators_optimized(df: pd.DataFrame, param: dict) -> pd.DataFrame:
    #df = df.reset_index(drop=True).copy()
    df = df.copy()
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    ts = df['time'].astype('int64').values // 10**9

    rsi, ema, macd_l, macd_s, macd_h, cci, tv, vol, bb_m, bb_h, bb_l, atr = compute_all_indicators(
        c, h, l, ts,
        param.get('ema_period', 9),
        param.get('rsi_period', 14),
        param.get('macd_fast', 12),
        param.get('macd_slow', 26),
        param.get('macd_signal', 9),
        param.get('cci_period', 20),
        param.get('atr_period', 14)
    )

    df['RSI'] = rsi
    df['EMA'] = ema
    df['MACD_line'] = macd_l
    df['MACD_signal'] = macd_s
    df['MACD_hist'] = macd_h
    df['CCI'] = cci
    df['time_vol'] = tv
    df['volatility'] = vol
    df['bb_mavg'] = bb_m
    df['bb_hband'] = bb_h
    df['bb_lband'] = bb_l
    df['ATR'] = atr

    df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)
    df['time_live'] = (df['volatility'] / (df['time_diff'] + 1e-6)).replace([np.inf, -np.inf], 0).fillna(0)

    return df.dropna()

def calculate_indicators(df : pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    #df.reset_index(drop=True)
    features = config['features']
    param = config["parameters"]
    try:
        df['bb_mavg'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_hband'] = df['bb_mavg'] + 2 * df['bb_std']
        df['bb_lband'] = df['bb_mavg'] - 2 * df['bb_std']
    #    df['bb_max'] = df['bb_mavg'] + param.get('niveau', 0.9) * df['bb_std']
    #    df['bb_min'] = df['bb_mavg'] - param.get('niveau', 0.9) * df['bb_std']
        target_col = config["target"]["target_col"]
        if not isinstance(target_col, list):
            target_col = [target_col]
        if 'EMA' in features or "EMA" in target_col or 'diff_ema' in features or 'diff_ema' in target_col:
            df['EMA'] = trend.EMAIndicator(df['close'], window=param.get('ema_period', 9)).ema_indicator()
        if 'RSI' in features or 'RSI' in target_col or 'diff_rsi' in features or 'diff_rsi' in target_col:
            df['RSI'] = momentum.RSIIndicator(df['close'], window=param.get('rsi_period', 14)).rsi()
        if 'MACD_hist' in features or "MACD_hist" in target_col or 'diff_macd' in features or 'diff_macd' in target_col:
            pmacd = param.get('macd', {"macd_fast":12, "macd_slow": 26, "macd_signal": 9})
            macd = trend.MACD(df['close'], window_fast=pmacd["macd_fast"], window_slow=pmacd["macd_slow"], window_sign=pmacd["macd_signal"])
            df['MACD_line'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
        if 'ATR' in features or 'ATR' in target_col or 'diff_atr' in features or 'diff_atr' in target_col:
            df['ATR'] = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=param.get('atr_period', 14)).average_true_range()
        if 'Stoch_RSI' in features:
            df['Stoch_RSI'] = momentum.StochRSIIndicator(df['close'], window=param.get('stochRsi_period', 14)).stochrsi()
        if 'Williams_R' in features:
            df['Williams_R'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=param.get('williamsR_period',14)).williams_r()
        if 'CCI' in features or 'CCI' in target_col or 'diff_cci' in features or 'diff_cci' in target_col:
            df['CCI'] = trend.CCIIndicator(df['high'], df['low'], df['close'], window=param.get('cci_period', 14)).cci()
        if 'time_live' in features:
            df = calculate_time_live(df, config)
        #print(f"Colonnes après direction_openr_closer: {df.columns.tolist()}")
        return df.dropna()   #.reset_index(drop=True)
    except BaseException as e:
        print(f"Erreur dans calculate_indicators: {e}")
        raise "Erreur dans calculate_indicators"

@njit(fastmath=True, cache=True)
def compute_signals_numba(
        close, open_, ema, rsi, macd_hist, atr, stoch_rsi, williams_r, cci,
        rsi_high, rsi_low, s_rsi_high, s_rsi_low,
        williams_high, williams_low, cci_high, cci_low
):
    n = len(close)
    # Initialisation des buffers (toujours des zéros pour éviter les problèmes de NaN)
    direction = np.zeros(n, dtype=np.int32)
    sigo = np.zeros(n, dtype=np.int32)
    sigc = np.zeros(n, dtype=np.int32)

    diff_ema = np.zeros(n, dtype=np.float64)
    diff_rsi = np.zeros(n, dtype=np.float64)
    diff_macd = np.zeros(n, dtype=np.float64)
    diff_atr = np.zeros(n, dtype=np.float64)
    diff_cci = np.zeros(n, dtype=np.float64)
    diff_close = np.zeros(n, dtype=np.float64)
    signal_ema = np.zeros(n, dtype=np.int32)
    signal_rsi = np.zeros(n, dtype=np.int32)
    signal_macd = np.zeros(n, dtype=np.int32)
    signal_atr = np.zeros(n, dtype=np.int32)
    signal_cci = np.zeros(n, dtype=np.int32)

    current_sigc = 0
    # opening_cond sera 1 si valide, 0 sinon
    opening_cond = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        # 1. Direction
        if close[i] > open_[i]:
            direction[i] = 1
        elif open_[i] > close[i]:
            direction[i] = -1
        elif i > 0:
            direction[i] = direction[i - 1]
        if i > 0 and close[i - 1] != 0:
            diff_close[i] = (close[i] - close[i - 1]) / close[i - 1]

        # 2. SIGC
        if i > 0 and direction[i] != direction[i - 1]: current_sigc = 4 * direction[i]
        sigc[i] = current_sigc

        # 3. Calculs indicateurs (avec protection contre les zéros/vides)
        buy = direction[i] == 1
        sell = direction[i] == -1

        # EMA
        if i > 0 and close[i] != 0:
            signal_ema[i] = 1 if (close[i] > ema[i] and ema[i] >= ema[i - 1]) else (
                -1 if (close[i] < ema[i] and ema[i] <= ema[i - 1]) else 0)
            diff_ema[i] = (ema[i] - close[i]) / close[i]

        # RSI
        signal_rsi[i] = 1 if rsi[i] < rsi_low else (-1 if rsi[i] > rsi_high else 0)
        diff_rsi[i] = (rsi[i] - 50.0) / 50.0

        # MACD
        signal_macd[i] = 1 if macd_hist[i] > 0 else (-1 if macd_hist[i] < 0 else 0)
        diff_macd[i] = macd_hist[i]

        # CCI
        signal_cci[i] = 1 if cci[i] < cci_low else (-1 if cci[i] > cci_high else 0)
        diff_cci[i] = cci[i] / 200.0

        # ATR
        if i > 50:
            m_atr = np.mean(atr[i - 50:i])
            signal_atr[i] = 1 if atr[i] > m_atr else (-1 if atr[i] < m_atr else 0)
            diff_atr[i] = (atr[i] - m_atr) / (m_atr + 1e-12)

        # 4. Logique de décision (Fusionnée ici)
        signals = np.empty(5, dtype=np.int32)
        signals[0] = signal_ema[i]
        signals[1] = signal_rsi[i]
        signals[2] = signal_macd[i]
        signals[3] = signal_atr[i]
        signals[4] = signal_cci[i]

        valid_signals = signals[signals != 0]

        if len(valid_signals) > 0 and np.all(valid_signals == valid_signals[0]):
            sigo[i] = valid_signals[0]
            opening_cond[i] = True

    return (direction, sigo, sigc, diff_close, diff_ema, diff_rsi,
            diff_macd, diff_atr, diff_cci, signal_ema, signal_rsi,
            signal_macd, signal_atr, signal_cci)

def choix_features_numba(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    for col in indic_col:
        if col not in df.columns:
            print(f"colonne manquante {col} de {df.columns.tolist()}")
            raise ValueError("colonne manquante")

    n = len(df)
    params = cfg.get("parameters", {})

    # Helper sécurisé
    def get_col(name):
        return df[name].values.astype(np.float64) if name in df.columns else np.zeros(n, dtype=np.float64)
    # Appel
    res = compute_signals_numba(
        df['close'].values, df['open'].values,
        get_col('EMA'), get_col('RSI'), get_col('MACD_hist'), get_col('ATR'),
        get_col('Stoch_RSI'), get_col('Williams_R'), get_col('CCI'),
        params.get('rsi_high', 70.0), params.get('rsi_low', 30.0),
        params.get('s_rsi_high', 0.8), params.get('s_rsi_low', 0.2),
        params.get('williams_high', -20.0), params.get('williams_low', -80.0),
        params.get('cci_high', 100.0), params.get('cci_low', -100.0)
    )

    # ASSIGNATION (avec diff_close à l'index 3)
    df['direction'] = res[0]
    df['sigo'] = res[1]
    df['sigc'] = res[2]
    df['diff_close'] = res[3]
    df['diff_ema'], df['diff_rsi'], df['diff_macd'], df['diff_atr'], df['diff_cci'] = res[4:9]
    df['signal_ema'], df['signal_rsi'], df['signal_macd'], df['signal_atr'], df['signal_cci'] = res[9:14]

    for col in diff_col:
        if col not in df.columns:
            print(f"diif colonne manquante {col} de {df.columns.tolist()}")
            raise ValueError("diff colonne manquante")
    return df

def choix_features(df: pd.DataFrame, cfg: dict):
    features = cfg['features']
    open_rules = cfg['open_rules']
    close_rules = cfg["close_rules"]
    param = cfg["parameters"]
    target = cfg["target"]
    rsi_high = param.get('rsi_high', 70)
    rsi_low = param.get('rsi_low', 30)
    s_rsi_high = param.get('s_rsi_high', 0.8)
    s_rsi_low = param.get('s_rsi_low', 0.2)
    williams_high = param.get('williams_low', -80)
    williams_low = param.get('williams_high', -20)
    cci_high = param.get('cci_high', 100)
    cci_low = param.get('cci_low', -100)
    """
    if 'closer' in df.columns:
        df['direction'] = np.where(df['close_renko'] > df['open_renko'], 1, np.where(df['open_renko'] > df['close_renko'], -1, 0))
    else:
    # distinguer close et close_renko ne sert à rien puisque pour les renko close est défini avec open et close
    """
    df['direction'] = np.where(df['close'] > df['open'], 1, np.where(df['open'] > df['close'], -1, 0))
    df['diff_close'] = (df['close'] - df['close'].shift(1))/df['close'].shift(1)
    buy_cond = (df['direction'] == 1)
    sell_cond = (df['direction'] == -1)
    df['sigc'] = 0
    if "EMA" in df.columns:
        buy_local  = ((df['close'] > df['EMA']) & (df['EMA'] >= df['EMA'].shift(1)))
        sell_local = ((df['close'] < df['EMA']) & (df['EMA'] <= df['EMA'].shift(1)))
        df['signal_ema'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_ema'] = (df['EMA'] - df['close'])/df['close']
    if "RSI" in df.columns:
        buy_local = (df['RSI'] < rsi_low)
        sell_local = (df['RSI'] > rsi_high)
        df['signal_rsi'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_rsi'] = (df['RSI'] - 50)/50
    if "MACD_hist" in df.columns:
        df['signal_macd'] = np.where(df['MACD_hist'] > 0, 1, np.where(df['MACD_hist'] < 0 , -1, 0))
        df['diff_macd'] = df['MACD_hist']
    if 'Stoch_RSI' in df.columns:
        buy_cond &= (df['Stoch_RSI'] < s_rsi_low)
        sell_cond &= (df['Stoch_RSI'] > s_rsi_high)
    if 'ATR' in df.columns:
        buy_local = (df['ATR'] > df['ATR'].mean())
        sell_local = (df['ATR'] < df['ATR'].mean())
        df['signal_atr'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_atr'] = (df['ATR'] - df['ATR'].mean())/df['ATR'].mean()
    if 'Williams_R' in df.columns:
        buy_cond &= (df['Williams_R'] < williams_low)
        sell_cond &= (df['Williams_R'] > williams_high)
    if 'CCI' in df.columns:
        buy_local = (df['CCI'] < cci_low)
        sell_local = (df['CCI'] > cci_high)
        df['signal_cci'] = np.select([buy_local, sell_local], [1, -1], default=0)
        df['diff_cci'] = df['CCI'] / 200
    if close_rules.get("close_sens", False):
        #clos_cond = ((df['direction'] != df['direction'].shift(1)))   #la direction actuelle avec la précédente
        #df['sigc'] = np.select([clos_cond & df['sigc']==0], [4*df['direction']], default=df['sigc'])
        clos_cond = (df['sigc'] == 0)  # toujours vraie donc sigc remplace direction
        df['sigc'] = np.select([clos_cond], [df['direction']], default=df['sigc'])
    first = True
    # une valeur par défaut
    opening_cond = (df['direction'] != 0)   # donc toujours vraie
    df['sigo'] = 0
    for col in signal_col:
        if first:
            if not col in df.columns:
                continue
            df['sigo'] = df[col]
            opening_cond = (df[col] != 0)
            first = False
            continue
        if col in df.columns:
            opening_cond &= ((df[col] == 0) | (df[col] == df['sigo']))
    df['sigo'] = np.where(opening_cond, 1 * df['sigo'], 0)
    return df


@njit(fastmath=True, cache=True)
def compute_decision_indicators(
        close, high, low, open_,
        bb_window=20,
        stoch_window=21,
        stoch_slow=5,
        stoch_signal=5,
        er_window=18,
        vwap_window=24,
        sar_step=0.02,
        sar_max=0.2,
):
    n = len(close)

    # === Buffers principaux ===
    bb_mavg = np.full(n, np.nan, dtype=np.float64)
    bb_hband = np.full(n, np.nan, dtype=np.float64)
    bb_lband = np.full(n, np.nan, dtype=np.float64)

    direction = np.zeros(n, dtype=np.int32)
    sigo = np.zeros(n, dtype=np.int32)
    sigc = np.zeros(n, dtype=np.int32)
    stoch = np.zeros(n, dtype=np.int32)
    er = np.zeros(n, dtype=np.float64)
    vwap_z = np.full(n, 0.0, dtype=np.float64)
    psar_signal = np.zeros(n, dtype=np.int32)

    # === SAR ===
    psar = np.full(n, 0.0, dtype=np.float64)
    af = np.full(n, sar_step, dtype=np.float64)
    is_long = np.full(n, True, dtype=np.bool_)

    # === VWAP ===
    tp_sum = np.zeros(n, dtype=np.float64)
    vol_sum = np.zeros(n, dtype=np.float64)

    # === Stochastic buffers ===
    k_fast = np.full(n, 50.0, dtype=np.float64)
    k_slow = np.full(n, 50.0, dtype=np.float64)
    d_line = np.full(n, 50.0, dtype=np.float64)

    for i in range(n):
        # ==================== DIRECTION ====================
        if close[i] > open_[i]:
            direction[i] = 1
        elif open_[i] > close[i]:
            direction[i] = -1
        elif i > 0:
            direction[i] = direction[i - 1]

        # ==================== BOLLINGER + JAPONAIS ====================
        if i >= bb_window - 1:
            slice_close = close[i - bb_window + 1:i + 1]
            ma = np.mean(slice_close)
            std = np.std(slice_close)
            bb_mavg[i] = ma
            bb_hband[i] = ma + 2 * std
            bb_lband[i] = ma - 2 * std

            sigo[i] = 1 if close[i] < ma else (-1 if close[i] > ma else 0)
            sigc[i] = 1 if low[i] * 1.002 < bb_lband[i] else (-1 if high[i] * 1.002 > bb_hband[i] else 0)

        # ==================== STOCHASTIC (fidèle) ====================
        if i >= stoch_window - 1:
            hh = np.max(high[i - stoch_window + 1:i + 1])
            ll = np.min(low[i - stoch_window + 1:i + 1])
            k_fast[i] = 100.0 * (close[i] - ll) / (hh - ll + 1e-12)

            # K Slow
            start = max(0, i - stoch_slow + 1)
            sum_k = 0.0
            count = 0
            for j in range(start, i + 1):
                sum_k += k_fast[j]
                count += 1
            k_slow[i] = sum_k / count if count > 0 else 50.0

            # %D (Signal)
            start_d = max(0, i - stoch_signal + 1)
            sum_d = 0.0
            count_d = 0
            for j in range(start_d, i + 1):
                sum_d += k_slow[j]
                count_d += 1
            d_line[i] = sum_d / count_d if count_d > 0 else k_slow[i]

            # Logique avancée
            k = k_slow[i]
            d = d_line[i]
            if k > 80:
                stoch[i] = -1
            elif k < 20:
                stoch[i] = 1
            elif k > d + 3:
                stoch[i] = 1
            elif k < d - 3:
                stoch[i] = -1
            else:
                stoch[i] = 0

        # ==================== SAR COMPLET ====================
        if i == 0:
            psar[i] = low[0]
            is_long[i] = True
            af[i] = sar_step
        else:
            if is_long[i - 1]:
                psar[i] = psar[i - 1] + af[i - 1] * (high[i - 1] - psar[i - 1])
                psar[i] = min(psar[i], min(low[i - 1], low[i - 2] if i >= 2 else low[i - 1]))
            else:
                psar[i] = psar[i - 1] - af[i - 1] * (psar[i - 1] - low[i - 1])
                psar[i] = max(psar[i], max(high[i - 1], high[i - 2] if i >= 2 else high[i - 1]))

            # Inversion
            if is_long[i - 1] and psar[i] > low[i]:
                is_long[i] = False
                psar[i] = high[i]
                af[i] = sar_step
            elif not is_long[i - 1] and psar[i] < high[i]:
                is_long[i] = True
                psar[i] = low[i]
                af[i] = sar_step
            else:
                is_long[i] = is_long[i - 1]
                # Augmentation AF
                if (is_long[i] and high[i] > high[i - 1]) or (not is_long[i] and low[i] < low[i - 1]):
                    af[i] = min(sar_max, af[i - 1] + sar_step)
                else:
                    af[i] = af[i - 1]

            psar_signal[i] = 1 if psar[i] < close[i] else (-1 if psar[i] > close[i] else 0)

        # ==================== EFFICIENCY RATIO ====================
        if i >= er_window - 1:
            change = abs(close[i] - close[i - er_window + 1])
            noise = 0.0
            for j in range(i - er_window + 1, i):
                noise += abs(close[j + 1] - close[j])
            er[i] = change / (noise + 1e-12) if noise > 0 else 0.0

        # ==================== VWAP CUMULATIF + Z-SCORE ====================
        tp = (high[i] + low[i] + close[i]) / 3.0
        tp_sum[i] = tp_sum[i - 1] + tp if i > 0 else tp
        vol_sum[i] = vol_sum[i - 1] + 1.0 if i > 0 else 1.0

        if i >= vwap_window - 1:
            vwap = tp_sum[i] / vol_sum[i]
            start = i - vwap_window + 1
            slice_std = np.std(close[start:i + 1])
            vwap_z[i] = (close[i] - vwap) / (slice_std + 1e-8)

    return (bb_mavg, bb_hband, bb_lband, direction, sigo, sigc,
            stoch, er, vwap_z, psar_signal)
