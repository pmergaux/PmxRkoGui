import time


import math
from datetime import datetime

import numpy as np
import pandas as pd
from mt5linux import MetaTrader5

from decision.candle_decision import signal_col, calculate_vwap_zscore, calculate_indicators, choix_features, \
    calculate_vwap_score, calculate_japonais, calculate_stochastic, calculate_sar, calculate_efficiency_ratio, \
    is_market_exploding, calculate_atr_4sl, is_market_safe, compute_decision_indicators
import utils.config_utils
from train.pipeline_manager import prepare_jepa_data

from train.prediction import prediction, tabicl_predict
from utils.model_utils import config_to_features, prepare_target_column, create_sequences_numba, kr_servers
from utils.scaler_utils import load_and_transform
from utils.utils import sens_lib, NONE, BUY, SELL, CLOSE, FCLOSE, \
    safe_format, get_clean_timestamp, VERT, RESET, ROUGE, BLEU

from collections import deque

class DynamicThresholdManager:
    def __init__(self, window_size=20, low_percentile=15, high_percentile=85):
        """
        window_size : Nombre de briques prises en compte pour calculer la distribution.
        low_percentile : Seuil de survente dynamique (ex: 15% des plus bas scores).
        high_percentile : Seuil de surachat dynamique (ex: 15% des plus hauts scores).
        """
        self.window_size = window_size
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.history = deque(maxlen=window_size)

    def update(self, current_pred):
        # Ajout de la dernière prédiction (après sigmoïde) dans l'historique
        self.history.append(current_pred)

    def get_thresholds(self):
        # Si on n'a pas assez de données, on renvoie des seuils neutres par défaut
        if len(self.history) < 20:
            return 0.5, 0.5

        # Calcul des percentiles sur la fenêtre glissante
        arr = np.array(self.history)
        low_thresh = np.percentile(arr, self.low_percentile)
        high_thresh = np.percentile(arr, self.high_percentile)

        return low_thresh, high_thresh

    def get_trading_signal(self, current_pred):
        low_thresh, high_thresh = self.get_thresholds()

        # Analyse par rapport aux seuils dynamiques du moment
        if current_pred >= high_thresh:
            return "HAUT DE CANAL (Attention surachat / Consolidation haute)"
        elif current_pred <= low_thresh:
            return "BAS DE CANAL (Attention survente / Rebond potentiel)"
        else:
            return "ZONE NEUTRE (Tendance en cours ou bruit)"

class IndicatorMonitor:
    def __init__(self, window_size=12, min_samples=4):
        self.history = deque(maxlen=window_size)
        self.min_samples = min_samples

    def update(self, new_pred):
        # 1. Conversion propre si c'est un scalaire (Python ou NumPy, y compris shape ())
        if isinstance(new_pred, (np.floating, float, int, np.integer)):
            self.history.append(float(new_pred))
            return
        # 2. Si c'est un tableau NumPy ou une liste, on gère par la taille/aplatissement
        try:
            arr = np.asarray(new_pred)
        except (ValueError, TypeError):
            # Si la conversion échoue, on tente une itération directe de secours
            for pred in new_pred:
                self.history.append(float(pred))
            return
        # Si c'est un scalaire emballé dans un tableau de taille 1 (shape () ou (1,))
        if arr.size == 1:
            self.history.append(float(arr.item()))
        else:
            # Si c'est un tableau/liste avec plusieurs éléments, on parcourt le tout à plat
            for pred in arr.flat:
                self.history.append(float(pred))
    @property
    def is_ready(self):
        """Vérifie si nous avons assez de données pour décider."""
        return len(self.history) >= self.min_samples

    def get_partial(self, num):
        return np.array(self.history)[-num:]

    def get_current_z(self):
        if not self.is_ready:
            return 0, 0
        arr = np.array(self.history)
        moy = np.mean(arr)
        try:
            std = np.std(arr)
            if std == 0: return 0, moy
            return (self.history[-1] - moy) / std, moy
        except Exception as e:
            print(f"std error : {e}")
            return 0, moy

    def get_thresholds(self, open_level=1.5, close_level=0.0):
        if not self.is_ready:
            return 0, 0, 0, 0
        Z, _ = self.get_current_z()
        if Z == 0:
            return 0, 0, 0, 0
        # Convention :
        # Le premier seuil (vendeur) doit être "haut" (Z + level)
        # Le deuxième seuil (acheteur) doit être "bas" (Z - level)
        # devrait être l'inverse à mon avis
        o_upper = Z + open_level  # Signal de vente (surachat)
        o_lower = Z - open_level  # Signal d'achat (survente)

        c_upper = Z + close_level
        c_lower = Z - close_level

        return o_upper, o_lower, c_upper, c_lower

def trading_decision(pos: int, po: float, pc: float, dd, dj, proba,
                     sl: float, tp: float, buy_thr=0.6, sell_thr=0.4, close_buy=0.47, close_sell=0.53,
                     pente=0, r2=0.5, er=0.5, extra=None, trace=False, config={}, date=None):
    if extra is None:
        extra = []
    if dd is None and proba is None:
        return NONE
    veto_config = config.get('veto', {})
    # Configuration des vetoes avec valeurs par défaut
    if veto_config is None:
        veto_config = {}
    r2_min = veto_config.get('r2_min', 0.3)
    r2_bb = veto_config.get('ré_bb', 0.6)
    er_min = veto_config.get('er_min', 0.38)
    pente_contrarienne_min = veto_config.get('pente_contrarienne_min', 0.15)
    er_pente_veto_trigger = veto_config.get('er_pente_veto_trigger', 0.45)
    trending_hard_thr = veto_config.get('is_trending_hard', 0.85)

    is_trending_hard = (r2 >= trending_hard_thr)
    # === RULES ===
    # ssc est la direction si close_sens est True
    ssd, ssc, sso = dd['direction'].iloc[-2], dd['sigc'].iloc[-2], dd['sigo'].iloc[-2]
    psd, psc, pso = dd['direction'].iloc[-3], dd['sigc'].iloc[-3], dd['sigo'].iloc[-3]
    # === BACKTEST === open enabled


    cs = -1 if proba < close_buy else 1 if proba > close_sell else 0
    cp = proba
    co = 1 if proba > buy_thr else -1 if proba < sell_thr else 0
    co_pure = co
    if is_trending_hard:
        # on anihile les IA
        if cs * pente < 0:
            co = NONE
            cs = NONE
    #cb = b_signal[-1]
    sigOpen = sigClose = NONE
    # 5. Sécurité TP/SL (Priorité absolue) FCLOSE=6
    if pos != 0:
        if (pc - po) * pos > tp if tp != 0 else False: sigClose = FCLOSE
        if (po - pc) * pos > sl if sl != 0 else False: sigClose = FCLOSE
    sigT = 0
    recs = 0
    """
    # abandon des indicateurs sur renko ?
    for col in signal_col:
        if col in dd.columns:
            sigT += int(dd.iloc[-2][col])
            recs += 1
    """
    sigT = sso
    recs = 1
    sigT = co + sigT
    recs += 2
    sigS = sigT
    # On suppose que extra contient le résultat de 'get_final_...'
    bbc = NONE
    if isinstance(extra, dict):
        for key, cx in extra.items():
            if key == 'bbc':
                bbc = cx
                continue
            sigT += cx[0]
            sigS += cx[1]
            if key == 'final':
                recs += 2
            else:
                recs += 1
    elif isinstance(extra, (int, float)):
        sigT += extra
        sigS += extra
        recs += 1
    recs = max(math.trunc(recs/2)-1 , 2)
    market_safe = True
    if dj is not None:
        market_safe = is_market_safe(dj, pos=len(dj) - 1, cfg=config)
    # 1. On vérifie si les indicateurs techniques sont neutres
    tech_neutral = (extra.get('sar')[0] == 0 and extra.get('stoch')[0] == 0)
    tech_opposit = (extra.get('sar')[1] != extra.get('stoch')[1])

    # 2. Détermination de sigOpen avec condition de fermeté
    # 3. Déterminer le signal d'ouverture théorique (le vote)
    temp_sigOpen = BUY if sigT >= recs else SELL if sigT <= -recs else NONE
    save_sigOpen = BUY if sigS >= recs else SELL if sigS <= -recs else NONE
    # 4. FILTRE DE RIGUEUR : Si tech_neutral, on exige un signal co (RNN franc)
    # Cela s'applique qu'on ait une position ou non.
    if tech_neutral and co == NONE:
        sigOpen = NONE
        if r2 < r2_min and er < er_min:
            save_sigOpen = NONE
    else:
        sigOpen = temp_sigOpen
        # 5. FILTRE R2 & ER & PENTE (Marge de sécurité statistique)
        # A. Veto global R2 & ER : Interdire toute ouverture si le marché est à la fois non directionnel (R2 bas) ET congestif (ER bas)
        if not market_safe or (r2 < r2_min and er < er_min) or sigOpen * co_pure < 0:
            sigOpen = NONE
            save_sigOpen = NONE
        else:
            if sigOpen * cs < 0:
                sigOpen = NONE
            # B. Veto Pente contraire : Ignorer les micro-pentes sauf si le marché est très directionnel
            # C. Tendance Forte (trending hard) : Veto absolu en cas de pente contraire sur R2 élevé
            elif sigOpen * pente < 0 and abs(pente) > pente_contrarienne_min or er >= er_pente_veto_trigger or is_trending_hard:
                sigOpen = NONE
            # D. Signal inverse de clôture (bbc) inutile si pente est de notre côté et r2 fort
            elif sigOpen * bbc < 0 and (r2 < r2_bb or sigOpen * pente < 0):
                sigOpen = NONE
            # 3. Amplification pour l'inversion (votre logique du 4*sigOpen)
            elif sigOpen * (1 if cp > 0.5 else -1) > 0 and abs(sigT) > recs:
                sigOpen = 4 * sigOpen
            if save_sigOpen != NONE and save_sigOpen * (1 if cp > 0.5 else -1) > 0 and abs(sigS) > recs:
                save_sigOpen = save_sigOpen * 4
    # ================================== cloture d'une position de sens pos
    if r2 > r2_min or er > er_min:
        if pos != 0 and sigClose == NONE and (sigT * pos < 0 or co_pure * pos < 0 or (
                bbc * pos < 0 and tech_opposit) or (
                                                      bbc * pos < 0 and extra.get('sar')[1] * pos < 0 and not tech_opposit) or
                                              (pos * cs < 0)):
            if is_trending_hard:
                # Si le R2 est énorme, on ignore le signal de fermeture mineur
                # on attend une vraie cassure ou le TP/SL
                pass
            else:
                sigClose = CLOSE
    # rappel CLOSE = 5 FCLOSE = 6 NCLOSE = 7
    if trace:
        indic = []
        for col in signal_col:
            if col in dd.columns:
                indic.append(col)
                indic.append(int(dd.iloc[-2][col]))
        print(f"{get_clean_timestamp() if date is None else date} sig={sens_lib[int(sigClose)]}#{sens_lib[int(sigOpen)]}({save_sigOpen:.0f}) "
              f"{'MS' if not market_safe else ''} prb={safe_format(proba[-1],'.3f')} cs={cs:.0f} co={co:.0f}({co_pure:.0f}) "
              f"sc={ssc} so={sso}={indic} pnte {pente:.2f} R2 {r2:.2f}/({r2_min}+{r2_bb}) ER {er:.2f}/{er_min} "
              f"ext={extra} ={sigT:.0f}({sigS:.0f})/{recs:.1f}")
    return sigClose, sigOpen, co_pure, co


def monitoring( proba, monitor_indic: IndicatorMonitor, monitor_means: IndicatorMonitor	):
    """
    # version toutes proba rythme lent 5 min
    try:
        self.monitor_indic.update(proba)
    except Exception as e:
        print(f"err update Z {type(proba)} {proba[-1]} {e}")
        raise e
    proba = proba[-1]
    """
    # version juste la dernière proba si rapide
    # print(f"type pr {type(proba)}")
    if isinstance(proba, np.ndarray):
        val = proba[-1]
    else:
        val = proba
    # Extraction sécurisée du scalaire brut Python
    if hasattr(val, "item"):
        val_clean = float(val.item())
    else:
        val_clean = float(val)
    # print(f"update 2 {type(val_clean)} {val_clean}")
    try:
        monitor_indic.update(val_clean)
    except Exception as e:
        print(f"err update 2 {type(val_clean)} {val_clean} {e}")
        raise e
    z_indic, moy = monitor_indic.get_current_z()
    monitor_means.update(moy)
    z_means, _ = monitor_means.get_current_z()
    return val_clean, z_indic, z_means, moy

# =========================================================================
#             décision version origine
# =========================================================================
def get_open_decision(situation):
    sigOpen = 0
    if utils.config_utils.VSIMPLE:
        if situation == 2: sigOpen = 1
        if situation == -2: sigOpen = -1
    else:
        if situation > 0: sigOpen = 1
        if situation < 0: sigOpen = -1
    return sigOpen

def get_close_decision(current_position, current_price, entry_price,
                          sl_dist, tp_dist, situation, sigOpen, trace=False):
    """
    prediction_prob: 0 à 1 (ex: 0.7 = forte proba de hausse)
    buy_thr: ex 0.55 (seuil d'entrée long)
    close_buy_thr: ex 0.50 (seuil de sortie long)
    """
    sigClose = 0
    # --- 1. GESTION DES POSITIONS OUVERTES ---
    if current_position == 1:  # On est LONG
        # B. Sortie sur SL/TP dur
        if (sl_dist != 0 and current_price <= (entry_price - sl_dist)) or (tp_dist != 0 and current_price >= (entry_price + tp_dist)):
            sigClose = FCLOSE
        # A. Sortie sur Probabilité (Si la proba tombe sous le seuil de maintien)
        elif situation == -1 or sigOpen == -1:
            sigClose = CLOSE

    if current_position == -1:  # On est SHORT
        # B. Sortie sur SL/TP dur
        if (sl_dist != 0 and current_price >= (entry_price + sl_dist)) or (tp_dist != 0 and current_price <= (entry_price - tp_dist)):
            sigClose = FCLOSE
        # A. Sortie sur Probabilité (Si la proba remonte au-dessus du seuil de maintien)
        # Note : pour le short, prediction_prob est proche de 0 (ex: 0.45 pour sortir)
        elif situation == 1 or sigOpen == 1:
            sigClose = CLOSE

    if trace:
        COLOR_c = BLEU if sigClose > 3 or sigClose == 0 else VERT if sigClose > 0 else ROUGE
        COLOR_o = VERT if sigOpen > 0 else ROUGE if sigOpen < 0 else BLEU
        print(
            f"{get_clean_timestamp()} sign={COLOR_c}{sens_lib[int(sigClose)]}{RESET}#{COLOR_o}{sens_lib[int(sigOpen)]}{RESET} "
            f"situ = {situation}")
    return sigClose

SA = 0
A = 1
N = 2
V = 3
SV = 4

def calcul_situation(monitor_valeurs: IndicatorMonitor, recent_bricks ,bornes, dest, trace):
    # Entrées valeurs croissantes de probabilités ou non
    all_val = monitor_valeurs.get_partial(3)
    #z_indic, moy = monitor_valeurs.get_current_z()
    #print(f"situation Z {z_indic:.3f} {moy:.2f}")
    fst_valeur, two_valeur, lst_valeur = all_val
    local_situation = dest[SV]		# hors bornes
    # attention bornes de 0 à 3, dest de 0 à 4
    for i in range(len(bornes)):
        if lst_valeur < bornes[i]:
            local_situation = dest[i]
            if trace and not utils.config_utils.VTOTALE:
                print(f"proba {lst_valeur:.4f}, at {i} borne {bornes[i]}, situ {local_situation}")
            break
    if not utils.config_utils.VTOTALE:
        if trace and local_situation == dest[SV]:
            print(f"proba {lst_valeur:.4f}, at borne {bornes[-1]}, situ {local_situation}")
        return local_situation
    # Conversion rapide uniquement sur ces quelques lignes
    # times_ms = recent_bricks['time'].to_numpy().astype('int64') / 1_000_000
    recent_bricks['direction'] = np.where(recent_bricks['close_renko'] > recent_bricks['open_renko'], 1, -1)
    directions = recent_bricks['direction'].to_numpy()
    direction = directions[-2] if directions[-3] == directions[-2] else 0
    if local_situation == dest[N]:			# position neutre théorique
        if utils.config_utils.VSIMPLE:
            return local_situation
        borneL = bornes[1]
        borneH = bornes[2]
        is_tot_values = (all(all_val >= borneL) & all(all_val <= borneH))
        is_lst_value = True
        is_fst_value = (borneL <= fst_valeur <= borneH)
        is_low = all_val[-2] < borneL
        is_upp = all_val[-2] > borneH
        is_proba_dn = all(np.diff(all_val) > 0)
        is_proba_up = all(np.diff(all_val) < 0)
        local_situation = dest[A] if is_low and is_proba_up else dest[V] if is_upp and is_proba_dn else dest[N]
        if trace:
            print(f"rand {N} val {all_val} borne {borneL} situ {local_situation} {borneH} fst {is_fst_value} low {is_low} high {is_upp} sens {direction}")

    elif local_situation == dest[A]:  # fermeture achat --> vente théoriquement
        if utils.config_utils.VSIMPLE:
            return local_situation
        borne = bornes[1]
        is_tot_values = all(all_val < borne)
        is_lst_value = True
        is_fst_value = (fst_valeur < borne)
        is_two_value = (two_valeur < borne)
        is_proba_dn = all(np.diff(all_val) > 0)
        is_proba_up = all(np.diff(all_val) < 0)
        local_situation = dest[V] if is_proba_up and (direction * dest[V] >= 0) \
            else dest[N] if (is_fst_value != is_two_value) or direction >= 0 \
            else dest[A]
        if trace:
            print(f"rang {A} val {all_val} situ {local_situation} borne {borne} fst {is_fst_value} two {is_two_value} sens {direction}")

    elif local_situation == dest[V]:  # fermeture vente --> achat théoriquement
        borne = bornes[2]
        is_tot_values = all(all_val > borne)
        is_lst_value = True
        is_fst_value = (fst_valeur > borne)
        is_two_value = (two_valeur > borne)
        is_proba_dn = all(np.diff(all_val) > 0)
        is_proba_up = all(np.diff(all_val) < 0)
        local_situation = dest[A] if is_proba_dn and (direction * dest[A] >= 0) \
            else dest[N] if (is_fst_value != is_two_value) or direction <= 0 \
            else dest[V]
        if trace:
            print(f"rang {V} val {all_val} situ {local_situation} borne {borne} fst {is_fst_value} two {is_two_value} sens {direction}")

    elif local_situation == dest[SA]:  # surachat -> vente théoriquement
        borne = bornes[0]
        is_tot_values = all(all_val <= borne)
        is_lst_value = True
        is_fst_value = (fst_valeur < borne)
        is_two_value = (two_valeur < borne)
        is_proba_up = all(np.diff(all_val) > 0)
        is_proba_dn = all(np.diff(all_val) < 0)
        if utils.config_utils.VSIMPLE:
            local_situation = dest[SV] if is_proba_dn and direction * dest[SV] >= 0 and not is_two_value \
                else dest[SA]
            return local_situation
        local_situation = dest[SV] if is_proba_dn and direction * dest[SV] >= 0 and not is_two_value \
            else dest[N] if (not is_fst_value and not is_two_value) or (direction * dest[SA] < 0) \
            else dest[V] if (is_fst_value != is_two_value) and (direction * dest[V] >= 0) \
            else dest[SA] if direction * dest[SA] >= 0 \
            else dest[N]
        # if is_tot_values: local_situation = dest[-1] if is_proba_up else dest[0] if is_proba_dn else 0
        if trace:
            print(f"rang {SA} val {all_val} situ {local_situation} borne {borne} fst {is_fst_value} two {is_two_value} sens {direction}")

    elif local_situation == dest[SV]:  # survente -> achat théoriquement
        borne = bornes[-1]
        is_tot_values = all(all_val >= borne)
        is_lst_value = True
        is_fst_value = (fst_valeur > borne)
        is_two_value = (two_valeur > borne)
        is_proba_up = all(np.diff(all_val) > 0)
        is_proba_dn = all(np.diff(all_val) < 0)
        if utils.config_utils.VSIMPLE:
            local_situation = dest[SA] if is_proba_up and direction * dest[SA] >= 0 and not is_two_value \
                else dest[SV]
            return local_situation
        local_situation = dest[SA] if is_proba_up and direction * dest[SA] >= 0 and not is_two_value \
            else dest[N] if (not is_two_value and not is_fst_value) or (direction * dest[SV] < 0) \
            else dest[A] if (is_two_value != is_fst_value) and (direction * dest[A] >= 0) \
            else dest[SV] if direction * dest[SV] >= 0 \
            else dest[N]
        # if is_tot_values: local_situation = dest[0] if is_proba_dn else dest[-1] if is_proba_up else 0
        if trace:
            print(f"rang {SV} val {all_val} situ {local_situation} borne {bornes[-1]} fst {is_fst_value} two {is_two_value} sens {direction}")

    return local_situation


def calcul_bornes(regression, param):
    if not regression:
        bornes = [param.get('threshold_sell', 0.25), param.get('close_buy', 0.4), param.get('close_sell', 0.6), param.get('threshold_buy', 0.75)]
    else:
        ol_r = param.get('open_level_rnn', 1.5)
        ol_t = param.get('open_level_tabicl', 1.5)
        cl_r = param.get('close_level_rnn', 0.1)
        cl_t = param.get('close_level_tabicl', 0.1)
        b_u = proba_final(ol_r, ol_t, 0.7)
        b_l = proba_final(cl_r, cl_t, 0.7)
        bornes = [-b_u, -b_l, b_l, b_u]
    return bornes

def proba_final(probas, weights=None):
    """
    Fusionne les probabilités avec alignement, pondération et sécurités anti-crash.
    """
    if not probas:  # Vérifie si le dict est vide ou None
        raise Exception("Le dictionnaire probas est vide ou None.")

    # Filtrer pour ne garder que les entrées valides (qui ne sont pas None)
    clean_probas = {}
    for k, v in probas.items():
        if v is not None:
            # S'assure qu'on manipule bien un tableau NumPy
            arr = np.array(v)
            if arr.size > 0:
                clean_probas[k] = arr

    if not clean_probas:
        return None

    nb = len(clean_probas)

    # Recherche de la longueur minimale pour l'alignement
    min_len = min(len(arr) for arr in clean_probas.values())

    # Nettoyage et alignement sur la fin des vecteurs
    new_probas = {}
    for key, arr in clean_probas.items():
        p_tab = arr[-min_len:]
        p_tab = np.nan_to_num(p_tab, nan=0.5)
        new_probas[key] = p_tab

    values = list(new_probas.values())

    # Calcul avec ou sans poids
    if weights is None:
        return np.mean(values, axis=0)
    else:
        weighted_sum = 0
        total_weight = 0
        for model_name, score in new_probas.items():
            weight = weights.get(model_name, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        # 3. Score moyen
        avg_score = weighted_sum / total_weight if total_weight != 0 else 0
        return avg_score

# =========================================================================
#           Etablissement des probabilites des modules
#=========================================================================

def stable_sigmoid(x):
    x = np.asarray(x)
    # x correspond ici à -proba * 100
    pos_mask = x >= 0
    # Pour x >= 0, on utilise la formule standard
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    # Pour x < 0, on utilise une forme mathématiquement équivalente anti-overflow
    exp_x = np.exp(x[~pos_mask])
    z[~pos_mask] = exp_x / (1 + exp_x)
    return z

def decision_ai(dfo, bricks4jepa, cfg, scaler, models, features_cols=None, test=False, trace=False):
    import torch
    features_cols_extracted, target_cols, total_cols = config_to_features(cfg)
    if features_cols is None:
        features_cols = features_cols_extracted
    if "renko_volatility_ratio" in dfo.columns and "renko_volatility_ratio" not in features_cols:
        features_cols.append("renko_volatility_ratio")
    # print(f"features_cols {features_cols} target_cols {target_cols} total_cols {total_cols} features_cols_extracted {features_cols_extracted} ")
    probas = {}
    try:
        if not isinstance(target_cols, list):
            target_cols = [target_cols]
        target_type = cfg['target']['target_type']
        try:
            dfi = prepare_target_column(dfo, target_cols[0], target_type).reset_index(drop=True)
        except BaseException as e:
            print("config err target", e)
            return None

        if 'target' in dfi.columns:
            target_cols = ['target']

        tabicl_window = cfg.get('parameters', {}).get('tabicl_window', 64)
        #print(f"DEBUG IN-MEMORY -> keys model: {list(models.keys())}")
        versions = cfg.get('live', {}).get('version', [])
        for key in versions:
            model = models[key]
            key_lower = key.lower()
            #print(f"DEBUG IN-MEMORY -> Type de model: {type(model)} | key: {key_lower}")
            try:
                # --- 1. Modèles RNN (GRU, LSTM) ---
                if key_lower in kr_servers:
                    df = dfi.ffill().dropna()
                    X_test = load_and_transform(scaler, df[features_cols])
                    y_test = df[target_cols].to_numpy(dtype=np.float32)
                    test_r = np.hstack([X_test, y_test])
                    seq_len = cfg.get(key_lower, {}).get(f'{key_lower}_seq_len', 24)
                    X_test_seq, _ = create_sequences_numba(test_r, seq_len, len(features_cols))
                    proba = prediction(model, X_test, X_test_seq, [key])
                # --- 2. Modèles JEPA (Inférence sur les N dernières briques) ---
                elif key_lower == 'jepa':
                    # print(f"DEBUG IN-MEMORY -> Type de model: {type(model)} | Contenu: {model}")
                    model, stats = model
                    model.eval()
                    # Extraction des N dernières briques nécessaires
                    jepa_seq_len = cfg.get("jepa", {}).get("SEQ_LEN", 64)
                    df_seq = bricks4jepa.tail(jepa_seq_len).copy().reset_index(drop=True)
                    if len(df_seq) < jepa_seq_len:
                        raise ValueError(
                            f"Pas assez de briques pour le JEPA (requis: {jepa_seq_len}, dispo: {len(df_seq)})")
                    # --- Préparation des 5 features de base ---
                    brick_size = cfg.get("parameters", {}).get("renko_size", 0)
                    features, raw_closes, _ = prepare_jepa_data(df_seq, brick_size, stats)
                    rel_price = ((raw_closes - raw_closes[0]) / brick_size)[:, np.newaxis]
                    # Assemblage final [Seq_len, 6] -> Tenseur [1, Seq_len, 6]
                    X_final = np.concatenate([features, rel_price], axis=-1)
                    X_tensor = torch.tensor(X_final, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        pred_brute, _ = model(X_tensor)
                        # On module la prédiction brute par le facteur de risque dynamique
                        proba = pred_brute.item()
                elif key_lower == 'tabfin':
                    # start = time.time()
                    finjepa_predictor = models.get('FINJEPA', None)
                    if test:
                        # 🚀 PARTIE GLOBALE : Exécutée UNE SEULE FOIS au tout début du test
                        if not hasattr(model, '_test_predictions_cached') or model._test_predictions_cached is None:
                            X_test_base = load_and_transform(scaler, dfi[features_cols])

                            # Extraction globale des embeddings Fin-JEPA
                            if finjepa_predictor is not None:
                                finjepa_predictor.model.eval()
                                with torch.no_grad():
                                    full_tensor = torch.tensor(X_test_base, dtype=torch.float32).to(
                                        finjepa_predictor.device)
                                    latent_embeddings = finjepa_predictor.model.context_encoder(
                                        full_tensor).cpu().numpy()
                                X_test_augmented = np.hstack([X_test_base, latent_embeddings])
                            else:
                                X_test_augmented = X_test_base

                            try:
                                # 1. On "fit" le TabICL une seule fois avec un historique initial
                                train_size = min(64, len(X_test_augmented) - 1)
                                X_train_init = X_test_augmented[:train_size]
                                y_train_init = dfi[target_cols].values.ravel()[:train_size]

                                model.fit(X_train_init, y_train_init)

                                # 2. On pré-calcule toutes les prédictions d'un bloc
                                model._test_predictions_cached = model.predict(X_test_augmented)
                            except Exception as e:
                                print(f"err prediction model TABFIN : {e}")
                                model._test_predictions_cached = None

                            model._test_counter = 0

                        # ⚡ PARTIE INCRÉMENTALE : Exécutée à chaque brique (Lecture instantanée à 0.00s)
                        if hasattr(model, '_test_predictions_cached') and model._test_predictions_cached is not None:
                            idx = min(model._test_counter, len(model._test_predictions_cached) - 1)
                            proba = model._test_predictions_cached[idx]
                            model._test_counter += 1
                        else:
                            # Fallback de secours si le cache a échoué
                            df_query = dfi.iloc[[-1]]
                            X_query = load_and_transform(scaler, df_query[features_cols])
                            proba = model.predict(X_query)
                    else:
                        # 1. On prend uniquement la fenêtre locale de contexte (ex: 64 briques) + la requête courante
                        window_size = tabicl_window + 1
                        df_window = dfi.tail(window_size) if len(dfi) >= window_size else dfi
                        df_context = df_window.iloc[:-1]
                        df_query = df_window.iloc[[-1]]
                        X_train_base = load_and_transform(scaler, df_context[features_cols])
                        y_train = df_context[target_cols].values.ravel()
                        X_query_base = load_and_transform(scaler, df_query[features_cols])
                        # 2. Extraction des embeddings Fin-JEPA uniquement sur cette mini-fenêtre (Instantané < 5ms)
                        if finjepa_predictor is not None:
                            finjepa_predictor.model.eval()
                            with torch.no_grad():
                                window_tensor = torch.tensor(
                                    load_and_transform(scaler, df_window[features_cols]),
                                    dtype=torch.float32
                                ).to(finjepa_predictor.device)
                                latent_window = finjepa_predictor.model.context_encoder(window_tensor).cpu().numpy()
                            latent_context = latent_window[:-1]
                            latent_query = latent_window[[-1]]
                            X_train = np.hstack([X_train_base, latent_context])
                            X_query = np.hstack([X_query_base, latent_query])
                        else:
                            X_train = X_train_base
                            X_query = X_query_base
                        # 3. Fit et Predict rapide sur la mini-fenêtre (exactement comme le modèle 'tab')
                        model.fit(X_train, y_train)
                        proba = model.predict(X_query)
                # print(f"Prediction time {key_lower}: {time.time() - start:.3f} seconds")
                elif key_lower == 'finjepa':
                    # Si 'model' est le FinJepaPredictor ou directement son modèle PyTorch
                    predictor = model
                    if hasattr(predictor, 'model'):
                        fin_net = predictor.model
                    else:
                        fin_net = predictor
                    fin_net.eval()
                    # On récupère le contexte nécessaire pour la Fin-JEPA (ex: context_len de la config)
                    fin_context_len = cfg.get("finjepa", {}).get("context_len", 60)
                    # On extrait les dernières features normalisées pour alimenter le réseau
                    df_seq = dfi.tail(fin_context_len).copy()
                    if len(df_seq) < fin_context_len:
                        # Padding ou gestion si pas assez de barres
                        X_seq_data = load_and_transform(scaler, dfi[features_cols])
                    else:
                        X_seq_data = load_and_transform(scaler, df_seq[features_cols])

                    X_tensor = torch.tensor(X_seq_data, dtype=torch.float32).unsqueeze(0)  # [1, Seq, Features]
                    if hasattr(predictor, 'device'):
                        X_tensor = X_tensor.to(predictor.device)

                    with torch.no_grad():
                        # Extraction de l'embedding ou prédiction directe selon votre architecture Fin-JEPA
                        latent_out = fin_net.context_encoder(X_tensor)
                        # Si votre Fin-JEPA possède une tête de prédiction finale pour la décision :
                        # (A adapter selon la structure exacte de votre classe FinJepaPredictor)
                        if hasattr(fin_net, 'predictor_head'):
                            pred_brute = fin_net.predictor_head(latent_out)
                        else:
                            # Fallback : on prend la moyenne des embeddings ou une projection simple
                            pred_brute = latent_out.mean()
                        proba = pred_brute.item() if hasattr(pred_brute, 'item') else float(pred_brute)
                elif key_lower == 'tab' or key_lower == 'tabff':
                    # start = time.time()
                    if test:
                        # 🚀 PARTIE GLOBALE : Exécutée UNE SEULE FOIS au tout début du test
                        if not hasattr(model, '_test_predictions_cached') or model._test_predictions_cached is None:
                            X_test_base = load_and_transform(scaler, dfi[features_cols])

                            try:
                                # 1. IMPORTANT : On "fit" le TabICL une seule fois avec un historique initial
                                train_size = min(64, len(X_test_base) - 1)
                                X_train_init = X_test_base[:train_size]
                                y_train_init = dfi[target_cols].values.ravel()[:train_size]

                                model.fit(X_train_init, y_train_init)

                                # 2. On pré-calcule toutes les prédictions d'un bloc d'un coup
                                model._test_predictions_cached = model.predict(X_test_base)
                            except Exception as e:
                                print(f"err prediction model {key}: {e}")
                                model._test_predictions_cached = None

                            model._test_counter = 0

                        # ⚡ PARTIE INCRÉMENTALE : Lecture instantanée dans le cache aux briques suivantes (0.00s)
                        if hasattr(model, '_test_predictions_cached') and model._test_predictions_cached is not None:
                            idx = min(model._test_counter, len(model._test_predictions_cached) - 1)
                            proba = model._test_predictions_cached[idx]
                            model._test_counter += 1
                        else:
                            # Fallback de secours si le cache a échoué
                            df_query = dfi.iloc[[-1]]
                            X_query = load_and_transform(scaler, df_query[features_cols])
                            proba = model.predict(X_query)
                    else:
                        # Contexte historique de 64 briques pour alimenter le Transformer
                        df_context = dfi.iloc[:-1].tail(tabicl_window) if len(dfi) > 1 else dfi
                        X_train = load_and_transform(scaler, df_context[features_cols])
                        y_train = df_context[target_cols].values.ravel()
                        df_query = dfi.iloc[[-1]]
                        X_query = load_and_transform(scaler, df_query[features_cols])
                        # Le .fit() instantané charge juste le contexteerr sa
                        model.fit(X_train, y_train)
                        proba = model.predict(X_query)
                # print(f"Prediction time {key_lower}: {time.time() - start:.2f} seconds")
                # --- 3. TabICL Fine-Tuné (TABFF) : Déjà entraîné, on prédit directement ---
                elif key_lower == 'tabff':
                    # Si TABFF a besoin de son contexte ou d'une requête simple :
                    # On utilise uniquement predict() (pas de .fit() lourd ici !)
                    df_query = dfi.iloc[[-1]]
                    X_query = load_and_transform(scaler, df_query)
                    # Note: Selon la structure exacte de votre wrapper TABFF,
                    # si un contexte est requis, il faut l'appeler sans relancer l'entraînement.
                    proba = model.predict(X_query)
                # --- 3. Modèles Classiques et Fine-Tunés (TABFF, CAT, LGBM) ---
                else:
                    # Une seule ligne suffit puisque les features sont globales
                    df_tree = dfi.iloc[[-1]]
                    X_test = load_and_transform(scaler, df_tree)
                    proba = prediction(model, X_test, None, [key])

                # Normalisation et nettoyage du signal
                try:
                    if proba is not None:
                        if cfg.get('parameters', {}).get('window_monitor', 0) == 0:
                            #proba = 1 / (1 + np.exp(-proba * 100))
                            proba = stable_sigmoid(-proba * 100)
                        proba_arr = np.asarray(proba).ravel()
                        if len(proba_arr) > 1:
                            proba_arr = [proba_arr[-1]]
                        probas[key] = np.clip(np.asarray(proba_arr), 0.001, 0.999)
                    else:
                        print(f"err prediction model {key} : proba is None")
                except Exception as e:
                    # Remplacez votre simple print par ceci pour forcer l'affichage de la ligne exacte :
                    print(f"err sigmoid model {key} : {e} (Type de proba: {type(proba)}, Valeur: {proba})")
                    import traceback
                    traceback.print_exc()

            except BaseException as e:
                print(f"err prediction model {key} :", e)

    except BaseException as e:
        print(f"err prediction general {cfg['live']['version']} :", e)

    return probas

# =========================================================================
#                   Fonction pour lea version Non RNN
#==========================================================================
"""
def decision_ai_aligned(dfo, cfg, scaler, models, features_cols=None):
    # Exécute decision_ai tout en réalignant les prédictions
    #sur l'index exact du DataFrame d'origine `dfo`.

    n_total = len(dfo)

    # 1. On récupère les proba brutes via votre fonction existante
    proba_rnn_raw, proba_tabicl_raw = decision_ai(dfo, cfg, scaler, models, features_cols)

    # 2. Initialisation de tableaux de la taille exacte de dfo remplis de NaN
    proba_rnn_aligned = np.full(n_total, np.nan)
    proba_tabicl_aligned = np.full(n_total, np.nan)

    features_cols_extracted, target_cols, total_cols = config_to_features(cfg)
    if features_cols is None:
        features_cols = features_cols_extracted
    if "renko_volatility_ratio" in dfo.columns and "renko_volatility_ratio" not in features_cols:
        features_cols.append("renko_volatility_ratio")

    # On simule les étapes de filtrage pour retrouver exactement les indices valides
    df = dfo.copy()
    try:
        df = prepare_target_column(df, target_cols[0], cfg['target']['target_type']).reset_index(drop=True)
    except:
        pass
    df_cleaned = df.ffill().dropna()

    # Calcul des indices valides après dropna
    valid_indices_dropna = df_cleaned.index.values

    version = cfg.get('live', {}).get('version', ['LSTM'])
    if isinstance(version, list): version = version[0]
    seq_len = cfg.get('gru', {}).get('gru_seq_len', 24) if version == 'GRU' else cfg.get('lstm', {}).get('lstm_seq_len',
                                                                                                         24)

    # 3. Réalignement pour TabICL (qui subit juste le dropna)
    if proba_tabicl_raw is not None:
        # TabICL s'applique sur df_cleaned complet
        min_len_tab = min(len(proba_tabicl_raw), len(valid_indices_dropna))
        target_idx_tab = valid_indices_dropna[-min_len_tab:]
        proba_tabicl_aligned[target_idx_tab] = proba_tabicl_raw[-min_len_tab:]

    # 4. Réalignement pour le RNN (qui subit dropna ET le décalage de séquence seq_len)
    if proba_rnn_raw is not None:
        # Le séquençage Numba supprime les (seq_len - 1) premières lignes de df_cleaned
        valid_indices_rnn = valid_indices_dropna[seq_len - 1:]
        min_len_rnn = min(len(proba_rnn_raw), len(valid_indices_rnn))
        target_idx_rnn = valid_indices_rnn[-min_len_rnn:]
        proba_rnn_aligned[target_idx_rnn] = proba_rnn_raw[-min_len_rnn:]

    return proba_rnn_aligned, proba_tabicl_aligned
"""
def decision_monitor(monitor_rnn, proba_rnn,
                     monitor_tabicl, proba_tabicl,
                     upd=True):
    if proba_rnn is None or proba_tabicl is None or len(proba_rnn) == 0 or len(proba_tabicl) == 0:
        print(f"err fourniture proba")
        raise ValueError("prediction nulle")

    # 1. Mise à jour et calcul des Z-scores actuels
    if upd:
        monitor_rnn.update(proba_rnn)
        monitor_tabicl.update(proba_tabicl)

    z_rnn, _ = monitor_rnn.get_current_z()
    z_tab, _ = monitor_tabicl.get_current_z()

    return z_rnn, z_tab

def get_rnn_anti_monitor(current_position, current_price, entry_price,
                         sl_dist, tp_dist, situation, trace=False):

    # --- 1. GESTION DES POSITIONS OUVERTES ---
    if current_position == 1:  # On est LONG
        if situation <= 0: sigClose = CLOSE
        # A. Sortie sur Probabilité (Si la proba tombe sous le seuil de maintien)
        # B. Sortie sur SL/TP dur
        elif (sl_dist != 0 and current_price <= (entry_price - sl_dist)) or (tp_dist != 0 and current_price >= (entry_price + tp_dist)):
            sigClose = FCLOSE

    if current_position == -1:  # On est SHORT
        # A. Sortie sur Probabilité (Si la proba remonte au-dessus du seuil de maintien)
        # Note : pour le short, prediction_prob est proche de 0 (ex: 0.45 pour sortir)
        if situation >= 0: sigClose = CLOSE
        # B. Sortie sur SL/TP dur
        elif (sl_dist != 0 and current_price >= (entry_price + sl_dist)) or (tp_dist != 0 and current_price <= (entry_price - tp_dist)):
            sigClose = FCLOSE

    sigOpen = BUY if situation > 0 else SELL if situation < 0 else NONE

    if trace:
        COLOR_c = BLEU if sigClose > 3 or sigClose == 0 else VERT if sigClose > 0 else ROUGE
        COLOR_o = VERT if sigOpen > 0 else ROUGE if sigOpen < 0 else BLEU
        print(
            f"{get_clean_timestamp()} sign={COLOR_c}{sens_lib[int(sigClose)]}{RESET}#{COLOR_o}{sens_lib[int(sigOpen)]}{RESET} ")
    return sigClose, sigOpen
    """
    # OUVERTURE : Stabilité par confirmation double
    # On exige que les deux modèles soient dans la même zone (tous les deux très bas ou très hauts)
    if signal_strength < -threshold_open and z_rnn < 0 and z_tab < 0:
        action_o = SELL
    elif signal_strength > threshold_open and z_rnn > 0 and z_tab > 0:
        action_o = BUY
    # FERMETURE : Protection du capital
    # On ferme dès que l'un des deux modèles revient vers la moyenne
    if (z_rnn >= cl_r or z_tab >= cl_t):
        action_c = BUY
    elif (z_rnn <= -cl_r or z_tab <= -cl_t):
        action_c = SELL
    """

def decision_rates_optimized(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Version la plus fidèle possible (Option B)"""
    df = df.copy().reset_index(drop=True)

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)

    (bb_mavg, bb_hband, bb_lband, direction, sigo, sigc,
     stoch, er, vwap_z, psar_signal) = compute_decision_indicators(
        close, high, low, open_,
        bb_window=20,
        stoch_window=cfg.get('stoch', {}).get('window', 21),
        stoch_slow=cfg.get('stoch', {}).get('slow', 5),
        stoch_signal=cfg.get('stoch', {}).get('signal', 5),
        er_window=cfg.get('er', {}).get('window', 18),
        vwap_window=cfg.get('zscore', {}).get('window', 24),
        sar_step=cfg.get('sar', {}).get('step', 0.02),
        sar_max=cfg.get('sar', {}).get('max', 0.2),
    )

    df['bb_mavg'] = bb_mavg
    df['bb_hband'] = bb_hband
    df['bb_lband'] = bb_lband
    df['direction'] = direction
    df['sigo'] = sigo
    df['sigc'] = sigc
    df['stoch'] = stoch
    df['er'] = er
    df['vwap_z'] = vwap_z
    df['psar'] = psar_signal

    return df

def decision_rates(df, cfg):
    jp = calculate_japonais(df)
    jp = calculate_stochastic(jp, cfg.get('stoch').get('window', 21),  cfg.get('stoch').get('slow', 5), cfg.get('stoch').get('signal', 5))
    jp = calculate_sar(jp, cfg.get('sar').get('window', 0.02), cfg.get('sar').get('maxi', 0.2))
    jp = calculate_efficiency_ratio(jp, cfg.get('er').get('window', 18))
    jp = calculate_vwap_zscore(jp, cfg.get('zscore').get('window', 24))
    # Astuce : Multiplier ER par la pente permet de "calmer" les pentes bruitées
    # df['weighted_slope'] = df['slope'] * df['er']
    return jp

def decision_bricks(bricks, cfg):
    try:
        display = calculate_indicators(bricks, cfg)
        display = choix_features(display, cfg)
    except Exception as e:
        print(f"decision err {e}")
        raise e
    if not 'sigc' in display.columns:
        display['sigc'] = NONE
    if not 'sigo' in display.columns:
        display['sigo'] = NONE
    return display


def get_last_decision(df, vol_log_pct, r2=0.5, er_min = 0.4, trace=True):
    # La limite est la même pour tout le monde
    limit = 2.5 if r2 > 0.8 else 2.2  # Passé à 2.2 selon nos analyses CSV

    last_z = df['vwap_z'].iloc[-1]
    last_er = df['er'].iloc[-1]

    # Votes bruts
    sar_vote = df['psar'].iloc[-1]
    stoch_vote = df['stoch'].iloc[-1]
    if trace:
        print(f"CNTRL vol_pct {vol_log_pct:.2f} limit {limit:.2f} Z {last_z:.2f} R2 {r2:.3f}/0.8 ER {last_er:.3f}/{er_min} sar {sar_vote:.0f} stoch {stoch_vote:.0f}")

    # 1. Filtre ER Global : Si l'ER est trop bas, on n'écoute personne
    if last_er < er_min:
        return 0, sar_vote, 0, stoch_vote
    # --- FILTRES DE RÉGIME (S'appliquent aux deux ou à rien) ---
    # A. La Lessiveuse ou le Marché Plat : VETO TOTAL
    # On ne veut aucun signal si le marché est incohérent ou mort
    if (vol_log_pct > 0.80 and r2 < 0.50) or (vol_log_pct < 0.01):
        return 0, sar_vote, 0, stoch_vote

    # B. La Nervosité (Interrupteur Global)
    # Au lieu de couper juste le stochastique, on décide que si c'est nerveux (>0.20),
    # on n'accepte un signal QUE s'ils sont d'accord (Convergence).
    # Cela évite qu'un indicateur qui bascule seul ne force une décision.
    if vol_log_pct > 0.20 and r2 < 0.80:
        if sar_vote != stoch_vote:
            return 0, sar_vote, 0, stoch_vote  # On attend la synchronisation

    # 2. Filtre Z-Score Global (Surchauffe)
    # Si le prix sort des limites, on neutralise TOUS les votes de même sens
    sar = sar_vote
    stoch = stoch_vote
    if last_z > limit:
        if sar_vote > 0: sar = 0
        if stoch_vote > 0: stoch = 0
    elif last_z < -limit:
        if sar_vote < 0: sar = 0
        if stoch_vote < 0: stoch = 0

    return sar, sar_vote, stoch, stoch_vote

def get_global_decision(df):
    # Calcul des indicateurs
    df = calculate_vwap_zscore(df)
    # Filtre de blocage (Anti-Surchauffe)
    # On définit une limite de confiance (ex: 2.0 sigma)
    limit = 2.0
    # Signaux de base (1, -1, 0)
    sar_sign = df['psar']
    stoch_sign = np.where(df['er'] > 0.4, df['stoch'], 0)  # ER filtre déjà ici
    # LOGIQUE DE FILTRAGE PAR LE Z-SCORE :
    # On bloque l'ACHAT si le Z-Score est trop élevé (> 2.0)
    if df['vwap_z'].iloc[-1] > limit:
        sar_sign = np.where(sar_sign == 1, 0, sar_sign)
        stoch_sign = np.where(stoch_sign == 1, 0, stoch_sign)
    # On bloque la VENTE si le Z-Score est trop bas (< -2.0)
    if df['vwap_z'].iloc[-1] < -limit:
        sar_sign = np.where(sar_sign == -1, 0, sar_sign)
        stoch_sign = np.where(stoch_sign == -1, 0, stoch_sign)
    return sar_sign, stoch_sign

def get_final_vote(df):
    # 1. Signaux de base
    sar_sign = df['psar']
    stoch_sign = df['stoch']

    # 2. Filtre ER
    stoch_filtered = np.where(df['er'] > 0.4, stoch_sign, 0)

    # 3. VWAP Score & Seuil dynamique
    df = calculate_vwap_score(df, 24)
    # On calcule l'écart-type sur les 100 dernières valeurs de la distance relative
    std_dist = df['vwap'].rolling(window=100).std()

    # IMPORTANT : Récupérer la dernière valeur pour le calcul scalaire
    current_threshold = std_dist.iloc[-1] * 2.0
    current_vwap_dist = df['vwap'].iloc[-1]

    # 4. Filtrage
    # Si on veut ACHETER (1) mais distance > seuil (trop cher) -> 0
    # On applique le filtre sur la dernière ligne
    final_sar = sar_sign.iloc[-1]
    if final_sar == 1 and current_vwap_dist > current_threshold:
        final_sar = 0
    elif final_sar == -1 and current_vwap_dist < -current_threshold:
        final_sar = 0

    final_stoch = stoch_filtered[-1]
    if final_stoch == 1 and current_vwap_dist > current_threshold:
        final_stoch = 0
    elif final_stoch == -1 and current_vwap_dist < -current_threshold:
        final_stoch = 0

    return final_sar, final_stoch

def sync_renko_with_r2(df_renko, df_candles):
    """
    df_renko : colonnes ['time', 'close', ...]
    df_candles : colonnes ['time', 'r2', ...] (le R2 déjà calculé sur 14 ou 20 bougies)
    """
    # On s'assure que les colonnes 'time' sont au format datetime
    df_renko['time'] = pd.to_datetime(df_renko['time'])
    df_candles['time'] = pd.to_datetime(df_candles['time'])

    # Tri par temps obligatoire pour merge_asof
    df_renko = df_renko.sort_values('time')
    df_candles = df_candles.sort_values('time')

    # On fusionne : pour chaque brique Renko, on prend le R2 du chandelier le plus récent
    df_combined = pd.merge_asof(df_renko, df_candles[['time', 'r2']],
                                on='time',
                                direction='backward')
    return df_combined


# ============================================================================
# NOUVELLES FONCTIONS POUR LA SOLUTION HYBRIDE
# ============================================================================

class ZoneStabilityFilter:
    def __init__(self, min_stability_time=300):
        self.min_stability_time = min_stability_time
        self.current_zone = None
        self.zone_start_time = None

    def update(self, zone, current_time):
        """
        Surveille directement la stabilité de la zone discrète (-2, -1, 0, 1, 2).
        """
        if zone != self.current_zone:
            self.current_zone = zone
            self.zone_start_time = current_time
            return False  # Changement de zone → instable
        else:
            if self.zone_start_time is not None:
                stability_duration = (current_time - self.zone_start_time).total_seconds()
                if stability_duration >= self.min_stability_time:
                    return True  # Zone stable depuis assez longtemps
            return False

class ZoneStabilityFilter_w_proba:
    """
    Filtre de stabilité temporelle pour éviter les faux signaux.
    Ne valide une zone que si elle reste stable pendant un temps minimal.
    """
    def __init__(self, min_stability_time=300):  # 5 min par défaut
        self.min_stability_time = min_stability_time
        self.current_zone = None
        self.zone_start_time = None
        self.zone_history = deque(maxlen=10)  # Dernières 10 zones

    def update(self, proba, bornes, current_time):
        """
        Met à jour le filtre avec la dernière probabilité et le temps actuel.
        Args:
            proba: float, probabilité actuelle
            bornes: list, liste des 4 seuils [threshold_sell, close_buy, close_sell, threshold_buy]
            current_time: datetime, heure actuelle
        Returns:
            int or None: la zone stable (0, ±1, ±2) ou None si instable
        """
        # Déterminer la zone actuelle
        if proba < bornes[0]:
            zone = -2  # SV
        elif proba < bornes[1]:
            zone = -1  # V
        elif proba < bornes[2]:
            zone = 0   # N
        elif proba < bornes[3]:
            zone = 1   # A
        else:
            zone = 2   # SA

        # Si la zone change
        if zone != self.current_zone:
            self.current_zone = zone
            self.zone_start_time = current_time
            self.zone_history.append((zone, current_time))
            return False  # Changement de zone → pas de décision
        else:
            # Vérifier si on est stable depuis assez longtemps
            if self.zone_start_time is not None:
                stability_duration = (current_time - self.zone_start_time).total_seconds()
                if stability_duration >= self.min_stability_time:
                    return True # Zone stable → décision valide
            return False  # Zone instable → pas de décision

def calcul_bornes_dynamiques(param, df, r2, er):
    """
    Calcule des bornes dynamiques en fonction de la volatilité et de la confiance des modèles.
    Args:
        regression: bool, si True utilise la logique RNN
        param: dict, paramètres de configuration
        df: DataFrame, données avec indicateurs (ATR, etc.)
        r2: float, coefficient de détermination
        er: float, efficiency ratio
    Returns:
        list: [threshold_sell, close_buy, close_sell, threshold_buy]
    """
    # 1. Bornes de base
    parameters = param.get('parameters', {})
    base_bornes = [
        parameters.get('threshold_sell', 0.25),
        parameters.get('close_buy', 0.4),
        parameters.get('close_sell', 0.6),
        parameters.get('threshold_buy', 0.75)
    ]

    # 2. Calcul de la volatilité (ATR normalisé)
    if 'ATR' in df.columns and len(df) > 1:
        atr = df['ATR'].iloc[-1]
        atr_mean = df['ATR'].rolling(min(50, len(df))).mean().iloc[-1]
        volatility_ratio = atr / atr_mean if atr_mean != 0 else 1.0
    else:
        volatility_ratio = 1.0

    # 3. Calcul de la confiance des modèles
    confidence = (r2 + er) / 2 if (r2 + er) > 0 else 0.5

    # 4. Ajustement dynamique des bornes
    # Si volatilité élevée → élargir les zones (moins sensible)
    # Si confiance élevée → serrer les zones (plus précis)
    volatility_factor = 1.0 + (volatility_ratio - 1.0) * 0.3  # ±30% max
    confidence_factor = 1.0 - (1.0 - confidence) * 0.5  # 50% à 100%

    # Application brute
    b0 = base_bornes[0] * volatility_factor / confidence_factor
    b1 = base_bornes[1] * volatility_factor / confidence_factor
    b2 = base_bornes[2] * volatility_factor * confidence_factor
    b3 = base_bornes[3] * volatility_factor * confidence_factor

    # Sécurité anti-croisement avec un espacement minimal (ex: 0.05)
    eps = 0.05
    b0 = max(0.1, min(b0, 0.35))
    b1 = max(b0 + eps, min(b1, 0.45))
    b2 = max(b1 + eps, min(b2, 0.65))
    b3 = max(b2 + eps, min(b3, 0.90))
    dynamic_bornes = [b0, b1, b2, b3]

    if parameters.get("window_monitor",0) > 0:
        ol_r = parameters.get('open_level_rnn', 1.5)
        ol_t = parameters.get('open_level_tabicl', 1.5)
        cl_r = parameters.get('close_level_rnn', 0.1)
        cl_t = parameters.get('close_level_tabicl', 0.1)
        b_u = proba_final(ol_r, ol_t, 0.7)
        b_l = proba_final(cl_r, cl_t, 0.7)
        dynamic_bornes = [-b_u, -b_l, b_l, b_u]

    return dynamic_bornes

def detect_market_regime(df, regime_params=None, slope=0.0, volatility=1.0):
    """
    Détecte le régime du marché avec des paramètres configurables.
    Args:
        df: DataFrame avec colonnes 'close', 'high', 'low'
        regime_params: dict avec clés :
            - regression_window (int, défaut=20)
            - adx_period (int, défaut=14)
            - volatility_window (int, défaut=50)
            - volatility_threshold (float, défaut=1.5)
            - adx_threshold (int, défaut=25)
    Returns:
        str: "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"
    """
    # Paramètres par défaut
    params = {
        "regression_window": 20,
        "adx_period": 14,
        "volatility_window": 50,
        "volatility_threshold": 1.5,
        "adx_threshold": 25
    }
    if regime_params is not None:
        params.update(regime_params)
    reg_win = params["regression_window"]
    if len(df) < reg_win  or 'close' not in df.columns:
        return "RANGING"
    """
    # 1. Régression linéaire
    close_prices = df['close'].iloc[-reg_win:].values
    x = np.arange(len(close_prices))
    slope, _ = np.polyfit(x, close_prices, 1)
    # 2. Volatilité
    volatility = np.std(close_prices)
    """
    if len(df) > params["volatility_window"]:
        volatility_mean = df['close'].rolling(params["volatility_window"]).std().iloc[-1]
        volatility_ratio = volatility / volatility_mean if volatility_mean != 0 else 1.0
    else:
        volatility_ratio = 1.0

    # 3. ADX
    adx_period = params["adx_period"]
    if 'high' in df.columns and 'low' in df.columns:
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = -df['low'].diff().clip(upper=0)
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(adx_period).mean().iloc[-1]
        if atr == 0 or pd.isna(atr):
            atr = 1.0
        plus_di = 100 * (plus_dm.rolling(adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr)
        # CORRECTION ICI : Utilisation de np.where pour éviter le test 'if' sur une Série
        denominator = plus_di + minus_di
        dx = pd.Series(
            np.where(denominator != 0, 100 * abs(plus_di - minus_di) / denominator, 0),
            index=df.index
        )
        adx_series = dx.rolling(adx_period).mean()
        adx = adx_series.iloc[-1] if len(dx) >= adx_period and not pd.isna(adx_series.iloc[-1]) else 0
    else:
        adx = 0

    # 4. Détection du régime
    if adx > params["adx_threshold"]:
        if slope > 0:
            return "TRENDING_UP"
        else:
            return "TRENDING_DOWN"
    elif volatility_ratio > params["volatility_threshold"]:
        return "VOLATILE"
    else:
        return "RANGING"

def weighted_decision(proba_dict, weights, df, bornes, slope):
    """
    Combine les prédictions de plusieurs modèles IA avec des règles métiers.
    Args:
        proba_dict: dict {model_name: proba_array}, probabilités par modèle
        weights: dict {model_name: weight}, poids par modèle
        df: DataFrame, données avec indicateurs (RSI, MACD, etc.)
        bornes: list, liste des 4 seuils
    Returns:
        float: score final entre -2 et +2
    """
    # 1. Calcul des scores par modèle
    model_scores = {}
    for model_name, proba in proba_dict.items():
        if proba is None:
            continue
        # Gestion des listes, tableaux numpy et tenseurs PyTorch
        if hasattr(proba, 'detach'):  # C'est un tenseur PyTorch
            proba = proba.detach().cpu().numpy()
        last_proba = proba[-1] if isinstance(proba, (list, np.ndarray)) else float(proba)
        model_scores[model_name] = soft_zone_score(last_proba, bornes)

    if len(model_scores) == 0:
        return 0.0
    # 2. Pondération par les poids
    weighted_sum = 0
    total_weight = 0
    for model_name, score in model_scores.items():
        weight = weights.get(model_name, 1.0)
        weighted_sum += score * weight
        total_weight += weight
    # 3. Score moyen
    avg_score = weighted_sum / total_weight if total_weight != 0 else 0
    # 4. Ajustement par les règles métiers
    # Exemple : Si RSI > 70 → réduire le score (surachat)
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if rsi > 70:
            avg_score *= 0.7  # Réduction de 30%
        elif rsi < 30:
            avg_score *= 1.3  # Amplification de 30%
    # Exemple : Si MACD < 0 → réduire le score (tendance baissière)
    if 'MACD_hist' in df.columns:
        macd = df['MACD_hist'].iloc[-1]
        if macd < 0:
            avg_score *= 0.8
    # Exemple : Si la pente est forte → amplifier le score
    if slope > 0.01:
        avg_score *= 1.2
    elif slope < -0.01:
        avg_score *= 0.8
    return avg_score

def soft_zone_score(proba, bornes):
    """
    Calcule un score continu entre -2 et +2 en fonction de la proba.
    Plus la proba est éloignée des bornes, plus le score est extrême.
    Args:
        proba: float, probabilité (0-1)
        bornes: list, liste des 4 seuils [threshold_sell, close_buy, close_sell, threshold_buy]
    Returns:
        float: score entre -2 et +2
    """
    eps = 1e-6
    if proba <= bornes[0]:
        denom = bornes[0] if bornes[0] != 0 else eps
        return -2 + (proba / denom) * 1
    elif proba <= bornes[1]:
        denom = (bornes[1] - bornes[0])
        denom = denom if abs(denom) > eps else eps
        return -1 + ((proba - bornes[0]) / denom) * 1
    elif proba <= bornes[2]:
        return 0
    elif proba <= bornes[3]:
        denom = (bornes[3] - bornes[2])
        denom = denom if abs(denom) > eps else eps
        return 0 + ((proba - bornes[2]) / denom) * 1
    else:
        denom = (1.0 - bornes[3])
        denom = denom if abs(denom) > eps else eps
        return 1 + ((proba - bornes[3]) / denom) * 1

def discretize_score(score):
    """
    Convertit un score continu en code discret (-2, -1, 0, 1, 2).
    Args:
        score: float, score entre -2 et +2
    Returns:
        int: code de situation (-2, -1, 0, 1, 2)
    """
    if score >= 1.5:
        return 2  # SA (Strong Buy)
    elif score >= 0.5:
        return 1  # A (Buy)
    elif score <= -1.5:  # ✅ Vérifié EN PREMIER pour les négatifs
        return -2  # SV (Strong Sell)
    elif score <= -0.5:
        return -1  # V (Sell)
    else:
        return 0  # N (Neutral)

def enhanced_decision(proba_dict, weights, df,
                      param, r2, er, slope,volatility,
                      zone_filter, time_current):
    # 1. Détection du régime
    try:
        regime = detect_market_regime(df, param.get('market_regime', None), slope, volatility)
    except Exception as e:
        print(f"Error in detect_market_regime: {e}")
        regime = ""
    try:
        bornes = calcul_bornes_dynamiques(param, df, r2, er)  # proba supprimé
    except Exception as e:
        print(f"Error in calcul_bornes_dynamiques: {e}")
        bornes = [-2, -1, 0, 1, 2]
    # 1. Calcul du score continu (indépendant de VDIRECT)
    # slope calculée avec r2 et er dans pmxRko
    try:
        final_score = weighted_decision(proba_dict, weights, df, bornes, slope)
    except Exception as e:
        print(f"Error in weighted_decision: {e}")
        final_score = 0
    # 2. Discrétisation (toujours [-2, -1, 0, 1, 2])
    zone = discretize_score(final_score)
    # 3. ✅ APPLICATION DE VDIRECT (inversion si suiveur)
    VDIRECT = utils.config_utils.VDIRECT
    if not VDIRECT:  # VDIRECT=False = suiveur → on inverse les zones
        zone = -zone  # [2, 1, 0, -1, -2]
    # 4. Filtrage temporel
    try:
        if not zone_filter.update(zone, time_current):
            return 0, regime
    except Exception as e:
        print(f"Error in zone_filter.update: {e}")
        return 0, regime
    # 5. ✅ Adaptation au régime (corrigée)
    if regime in ["TRENDING_UP", "TRENDING_DOWN"]:
        if VDIRECT:  # Contre-tendance : on veut des signaux CONTRE la tendance
            # Bloquer les signaux DANS le sens de la tendance
            if (regime == "TRENDING_UP" and zone > 0) or (regime == "TRENDING_DOWN" and zone < 0):
                return 0, regime
        else:  # Suiveur : on veut des signaux DANS le sens de la tendance
            # Bloquer les signaux CONTRE la tendance
            if (regime == "TRENDING_UP" and zone < 0) or (regime == "TRENDING_DOWN" and zone > 0):
                return 0, regime
    # 7. Logique VSIMPLE
    VSIMPLE = utils.config_utils.VSIMPLE
    if VSIMPLE and abs(zone) < 2:
        return 0, regime

    # 8. Logique VTOTALE
    VTOTALE = utils.config_utils.VTOTALE
    if VTOTALE and regime == "VOLATILE" and abs(zone) == 2:
        return -zone, regime
    return zone, regime

