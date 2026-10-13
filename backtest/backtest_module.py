import copy
import json
import os
import sys
import pickle
import time

import joblib
import gc

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import reservoir
from tensorflow.keras import backend as K

from decision.candle_decision import add_indicators_optimized, choix_features_numba
from optimize.optimize_periodicity import PmxRkoBacktester
from train.pipeline_manager import train_all_models, prepare_renko
from decision.trading_decision import decision_ai, decision_monitor, IndicatorMonitor, proba_final
import utils.config_utils
from utils.config_utils import config_to_hash, prepare_to_hashcode
from utils.model_utils import save_model_artifact, save_jepa, cntrl_jepa, kr_servers, save_finjepa, cntrl_finjepa
from utils.utils import ROUGE, RESET, BLEU
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def send_to_dashboard(pnl_list, result):
    url = "http://localhost:5000/data"
    if isinstance(pnl_list, np.ndarray):
        data_to_send = pnl_list.tolist()
    else:
        # Si c'est déjà une liste, on la garde telle quelle
        data_to_send = pnl_list
    payload = {
        "historique": data_to_send, # On envoie les 20 derniers points
        "result": result
    }
    try:
        # Timeout très court (0.5s) pour ne pas ralentir le backtest
        requests.post(url, json=payload, timeout=0.5)
        print(f"envoyé au dashboard")
    except:
        pass # Le dashboard est fermé, on continue le backtest sans bugger


def save_to_top10(config_std, score, result_copy, models, scaler, trial=None):
    """Sauvegarde intelligente Top 10 avec RNN + TabICL"""
    try:
        nd = result_copy.get('days', 6)
        if result_copy.get('trades', 0) < int(nd / 2.2) or result_copy.get('profit', 0) < 20:
            print(f"{ROUGE}+++++++++++++++++++++ result rejeté {score}{RESET}")
            return
    except Exception as e:
        print(f"err result {type(result_copy)}: {e}")
        return

    # Gestion du verrou (lock) pour éviter les accès concurrents
    lock_dir = os.path.join(ROOT_DIR, "save_model.lock")
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            time.sleep(0.05)

    try:
        pnl = result_copy['positions']
        del result_copy['positions']

        top_dir = "/media/pierre/datad/data/top10"
        os.makedirs(top_dir, exist_ok=True)

        # Liste des dossiers existants et récupération des scores
        existing = [d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))]
        scores = []
        for d in existing:
            try:
                meta_path = os.path.join(top_dir, d, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                        scores.append((meta.get('score', -9999), d))
            except:
                continue

            # Tri du meilleur au pire (du plus grand score au plus petit)
            scores.sort(key=lambda x: x[0], reverse=True)

            # Vérification si le score exact existe déjà
            for s, d in scores:
                if s == score:
                    return

            # 2. Filtrage strict pour un Top 5 maximum
            if len(scores) >= 5:
                # Le 5ème meilleur score se trouve à l'index 4
                fifth_best_score = scores[4][0]
                if score <= fifth_best_score:
                    # Le score est inférieur ou égal au 5ème : rejeté !
                    return
        # ==========================================================
        # VALIDÉ : Le modèle est bon, on effectue les contrôles finaux
        # ==========================================================
        if 'JEPA' in config_std["live"]["version"]:
            jepa_model, stats = models['JEPA']
            cntrl_jepa(jepa_model.state_dict(), config_std.get("jepa", {}))

        if 'FINJEPA' in config_std["live"]["version"]:
            finjepa_model, stats = models['FINJEPA']
            actual_finjepa = finjepa_model.model if hasattr(finjepa_model, 'model') else finjepa_model
            # Correction ici : on appelle bien cntrl_finjepa avec le state_dict
            cntrl_finjepa(actual_finjepa.state_dict(), config_std.get("finjepa", {}))

        # Création du dossier UNIQUEMENT MAINTENANT (évite les dossiers vides)
        hcode = config_std.get('live', {}).get('hcode', "unknowb")
        model_dir = os.path.join(top_dir, f"rank_temp_{hcode}")
        os.makedirs(model_dir, exist_ok=True)

        # ================== Affichage du best ==============
        if len(scores) == 0 or score >= scores[0][0]:
            try:
                result_copy['hcode'] = hcode
                result_copy['renko_size'] = config_std.get('parameters', {}).get('renko_size', 1)
                send_to_dashboard(pnl_list=pnl, result=result_copy)
            except Exception as e:
                print(f"err dispatch score {e} ")

        # ========= Sauvegarde de la Config ==================
        strategy_name = config_std.get('live', {}).get('name', 'unknown_strategy')
        target_col = config_std.get('target', {}).get('target_col', None)
        if target_col is not None and isinstance(target_col, list):
            target_col = target_col[0]
            config_std['target']['target_col'] = target_col
        nested_config = {strategy_name: config_std}
        config_path = os.path.join(model_dir, f"config_{hcode}.json")
        with open(config_path, 'w') as f:
            json.dump(nested_config, f, indent=4)
        print(f"  -> Config standard sauvegardée sous {config_path}")

        # ==================== SAUVEGARDE DES MODÈLES ====================
        for key, model in models.items():
            if model is not None:
                key_upper = key.upper()
                if 'TAB' == key_upper:
                    continue
                if 'TABFF' == key_upper:
                    temp_ckpt_dir = f"./ckpts_{key}_{os.getpid()}"
                    source_best_ckpt = os.path.join(temp_ckpt_dir, "best.ckpt")
                    target_ckpt_path = os.path.join(model_dir, f"{key}_{hcode}_best.ckpt")
                    if os.path.exists(source_best_ckpt):
                        import shutil
                        shutil.copy(source_best_ckpt, target_ckpt_path)
                        print(f"  -> Best checkpoint TABFF sauvegardé sous {target_ckpt_path}")
                    else:
                        print(f"  ⚠️ Attention: best.ckpt introuvable pour {key} dans {source_best_ckpt}")
                elif 'JEPA' == key_upper:
                    model_path = os.path.join(model_dir, f"{key}_{hcode}.pth")
                    jepa_model, stats = model
                    save_jepa(jepa_model, stats, model_path, config_std.get("jepa", {}))
                elif 'FINJEPA' == key_upper:
                    finjepa_path = os.path.join(model_dir, f"{key}_{hcode}.pth")
                    finjepa_model, stats = model
                    save_finjepa(finjepa_model, stats, finjepa_path, config_std.get("finjepa", {}))
                    print(f"  -> FinJepa sauvegardé sous {finjepa_path}")
                elif 'TABFIN' == key_upper:
                    try:
                        joblib.dump(model, os.path.join(model_dir, f"{key}_{hcode}.joblib"))
                        print(f"  -> TabFin (TabICL) sauvegardé sous {key}_{hcode}.joblib")
                    except Exception as e:
                        print(f"  ⚠️ Erreur sauvegarde TABFIN : {e}")
                else:
                    try:
                        if hasattr(model, 'save'):  # Keras
                            model.save(os.path.join(model_dir, f"{key}_{hcode}.keras"))
                        else:
                            joblib.dump(model, os.path.join(model_dir, f"{key}_{hcode}.joblib"))
                    except Exception as e:
                        print(f" attribut save err {e}")

        # ==================== SAUVEGARDE SCALER ====================
        if scaler is not None:
            with open(os.path.join(model_dir, f"scaler_{hcode}.pkl"), 'wb') as f:
                pickle.dump(scaler, f)

        # ==================== MÉTADONNÉES ====================
        meta = {
            "score": float(score),
            "result": result_copy
        }
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"🏆 {BLEU}Top 10 mis à jour !{RESET} Score: {score:.4f} → {hcode}\n{result_copy}")
        # élimination des excédents
        # ==================== NETTOYAGE STRICT TOP 5 ====================
        # On reliste tout le dossier pour purger s'il y a plus de 5 dossiers
        all_durs = [d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))]
        all_scores = []
        for d in all_durs:
            meta_path = os.path.join(top_dir, d, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    all_scores.append((meta.get('score', -9999), os.path.join(top_dir, d)))

        # On trie du meilleur au pire
        all_scores.sort(key=lambda x: x[0], reverse=True)

        # Si on dépasse 5 dossiers, on supprime définitivement les plus mauvais
        if len(all_scores) > 5:
            import shutil
            for bad_score, bad_path in all_scores[5:]:
                print(f"  -> Suppression du modèle excédentaire (score: {bad_score}) : {bad_path}")
                shutil.rmtree(bad_path, ignore_errors=True)

        print(f"🏆 {BLEU}Top 5 mis à jour !{RESET} Score: {score:.4f} → {hcode}\n{result_copy}")
    except BaseException as e:
        print(f"{ROUGE} err save score{RESET} {e}")
        raise e
    finally:
        if os.path.exists(lock_dir):
            os.rmdir(lock_dir)

def run_backtest(config_std, trial=None):
    total_time = time.time()
    score = -9999.0
    # Initialisation globale (à faire une seule fois au lancement)
    monitor_rnn = None
    monitor_tabicl = None
    try:
        df_renko = config_std.get('data', None)
        if df_renko is None or len(df_renko) < 100:
            return score, {}
        del config_std['data']
        # ====================== TON PIPELINE EXISTANT ======================
        # Remplace cette partie par ton vrai appel à train_all_models si tu l'as
        # Exemple simplifié :
        config_std['live']['hcode'] = config_to_hash(prepare_to_hashcode(config_std))
        config_def = copy.deepcopy(config_std)
        if config_std.get('parameters', {}).get("window_monitor", 0) > 0:
            monitor_rnn = IndicatorMonitor(config_std["parameters"]["window_monitor"])
            monitor_tabicl = IndicatorMonitor(config_std["parameters"]["window_monitor"])
        monitor_indic = IndicatorMonitor(12, 3)
        monitor_means = IndicatorMonitor(12, 3)
        train_result = train_all_models(config_std, df_renko, trial)
        config_def['jepa'] = copy.deepcopy(config_std['jepa'])
        if len(train_result) == 0:
            return score, {}
        models = train_result['models']
        scaler = train_result['scaler']
        df_renko_test = train_result['renko_test']
        del train_result['renko_test']
        # ====================== TON PIPELINE EXISTANT ======================
        delta = df_renko_test['time'].iloc[-1] - df_renko_test['time'].iloc[0]
        nb_jours = delta.total_seconds() / 86400.0
        if nb_jours < 1: nb_jours = 1.0  # Sécurité anti-division par zéro
        pmxTest = PmxRkoBacktester(config_std, None)
        #pmxTest.local = True
        pmxTest.scaler = scaler
        pmxTest.models = models
        pmxTest.all_bricks = df_renko_test.iloc[-400:]
        print(f"size {len(df_renko_test)} days {nb_jours} version {config_std['live']['version']} "
              f"VS {utils.config_utils.VSIMPLE}, VT {utils.config_utils.VTOTALE}, VD {utils.config_utils.VDIRECT} VE {utils.config_utils.VENHANCE}")

        mini = 128
        for key, model in models.items():
            key_lower = key.lower()
            if 'jepa' == key_lower:
                mini = config_std.get('jepa', {}).get("SEQ_LEN", 128)
            elif key in kr_servers:
                mini = config_std.get(key_lower, {}).get(f'{key_lower}_seq_len', 24)
        pmxTest.run_backtest_sequential(max(128, mini))
        trade_result = pmxTest.performance(trace=True)
        """
        # entraitement monitors sur train et val
        # print(f"TRAIN rnn {proba_rnn_train[-5:]}, tabicl {proba_tabicl_train[-5:]}")
        # print(f"VAL rnn {proba_rnn_val[-5:]}, tabicl {proba_tabicl_val[-5:]}")
        if config_std.get('parameters', {}).get("window_monitor", 0) > 0:
            try:
                decision_monitor(monitor_rnn, proba_rnn_train, monitor_tabicl, proba_tabicl_train,
                                 config_std["parameters"])
            except Exception as e:
                print(f"ERREUR dans decision_monitor train: {e}")
                return -9999.0, {}
            try:
                decision_monitor(monitor_rnn, proba_rnn_val, monitor_tabicl, proba_tabicl_val,
                                 config_std["parameters"])
            except Exception as e:
                print(f"ERREUR dans decision_monitor val: {e}")
                return -9999.0, {}
        if proba_rnn_val is not None:
            proba = proba_final(proba_rnn_val, proba_tabicl_val, 0.7, False)
            # 1. Calcul du nombre de fois que slope_window rentre dans proba (division entière)
            multiple = len(proba) // 3
            if multiple > 0:
                # 2. Prendre dans proba exactement ce multiple (les N derniers éléments qui forment des blocs parfaits)
                total_elements = multiple * 3
                subset = proba[-total_elements:]
                # 3. Split de ce subset en morceaux de taille exacte 'slope_window'
                chunks = np.array_split(subset, multiple)
                for chunk in chunks:
                    monitor_indic.update(chunk)
                    _, moy = monitor_indic.get_current_z()
                    # print(f"proba_moy {moy}")
                    monitor_means.update(moy)
            else:
                # Si proba est plus petit que slope_window, on traite ce qu'on a
                print(f"{ROUGE}proba is too short {len(proba)} < 3{RESET}")
                for val in proba:
                    monitor_indic.update(val)
                    _, moy = monitor_indic.get_current_z()
                    monitor_means.update(moy)
        X_test = df_renko_test
        try:
            proba_rnn, proba_tabicl = decision_ai(X_test, config_std, scaler, model_rnn, model_tabicl, None)
        except Exception as e:
            print(f"ERREUR dans ai_decision: {e}")
            return -9999.0, {}
        try:
            score, trade_result = backtest_decision(df_renko_test, None, monitor_rnn, proba_rnn, monitor_tabicl, proba_tabicl,
                                                    monitor_indic, monitor_means, config_std, trial)
            # print(f"score: {score:.4f}, result: {trade_result}")
        except Exception as e:
            print(f"ERREUR dans backtest_decision: {e}")
            return -9999.0, {}
        finally:
            del X_test
            del df_renko
        """
        try:
            score = trade_result['score']
            if score > 0:
                save_to_top10(config_def, score, trade_result, models, scaler, trial)
        except Exception as e:
            print(f"ERREUR dans save aucun trade ? : {e}")
            return score, {}
        return score, trade_result

    except Exception as e:
        print("ERREUR dans run_backtest:", e)
        return score, {}
    finally:
        K.clear_session()
        gc.collect()
        print(f"durée de ce test: {time.time() - total_time:.2f}s")
