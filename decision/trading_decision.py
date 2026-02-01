import numpy as np
from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE

def trading_decision(pos: int, po: float, pc: float, dd, dj, proba, sl: float, tp: float, buy_thr=0.6, sell_thr=0.4, close_buy=0.53, close_sell=0.47, trace=False):
    if dd is None and proba is None:
        return NONE
    nn = len(dd)
    if proba is not None:
        n = len(proba)
        if n < nn:
            nn = n
    o_signal = np.zeros(nn)
    c_signal = np.zeros(nn)
    # proba = np.full(len(df_test), 2.5)
    if proba is not None:
        proba = proba[-nn:]
        # === SIGNALS OUVERTURE ===
        p = np.clip(np.asarray(proba), 0.0, 1.0)
        o_signal[p >= buy_thr] = 1
        o_signal[p <= sell_thr] = -1
        # === SIGNALS FERMETURE === (exemple, même que ouverture, ou ajuster) jamais 2 choix en une condition
        # il faut séparer en 2 conditions liées par | ou &
        c_signal[p >= close_buy] = 1
        c_signal[p <= close_sell] = -1
        c_signal[(p < close_buy) & (p > close_sell)] = 5
    # === RULES ===
    ssd, ssc, sso = dd['direction'].iloc[-2], dd['sigc'].iloc[-2], dd['sigo'].iloc[-2]
    psd, psc, pso = dd['direction'].iloc[-3], dd['sigc'].iloc[-3], dd['sigo'].iloc[-3]
    if dj is not None:
        djd, djc, djo = dj['direction'].iloc[-1], dj['sigc'].iloc[-1], dj['sigo'].iloc[-1]
    else:
        djd = np.zeros(nn)
        djc = np.zeros(nn)
        djo = np.zeros(nn)
    # === BACKTEST === open enabled
    cs = c_signal[-1]
    co = o_signal[-1]
    if trace:
        print(f"prob c={cs} o={co}")
    sigOpen = sigClose = NONE

    def to_by_close(pos, ssc, sso, cs, co):
        sigClose = NONE
        if pos != NONE:
            if (((ssc == 4 or cs == 5) and (pos != sso and sso != NONE)) or
                    (pos == SELL and (cs == BUY or co == BUY)) or
                    (pos == BUY and (cs == SELL or co == SELL))):
                sigClose = CLOSE
        return sigClose

    if pos == 0:
        if sso == BUY or sso == SELL or co == BUY or co == SELL:
            if sso != NONE and co != NONE and sso != co:
                sigOpen = NONE
            signal = sso if sso != NONE else co
            if signal != cs and cs != 5 and cs != 0:
                sigOpen = NONE
            # ajout japonaises
            # if dj is not None and djc[-1] != NONE and djc[-1] != signal:                return NONE
            # ajout 2 open m^sens ? a revoir
            # if signal == pso or pso == NONE or psd = ssd:
            # ajout non signal de cloture
            # if ssc != 4
            sigOpen =  signal
    # ==== cloture d'une position de sens pos
    if pos != 0:
        # tp ou sl ou risk etc...
        TP = (pc - po)*pos > tp if tp != 0 else False
        SL = (po - pc)*pos > sl if sl != 0 else False
        if TP or SL:
            sigClose = FCLOSE
        # ajout close si hors bbands
        # if (djc[-1] == BUY and pos == BUY) or (djc[-1] == SELL and pos == SELL): return NCLOSE
        # demande cloture normale ou inversion sens
        else:
            sigClose = to_by_close(pos, ssc, sso, cs, co)
    else:
        sigClose = to_by_close(sigOpen, ssc, sso, cs, co)
    return sigClose, sigOpen
