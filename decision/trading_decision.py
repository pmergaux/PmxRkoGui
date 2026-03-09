import numpy as np

from decision.candle_decision import signal_col
from utils.utils import NONE, BUY, SELL, CLOSE, FCLOSE, DIRBUYCLOSE, DIRSELLCLOSE, EMABUYCLOSE, EMASELLCLOSE, \
    PROBANEUTRE, safe_format


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
    b_signal = np.zeros(nn)
    # proba = np.full(len(df_test), 2.5)
    try:
        if proba is not None:
            proba = proba[-nn:]
            # === SIGNALS OUVERTURE ===
            p = np.clip(np.asarray(proba), 0.0, 1.0)
            o_signal[p >= buy_thr] = BUY
            o_signal[p <= sell_thr] = SELL
            # === SIGNALS FERMETURE === (exemple, même que ouverture, ou ajuster) jamais 2 choix en une condition
            # il faut séparer en 2 conditions liées par | ou &
            c_signal[p >= close_buy] = BUY
            c_signal[p <= close_sell] = SELL
            #c_signal[(p < close_buy) & (p > close_sell)] = NONE
        else:
            proba = [0.5]
    except BaseException as e:
        print(f"proba={proba} et err {e}")
    # === RULES ===
    # ssc est la direction si close_sens est True
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
    #cb = b_signal[-1]
    sigOpen = sigClose = NONE
    # un des indicateurs n'est du même sens que pos
    def close_indicators(dd, pos):
        closing = False
        for col in signal_col:
            if col in dd.columns:
                closing |= ((dd.iloc[-2][col] != pos) & (dd.iloc[-2][col] != NONE))
        return closing

    def to_be_openned(ssc, sso, cs, co):
        if abs(sso) == BUY or abs(co) == BUY:
            #if sso != NONE and co != NONE and sso != co:                 return NONE
            signal = ssc + sso + cs
            signal = 1 if signal > 1 else -1 if signal < -1 else signal
            if signal == NONE:
                signal = sso if sso != NONE else co
            # ajout japonaises
            # if dj is not None and djc[-1] != NONE and djc[-1] != signal:                return NONE
            # ajout 2 open m^sens ? a revoir
            # if signal == pso or pso == NONE or psd = ssd:
            # ajout non signal de cloture
            # if ssc != 4
            return signal
        return NONE
    # si cs == PROBANEUTRE pas de test, sinon proba en dehors donc cb donne la tendance et n'est pas NONE
    def to_be_closed(pos, ssc, sso, cs):
        if pos != NONE:
            scs = SELL if ssc < 0 else BUY if ssc > 0 else NONE
            if ((scs != NONE and pos != scs and pos != sso and sso != NONE) or
                    (cs != NONE and pos != cs)):
                return CLOSE
        return NONE

    if pos == 0:
        sigOpen = to_be_openned(ssd, sso, cs, co)
        sigClose = to_be_closed(sigOpen, ssc, sso, cs)
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
        elif close_indicators(dd, pos):
            sigClose = CLOSE
        else:
            sigClose = to_be_closed(pos, ssc, sso, cs)
        sigOpen = to_be_openned(ssd, sso, cs, co)
    if trace:
        indic = []
        for col in signal_col:
            if col in dd.columns:
                indic.append(col)
                indic.append(int(dd.iloc[-2][col]))
        print(f"sign={sigClose:.0f}#{sigOpen:.0f} prob={safe_format(proba[-1],'.3f')} cs={cs:.0f} co={co:.0f} "
              f"sc={ssc} so={sso}={indic}")
    return sigClose, sigOpen
