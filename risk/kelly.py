# risk/kelly.py
def kelly_fraction(win_prob: float, odds: float = 1.0) -> float:
    """
    Calcule la fraction optimale du capital à risquer selon le Kelly Criterion.

    Formule : f* = (bp - q) / b
    Où :
        p = probabilité de gain
        q = 1 - p
        b = odds (gain net / mise)

    Args:
        win_prob (float): Probabilité de gain (ex: 0.60)
        odds (float): Ratio gain/perte (ex: 1.0 pour RR 1:1)

    Returns:
        float: Fraction du capital à risquer (0.0 à 1.0)
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    q = 1 - win_prob
    return max(0.0, (win_prob * (odds + 1) - 1) / odds)


def dynamic_lot_size(equity: float, risk_percent: float, kelly_f: float, max_lot: float = 10.0) -> float:
    """
    Convertit la fraction Kelly en taille de lot réelle.

    Args:
        equity (float): Capital actuel
        risk_percent (float): Risque fixe par trade (%)
        kelly_f (float): Fraction Kelly
        max_lot (float): Lot maximum autorisé

    Returns:
        float: Taille de lot
    """
    risk_amount = equity * (risk_percent / 100)
    base_lot = risk_amount / 1000  # 1 lot = 1000 $ de risque (approximation)
    return min(max_lot, base_lot * kelly_f)


# === EXEMPLE D'UTILISATION ===
if __name__ == "__main__":
    equity = 10000
    win_prob = 0.62
    odds = 1.5
    risk_percent = 2.0

    kelly = kelly_fraction(win_prob, odds)
    half_kelly = kelly * 0.5
    lot = dynamic_lot_size(equity, risk_percent, half_kelly)

    print(f"Kelly pur : {kelly:.1%}")
    print(f"Half Kelly : {half_kelly:.1%}")
    print(f"Lot size : {lot:.2f}")
