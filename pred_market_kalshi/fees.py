"""
Fee Formula: Fee = ceil(count * fee_rate * min(price, 1 - price))
"""
import math

def calculate_kalshi_taker_fee(price: float, count: float) -> float:
    """
    Calculates estimated taker execution fee on Kalshi.
    
    Parameters:
        price: Probability price (0.01 to 0.99)
        count: Number of contracts traded
    """
    if price <= 0.0 or price >= 1.0 or count <= 0:
        return 0.0

    # Implied Kalshi standard taker fee multiplier (~3.5% of probability risk)
    fee_rate = 0.035 
    implied_risk = min(price, 1.0 - price)
    
    # Fees are charged in cents and rounded up
    raw_fee_cents = count * fee_rate * implied_risk * 100.0
    return math.ceil(raw_fee_cents) / 100.0