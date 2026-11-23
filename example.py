import numpy as np
import pandas as pd

def covariance_before(prices, date, window=100):
    """Use only prices observed strictly before the allocation date."""
    history = prices.sort_index().ffill().dropna()
    history = history.loc[history.index < date].tail(window + 1)
    if len(history) != window + 1:
        raise ValueError("Insufficient common price history")
    return history.pct_change().dropna().cov()

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=102)
    prices = pd.DataFrame(np.exp(rng.normal(0, .01, (102, 3)).cumsum(axis=0)), index=dates)
    result = covariance_before(prices, dates[-1])
    assert np.allclose(result, prices.iloc[:-1].pct_change().dropna().cov())
    print(result)
