import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope='session')
def simple_df():
    n = 1000
    p = np.repeat(np.exp(2) / (1 + np.exp(2)), repeats=n)
    df = pd.DataFrame({
        'success': np.random.binomial(n=100, size=len(p), p=p),
        'total': np.repeat(100, repeats=len(p))
    })
    return df


@pytest.fixture(scope='session')
def df():
    n = 1000
    x = np.random.randn(n)
    beta = 2
    p = np.exp(x * beta) / (1 + np.exp(x * beta))
    df = pd.DataFrame({
        'success': np.random.binomial(n=100, size=len(p), p=p),
        'total': np.repeat(100, repeats=len(p)),
        'x1': x
    })
    return df
