import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope='session')
def intercept():
    return 1


@pytest.fixture(scope='session')
def slope():
    return 2


@pytest.fixture(scope='session')
def n():
    return 2000


@pytest.fixture(scope='session')
def simple_df(slope, n):
    p = np.repeat(np.exp(slope) / (1 + np.exp(slope)), repeats=n)
    df = pd.DataFrame({
        'success': np.random.binomial(n=100, size=len(p), p=p),
        'total': np.repeat(100, repeats=len(p))
    })
    return df


@pytest.fixture(scope='session')
def df(intercept, slope, n):
    np.random.seed(0)
    x = np.random.randn(n)
    p = np.exp(intercept + x * slope) / (1 + np.exp(intercept + x * slope))
    df = pd.DataFrame({
        'success': np.random.binomial(n=100, size=len(p), p=p),
        'total': np.repeat(100, repeats=len(p)),
        'p': p,
        'x1': x
    })
    return df


@pytest.fixture(scope='session')
def spline_df(n):
    np.random.seed(0)
    x = np.random.uniform(low=0, high=10, size=n)
    p = np.exp(np.sin(x)) / (1 + np.exp(np.sin(x)))
    df = pd.DataFrame({
        'success': np.random.binomial(n=100, size=len(p), p=p),
        'total': np.repeat(100, repeats=len(p)),
        'p': p,
        'x1': x
    })
    return df


@pytest.fixture(scope='session')
def spline_concave_df(n):
    np.random.seed(0)
    x = np.random.uniform(low=0, high=np.pi, size=n)
    p = np.exp(np.sin(x)) / (1 + np.exp(np.sin(x)))
    df = pd.DataFrame({
        'success': np.random.binomial(n=100, size=len(p), p=p),
        'total': np.repeat(100, repeats=len(p)),
        'p': p,
        'x1': x
    })
    return df
