import numpy as np
import pytest
from flipper.model.model import LRBinomModel
from flipper.run.run import BinomBootstrap

from anml.solvers.interface import Solver


@pytest.mark.parametrize("seed", np.arange(10))
def test_binom_bootstrap(df, seed):
    np.random.seed(seed)
    mod = LRBinomModel()
    sol = Solver()
    boot = BinomBootstrap(model=mod, solver=sol, df=df)
    sample = boot._sample(df=df, col_obs='success', col_total='total')
    np.testing.assert_array_equal(
        sample['total'],
        df['total']
    )
    np.testing.assert_array_equal(
        sample['x1'],
        df['x1']
    )
    np.testing.assert_array_equal(
        sample['total'],
        df['total']
    )
    assert not (sample['success'].values == df['success'].values).all()
