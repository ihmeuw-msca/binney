from typing import Dict

import numpy as np
from binney.run.run import BinneyRun


def test_hierarchical_bootstrap(n, group_data):
    np.random.seed(99)
    n_boots = 30
    b_run = BinneyRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=group_data,
        col_group='g',
        solver_method='scipy',
        data_type='bernoulli'
    )
    b_run.fit()
    b_run.make_uncertainty(n_boots=n_boots)
    assert len(b_run.bootstrap.parameters) == n_boots
    for param in b_run.bootstrap.parameters:
        assert isinstance(param, Dict)
        for group, item in param.items():
            assert len(param[group]) == 2
    draws = b_run.predict_draws(df=group_data)
    assert draws.shape == (n_boots, n)
    assert all(draws.var(axis=1) > 0)
    assert (group_data['p'].values > np.quantile(draws, q=0.025, axis=0)).all()
    assert (group_data['p'].values < np.quantile(draws, q=0.975, axis=0)).all()
