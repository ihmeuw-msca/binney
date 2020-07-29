import numpy as np
from flipper.data.data import LRSpecs, BinomDataSpecs


def test_binom_data_specs():
    specs = BinomDataSpecs(
        col_obs='success',
        col_total='total'
    )
    assert specs.col_total == 'total'
    assert specs.col_obs == 'success'
    assert specs.col_obs_se is None
    assert specs.col_groups is None


def test_lr_specs(df, n):
    specs = LRSpecs(
        col_success='success',
        col_total='total',
        covariates=['x1']
    )
    specs.configure_data(df)
    dd = specs.data._param_set[0].design_matrix_fe
    assert dd.shape == (n, 2)
    np.testing.assert_array_equal(
        dd[:, 0],
        np.ones(shape=n)
    )
    np.testing.assert_array_equal(
        dd[:, 1],
        df['x1'].values
    )
