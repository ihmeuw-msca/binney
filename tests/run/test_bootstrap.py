import numpy as np
import pytest
from binney.model.model import BinomialModel
from binney.run.run import BinneyRun
from binney.run.bootstrap import BinomialBootstrap, BernoulliBootstrap

from anml.solvers.interface import Solver


@pytest.mark.parametrize("seed", np.arange(10))
def test_binomial_sampling(df, seed):
    np.random.seed(seed)
    mod = BinomialModel()
    sol = Solver()
    boot = BinomialBootstrap(model=mod, solver=sol, df=df)
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


def test_bernoulli_sampling(bernoulli_df):
    mod = BinomialModel()
    sol = Solver()
    boot = BernoulliBootstrap(model=mod, solver=sol, df=bernoulli_df)
    sample = boot._sample(df=bernoulli_df)
    assert len(sample) == len(bernoulli_df)


def test_bootstrap_run(df, n):
    np.random.seed(99)
    b_run = BinneyRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df,
        solver_method='ipopt',
        data_type='binomial'
    )
    b_run.fit()
    b_run.make_uncertainty(n_boots=15)
    assert b_run.bootstrap.parameters.shape == (15, 2)
    np.testing.assert_array_almost_equal(
        b_run.bootstrap.parameters[0, :],
        np.array([0.99940296, 1.99394325])
    )
    np.testing.assert_array_almost_equal(
        b_run.bootstrap.parameters[5, :],
        np.array([1.01366631, 1.99569274])
    )
    np.testing.assert_array_almost_equal(
        b_run.bootstrap.parameters.mean(axis=0),
        b_run.params_opt,
        decimal=1
    )
    uis = np.quantile(b_run.bootstrap.parameters, q=[0.025, 0.975], axis=0)
    assert all(b_run.params_opt > uis[0, :])
    assert all(b_run.params_opt < uis[1, :])
    draws = b_run.predict_draws(df=df)
    assert draws.shape == (15, n)
    assert all(draws.var(axis=1) > 0)


def test_bernoulli_run(bernoulli_df, n):
    np.random.seed(99)
    b_run = BinneyRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=bernoulli_df,
        solver_method='ipopt',
        data_type='bernoulli'
    )
    b_run.fit()
    b_run.make_uncertainty(n_boots=15)
    assert b_run.bootstrap.parameters.shape == (15, 2)
    np.testing.assert_array_almost_equal(
        b_run.bootstrap.parameters.mean(axis=0),
        b_run.params_opt,
        decimal=1
    )
    uis = np.quantile(b_run.bootstrap.parameters, q=[0.025, 0.975], axis=0)
    assert all(b_run.params_opt > uis[0, :])
    assert all(b_run.params_opt < uis[1, :])
    draws = b_run.predict_draws(df=bernoulli_df)
    assert draws.shape == (15, n)
    assert all(draws.var(axis=1) > 0)
