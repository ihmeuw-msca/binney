import numpy as np
import pytest
import pandas as pd

from flipper.run.run import BinomRun, RunException

REL_TOL = 1e-2


def test_scipy_opt(df, intercept, slope):
    true_params = [intercept, slope]
    b_run = BinomRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df,
        solver_method='scipy'
    )
    b_run.fit(
        x_init=[1., 1.],
        solver_options={
            'maxiter': 500,
            'maxcor': 25
        }
    )
    rel_error = (b_run.solver.x_opt - true_params) / true_params
    assert all(rel_error < REL_TOL)
    assert b_run.solver.predict().shape == (len(df), )
    assert (0 <= b_run.solver.predict()).all()
    assert (1 >= b_run.solver.predict()).all()


def test_ipopt(df, intercept, slope):
    true_params = [intercept, slope]
    b_run = BinomRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df,
        solver_method='ipopt'
    )
    b_run.fit(x_init=[1., 1.])
    rel_error = (b_run.solver.x_opt - true_params) / true_params
    assert all(rel_error < REL_TOL)


def test_unrecognized_solver(df, intercept, slope):
    with pytest.raises(RunException):
        BinomRun(
            col_success='success',
            col_total='total',
            covariates=['x1'],
            df=df,
            solver_method='my_random_solver'
        )


def test_fractional_outcomes():
    p = np.repeat(np.exp(3) / (1 + np.exp(3)), repeats=1e3)
    np.random.seed(100)
    df = pd.DataFrame({
        'success': np.random.binomial(n=1, size=len(p), p=p),
        'total': np.repeat(1, repeats=len(p))
    })
    b_run = BinomRun(
        col_success='success',
        col_total='total',
        df=df,
        solver_method='scipy'
    )
    b_run.fit(x_init=[1.])
    assert np.round(b_run.solver.x_opt) == 3


def test_splines(spline_df):
    splines = {
        'x1': {
            'degree': 3,
            'knots_num': 4,
            'knots_type': 'frequency'
        }
    }
    b_run = BinomRun(
        col_success='success',
        col_total='total',
        df=spline_df,
        splines=splines,
        solver_method='ipopt'
    )
    b_run.fit(x_init=[0.0] * 6)
    predictions = b_run.predict()
    np.testing.assert_array_almost_equal(
        x=predictions,
        y=spline_df['p'].values,
        decimal=1
    )


def test_spline_constraints(spline_concave_df):
    splines = {
        'x1': {
            'degree': 3,
            'knots_num': 4,
            'knots_type': 'frequency',
            'convex': True
        }
    }
    b_run = BinomRun(
        col_success='success',
        col_total='total',
        df=spline_concave_df,
        splines=splines,
        solver_method='ipopt'
    )
    b_run.fit(x_init=[0.0] * 6)
    predictions = b_run.predict()
    np.testing.assert_array_almost_equal(
        x=predictions,
        y=spline_concave_df['p'].values,
        decimal=1
    )
