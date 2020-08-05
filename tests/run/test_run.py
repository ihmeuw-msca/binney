import numpy as np
import pytest
import pandas as pd

from flipper.run.run import FlipperRun, RunException

REL_TOL = 1e-2


def test_scipy_opt(df, intercept, slope):
    true_params = [intercept, slope]
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df,
        solver_method='scipy',
        solver_options={
            'maxiter': 500
        }
    )
    b_run.fit()
    rel_error = (b_run.params_opt - true_params) / true_params
    assert all(rel_error < REL_TOL)
    assert b_run.solver.predict().shape == (len(df), )
    assert (0 <= b_run.solver.predict()).all()
    assert (1 >= b_run.solver.predict()).all()


def test_ipopt(df, intercept, slope):
    true_params = [intercept, slope]
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df,
        solver_method='ipopt'
    )
    b_run.fit()
    rel_error = (b_run.params_opt - true_params) / true_params
    assert all(rel_error < REL_TOL)


def test_unrecognized_solver(df):
    with pytest.raises(RunException):
        FlipperRun(
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
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        df=df,
        solver_method='scipy'
    )
    b_run.fit()
    assert np.round(b_run.params_opt) == 3


def test_splines(spline_df):
    splines = {
        'x1': {
            'degree': 3,
            'knots_num': 4,
            'knots_type': 'frequency'
        }
    }
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        df=spline_df,
        splines=splines,
        solver_method='ipopt'
    )
    b_run.fit()
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
            'concave': True
        }
    }
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        df=spline_concave_df,
        splines=splines,
        solver_method='ipopt'
    )
    b_run.fit()
    predictions = b_run.predict()
    np.testing.assert_array_almost_equal(
        x=predictions,
        y=spline_concave_df['p'].values,
        decimal=1
    )


def test_spline_constraints_fail(spline_concave_df):
    splines = {
        'x1': {
            'degree': 3,
            'knots_num': 4,
            'knots_type': 'frequency',
            'convex': True
        }
    }
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        df=spline_concave_df,
        splines=splines,
        solver_method='ipopt'
    )
    b_run.fit()
    predictions = b_run.predict()
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(
            x=predictions,
            y=spline_concave_df['p'].values,
            decimal=1
        )


def test_new_data(df, new_df):
    b_run = FlipperRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df,
        solver_method='ipopt'
    )
    b_run.fit()
    predict_1 = b_run.predict()
    predict_2 = b_run.predict(new_df=new_df)

    assert predict_1.shape == predict_2.shape
    assert all(predict_1 != predict_2)
