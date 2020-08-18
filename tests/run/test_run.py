import numpy as np
import pytest
import pandas as pd

from binney.run.run import BinneyRun, RunException

REL_TOL = 1e-2


def test_scipy_opt(df, intercept, slope):
    true_params = [intercept, slope]
    b_run = BinneyRun(
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
    b_run = BinneyRun(
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
        BinneyRun(
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
    b_run = BinneyRun(
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
    b_run = BinneyRun(
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
    b_run = BinneyRun(
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
    b_run = BinneyRun(
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
    b_run = BinneyRun(
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


def test_hierarchy_run(group_data):
    b_run = BinneyRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=group_data,
        solver_method='ipopt'
    )
    b_run.fit()
    preds = b_run.predict(new_df=group_data)
    b_run_grp = BinneyRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=group_data,
        solver_method='ipopt',
        col_group='g',
        coefficient_prior_var=5.
    )
    b_run_grp.fit()
    preds_grp = b_run_grp.predict(new_df=group_data)
    res = preds - group_data['p']
    res_grp = preds_grp - group_data['p']
    assert (np.abs(res) > np.abs(res_grp)).mean() >= 0.975


def test_hierarchy_spline(group_data_2):
    b_run_grp = BinneyRun(
        col_success='success',
        col_total='total',
        splines={
            'x1': {
                'degree': 3,
                'knots_type': 'frequency',
                'knots_num': 3
            },
            'x2': {
                'degree': 3,
                'knots_type': 'frequency',
                'knots_num': 3
            }
        },
        df=group_data_2,
        solver_method='ipopt',
        col_group='g',
        coefficient_prior_var=5.,
        solver_options={'max_iter': 100}
    )
    b_run_grp.fit()
    b_run_grp.predict(group_data_2)
