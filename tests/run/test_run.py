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


@pytest.mark.skip("No constraint matrix implemented"
                  "for ipopt to work.")
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
