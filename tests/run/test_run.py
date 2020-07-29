import numpy as np

from flipper.run.run import BinomRun

REL_TOL = 1e-2


def test_lr_specs(df, intercept, slope):
    true_params = [intercept, slope]
    b_run = BinomRun(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        df=df
    )
    b_run.fit(x_init=[1., 1.])
    rel_error = (b_run.solver.x_opt - true_params) / true_params
    assert all(rel_error < REL_TOL)
