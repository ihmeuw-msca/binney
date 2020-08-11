import numpy as np

from binney.solvers.solver import ScipySolver
from binney.solvers.hierarchical_solver import Hierarchy
from binney.data.data import LRSpecs
from binney.model.model import BinomialModel


def test_hierarchy(group_data):
    lr_specs = LRSpecs(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        col_group='g'
    )
    lr_specs.configure_data(df=group_data)
    model = BinomialModel()
    model.attach_specs(lr_specs)
    solver = ScipySolver(model_instance=model)
    solver.attach_lr_specs(lr_specs)
    h = Hierarchy(solver=solver, coefficient_prior_var=1.)
    assert h.coefficient_prior_var == 1.
    options = {'solver_options': {}}
    params_init = np.zeros(2)

    h.fit(x_init=params_init, options=options, data=lr_specs.data)
    intercepts = []
    slopes = []
    us = group_data.u.unique().tolist()
    for key, value in h.x_opt.items():
        intercepts.append(value[0])
        slopes.append(value[1])

    u_hat = intercepts - np.mean(intercepts)
    np.testing.assert_array_almost_equal(u_hat, us, decimal=1)
    predictions = h.predict(new_df=group_data)
    np.testing.assert_array_almost_equal(
        predictions,
        group_data['p'],
        decimal=1
    )


def test_hierarchy_tight(group_data):
    lr_specs = LRSpecs(
        col_success='success',
        col_total='total',
        covariates=['x1'],
        col_group='g'
    )
    lr_specs.configure_data(df=group_data)
    model = BinomialModel()
    model.attach_specs(lr_specs)
    solver = ScipySolver(model_instance=model)
    solver.attach_lr_specs(lr_specs)
    h = Hierarchy(solver=solver, coefficient_prior_var=1e-5)
    assert h.coefficient_prior_var == 1e-5
    options = {'solver_options': {}}
    params_init = np.zeros(2)

    h.fit(x_init=params_init, options=options, data=lr_specs.data)
    intercepts = []
    slopes = []
    for key, value in h.x_opt.items():
        intercepts.append(value[0])
        slopes.append(value[1])
    u_hat = intercepts - np.mean(intercepts)
    np.testing.assert_almost_equal(u_hat, np.zeros(u_hat.shape), decimal=1)
