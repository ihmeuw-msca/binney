import numpy as np

from flipper.data.data import LRSpecs
from flipper.model.model import LRBinomModel


def test_lr_binom_model_simple(simple_df):
    specs = LRSpecs(
        col_success='success',
        col_total='total'
    )
    specs.configure_data(simple_df)
    model = LRBinomModel()
    model.attach_specs(lr_specs=specs)
    objective = model.objective(x=np.array([2]), data=specs.data)
    grad = model.gradient(x=np.array([2]), data=specs.data)
    assert isinstance(objective, float)
    assert grad.shape == (1,)


def test_lr_binom_one_cov(df):
    specs = LRSpecs(
        col_success='success',
        col_total='total',
        covariates=['x1']
    )
    specs.configure_data(df)
    model = LRBinomModel()
    model.attach_specs(lr_specs=specs)
    objective = model.objective(x=np.array([0, 2]), data=specs.data)
    grad = model.gradient(x=np.array([0, 2]), data=specs.data)
    assert isinstance(objective, float)
    assert grad.shape == (2,)
