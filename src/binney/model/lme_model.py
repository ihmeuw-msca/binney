import numpy as np
from typing import Optional

from anml.data.data import Data
from anml.models.interface import Model

from binney.data.residual_specs import ResidualSpecs


class LMEModel(Model):

    def __init__(self):
        """
        Random intercept model.
        """
        super().__init__()
        self.r_specs = None

    def attach_specs(self, r_specs: ResidualSpecs):
        self.r_specs = r_specs.parameter_set.re_prior

    @property
    def design_matrix(self):
        return self.r_specs.design_matrix_re

    @property
    def re_var(self):
        return self.r_specs.parameter_set.variables[0].std ** 2

    def objective(self, x: np.ndarray, data: Data) -> float:
        y = data.data['obs']
        se = data.data['obs_se']
        prediction = self.design_matrix.dot(x)

        val = 0.
        val += 0.5 * np.sum(((y - prediction)/se)**2)
        val += self.r_specs.parameter_set.variables[0].re_prior.error_value(x)
        return val

    def gradient(self, x: np.ndarray, data: Data):
        y = data.data['obs']
        se = data.data['obs_se']
        prediction = self.design_matrix.dot(x)
        residual = y - prediction

        grad = 0
        grad += -(self.design_matrix.T/se).dot(residual/se)
        grad += ((x - np.mean(x))/self.re_var)
        return grad

    def forward(self, x: np.ndarray, mat: Optional[np.ndarray] = None):
        if mat is None:
            mat = self.design_matrix
        return mat.dot(x)
