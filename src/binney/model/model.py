import numpy as np
from typing import Optional
from anml.models.interface import Model
from anml.data.data import Data
from anml.parameter.utils import build_linear_constraint

from binney.data.data import LRSpecs
from binney.utils import expit


class BinomialModel(Model):

    def __init__(self):

        super().__init__()
        self.lr_specs = None
        self.C = None
        self.c_lb = None
        self.c_ub = None

    @property
    def parameter_set(self):
        return self.lr_specs.parameter_set

    def attach_specs(self, lr_specs: LRSpecs):
        self.lr_specs = lr_specs
        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self.parameter_set.constr_matrix_fe,
             self.parameter_set.constr_lb_fe,
             self.parameter_set.constr_ub_fe),
        ])

    def detach_specs(self):
        self.lr_specs = None
        self.C = None
        self.c_lb = None
        self.c_ub = None

    @property
    def design_matrix(self):
        return self.parameter_set.design_matrix_fe

    @staticmethod
    def _g(m, x, design_matrix):
        return np.sum([
            m[i] * np.log(1 + np.exp(design_matrix[i, :].dot(x)))
            for i in range(len(m))
        ])

    def objective(self, x: np.ndarray, data: Data):
        y = data.data['obs']
        m = data.data['total']

        val = 0.
        val += self._g(m, x, self.design_matrix) - y.T.dot(self.design_matrix).dot(x)

        i = 0
        for variable in self.parameter_set.variables:
            val += np.sum(variable.fe_prior.error_value(x[i:i + variable.num_fe]))
            i += variable.num_fe
        return val

    @staticmethod
    def _grad_g(m, x, design_matrix):
        inner = np.array([
            expit(design_matrix[i, :].T.dot(x))
            for i in range(len(m))
        ])
        return design_matrix.T.dot(np.diag(m)).dot(inner)

    def gradient(self, x: np.ndarray, data: Data):
        y = data.data['obs']
        m = data.data['total']
        val = 0.
        val += self._grad_g(m, x, self.design_matrix) - self.design_matrix.T.dot(y)

        i = 0
        for variable in self.parameter_set.variables:
            val += np.sum(variable.fe_prior.grad(x[i:i+variable.num_fe]))
            i += variable.num_fe
        return val

    def forward(self, x: np.ndarray, mat: Optional[np.ndarray] = None):
        if mat is None:
            mat = self.design_matrix
        return expit(mat.dot(x))
