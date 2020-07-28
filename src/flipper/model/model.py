import numpy as np
from typing import Optional
from anml.models.interface import Model
from anml.data.data import Data

from flipper.data.data import LRSpecs
from flipper.utils import expit


class LRBinomModel(Model):

    def __init__(self, lr_specs: LRSpecs):

        super().__init__()
        self.p_specs = lr_specs.parameter_set

    @property
    def design_matrix(self):
        return self.p_specs.design_matrix_fe

    def _loss(self):
        pass

    @staticmethod
    def _g(m, x, design_matrix):
        return np.sum([
            m[i] * np.log(1 + np.exp(design_matrix[i, :].dot(x)))
            for i in range(len(m))
        ])

    def objective(self, x: np.ndarray, data: Data):
        y = data.data['obs']
        m = data.data['total']
        return self._g(m, x, self.design_matrix) - y.T.dot(self.design_matrix).dot(x)

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
        return self._grad_g(m, x, self.design_matrix) - self.design_matrix.T.dot(y)

    def forward(self, x: np.ndarray, mat: Optional[np.ndarray] = None):
        if mat is None:
            mat = self.design_matrix
        return expit(mat.dot(x))
