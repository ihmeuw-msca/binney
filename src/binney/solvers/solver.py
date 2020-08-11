import numpy as np
from typing import Optional
import pandas as pd

from anml.solvers.interface import Solver
from anml.solvers.base import ScipyOpt, IPOPTSolver

from binney.data.data import LRSpecs


class Base(Solver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lr_specs = None

    def attach_lr_specs(self, lr_specs: LRSpecs):
        self.lr_specs = lr_specs

    def detach_lr_specs(self):
        self.lr_specs = None

    def predict(self, x: Optional[np.ndarray] = None,
                new_df: Optional[pd.DataFrame] = None, **kwargs) -> np.ndarray:
        if x is None:
            x = self.x_opt
        if new_df is None:
            return self.model.forward(x)
        else:
            self.lr_specs.configure_new_data(df=new_df)
            return self.model.forward(
                x,
                mat=self.lr_specs.parameter_set.design_matrix_fe
            )


class ScipySolver(Base, ScipyOpt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IpoptSolver(Base, IPOPTSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
