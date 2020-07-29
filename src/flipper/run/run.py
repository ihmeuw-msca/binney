import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

from anml.solvers.base import ScipyOpt, IPOPTSolver

from flipper.model.model import LRBinomModel
from flipper.data.data import LRSpecs
from flipper import FlipperException


class RunException(FlipperException):
    pass


class BinomRun:
    def __init__(self, df: pd.DataFrame, col_success: str, col_total: str,
                 covariates: Optional[List[str]] = None,
                 splines: Optional[List[Dict[str, Any]]] = None,
                 solver_method: str = 'scipy'):

        self.lr_specs = LRSpecs(
            col_success=col_success,
            col_total=col_total,
            covariates=covariates,
            splines=splines
        )
        self.lr_specs.configure_data(df=df)

        self.model = LRBinomModel(lr_specs=self.lr_specs)

        if solver_method == 'scipy':
            self.solver = ScipyOpt(self.model)
        elif solver_method == 'ipopt':
            self.solver = IPOPTSolver(self.model)
        else:
            raise RunException(f"Unrecognized solver method {solver_method}."
                               "Please pass one of 'scipy' or 'ipopt'.")

    def fit(self, x_init: Union[np.ndarray, List[float]],
            solver_options: Optional[Dict[str, Any]] = None,
            method: str = 'L-BFGS-B'):
        if solver_options is None:
            solver_options = dict()
        self.solver.fit(
            x_init=x_init, options={
                'solver_options': solver_options,
                'method': method
            },
            data=self.lr_specs.data
        )

    def predict(self, **kwargs):
        return self.solver.predict(**kwargs)
