import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from copy import copy

from anml.solvers.interface import Solver
from anml.solvers.base import ScipyOpt, IPOPTSolver
from anml.data.data import Data

from flipper.model.model import LRBinomModel
from flipper.data.data import LRSpecs
from flipper.run.bootstrap import BinomialBootstrap, BernoulliBootstrap
from flipper import FlipperException


class RunException(FlipperException):
    pass


class FlipperRun:
    def __init__(self, df: pd.DataFrame, col_success: str, col_total: str,
                 covariates: Optional[List[str]] = None,
                 splines: Optional[Dict[str, Dict[str, Any]]] = None,
                 solver_method: str = 'scipy', solver_options: Optional[Dict[str, Any]] = None,
                 data_type: str = 'bernoulli'):

        # Check the data type
        if data_type not in ['bernoulli', 'binomial']:
            raise FlipperException(f"Data type must be one of 'bernoulli' or 'binomial'. "
                                   f"Got {data_type}.")
        self.data_type = data_type

        # Configure the data specs
        self.lr_specs = LRSpecs(
            col_success=col_success,
            col_total=col_total,
            covariates=covariates,
            splines=splines
        )
        self.lr_specs.configure_data(df=df)

        # Set up the model
        self.model = LRBinomModel()
        self.model.attach_specs(lr_specs=self.lr_specs)

        # Set up the solver
        if solver_method == 'scipy':
            self.solver = ScipyOpt(self.model)
        elif solver_method == 'ipopt':
            self.solver = IPOPTSolver(self.model)
        else:
            raise RunException(f"Unrecognized solver method {solver_method}."
                               "Please pass one of 'scipy' or 'ipopt'.")
        if solver_options is None:
            solver_options = dict()
        self.options = {
            'solver_options': solver_options
        }

        # Configure bootstrap object based on
        # the data type
        if data_type == 'bernoulli':
            self.bootstrap = BernoulliBootstrap(
                solver=self.solver, model=self.model, df=df
            )
        elif data_type == 'binomial':
            self.bootstrap = BinomialBootstrap(
                solver=self.solver, model=self.model, df=df
            )
        self.bootstrap.attach_specs(lr_specs=self.lr_specs)

        # Placeholders for parameters and initial values
        self.params_init = np.zeros(self.model.design_matrix.shape[1])
        self.params_opt = None

    def _fit(self, solver: Solver, data: Data):
        solver.fit(
            x_init=self.params_init, options=self.options, data=data
        )

    def fit(self):
        self._fit(solver=self.solver, data=self.lr_specs.data)
        self.params_opt = copy(self.solver.x_opt)

    def predict(self, new_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        if new_df is None:
            return self.model.forward(self.params_opt)
        else:
            self.lr_specs.configure_new_data(df=new_df)
            return self.model.forward(self.params_opt, mat=self.lr_specs.parameter_set.design_matrix_fe)

    def predict_draws(self, df: pd.DataFrame) -> np.ndarray:
        draw_matrix = []
        for i in range(self.bootstrap.parameters.shape[0]):
            self.bootstrap.lr_specs.configure_new_data(df=df)
            draws = self.model.forward(x=self.bootstrap.parameters[i, :],
                                       mat=self.bootstrap.lr_specs.parameter_set.design_matrix_fe)
            draw_matrix.append(draws)
        draw_matrix = np.vstack(draw_matrix)
        return draw_matrix

    def make_uncertainty(self, n_boots: int = 100, **kwargs):
        """
        Runs bootstrap re-sampling to get uncertainty
        in the parameters.

        Parameters
        ----------
        n_boots
        kwargs
        """
        self.bootstrap.run_bootstraps(
            n_bootstraps=n_boots,
            fit_callable=self._fit,
            **kwargs
        )
