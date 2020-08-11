from typing import Optional, Dict
import numpy as np
import pandas as pd
from copy import copy

from anml.solvers.composite import CompositeSolver
from anml.data.data import Data

from binney.solvers.solver import Base


class Hierarchy(CompositeSolver):

    def __init__(self, solver: Base, coefficient_prior_var: float):
        """
        Hierarchical solver that first solves the problem with
        all of the data, then uses those fixed effects as priors
        for group-specific models.

        Parameters
        ----------
        solver
            Any solver.
        coefficient_prior_var
            Variance of the prior to pass down to the group-specific
            models.
        """
        super().__init__([solver])

        self.coefficient_prior_var = coefficient_prior_var
        self.x_opt = dict()

    def _cache_result(self):
        return copy(self.solvers[0].x_opt).tolist()

    @property
    def lr_specs(self):
        return self.solvers[0].lr_specs

    def fit(self, x_init: np.ndarray, data: Data, **kwargs):
        self.solvers[0].fit(x_init=x_init, data=data, **kwargs)
        prior = self._cache_result()
        self.lr_specs.make_parameter_set(
            coefficient_priors=prior,
            coefficient_prior_var=self.coefficient_prior_var
        )
        unique_groups = np.unique(data.data['groups'].ravel())
        group_indices = data.data['groups'].ravel()
        df = data._df.copy()
        for group in unique_groups:
            group_index = np.where(group_indices == group)[0]
            self.lr_specs.configure_data(df=df.iloc[group_index])
            self.solvers[0].model.attach_specs(self.lr_specs)
            self.solvers[0].fit(x_init=self._cache_result(), data=self.lr_specs.data, **kwargs)
            self.x_opt[group] = self._cache_result()

    def predict(self, new_df: pd.DataFrame, x: Optional[Dict[str, np.ndarray]] = None):
        if x is None:
            x = self.x_opt
        predictions = np.empty(len(new_df))
        self.lr_specs.configure_data(new_df)
        unique_groups = np.unique(self.lr_specs.data.data['groups'].ravel())
        group_indices = self.lr_specs.data.data['groups'].ravel()
        for group in unique_groups:
            group_index = np.where(group_indices == group)[0]
            if group not in self.x_opt:
                raise RuntimeError(f"Could not find group identifier {group} in the original fit."
                                   f"Available groups are {self.x_opt.keys()}.")
            group_preds = self.solvers[0].predict(
                x=x[group],
                new_df=self.lr_specs.data._df.iloc[group_index]
            )
            predictions[group_indices == group] = group_preds

        return predictions
