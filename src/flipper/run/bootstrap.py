from copy import deepcopy

import numpy as np
import pandas as pd
from flipper.data.data import LRSpecs
from flipper.model.model import LRBinomModel

from anml.bootstrap.nonparametric import NPBootstrap


class BinomBootstrap(NPBootstrap):
    """
    Non-parametric bootstrap implementation for the BinomRun
    modeling process.
    """
    def __init__(self, model: LRBinomModel, df: pd.DataFrame, **kwargs):

        super().__init__(model=model, **kwargs)
        self.df = df
        self.lr_specs = None

    def attach_specs(self, lr_specs: LRSpecs):
        self.lr_specs = deepcopy(lr_specs)

    def detach_specs(self):
        self.lr_specs = None

    @staticmethod
    def _sample(df: pd.DataFrame, col_obs: str, col_total: str) -> pd.DataFrame:
        """
        Creates a new data frame by sampling from the binomial distribution
        with p = k / n and n = n from the original data, where n is the sample
        size and k is the number of successes.

        Returns
        -------
        data frame with re-sampled observations
        """
        sample_df = df.copy()
        p = sample_df[col_obs] / sample_df[col_total]
        sample_df[col_obs] = np.random.binomial(n=sample_df[col_total], p=p)
        return sample_df

    def _process(self, fit_callable, **kwargs):
        new_df = self._sample(
            df=self.df,
            col_obs=self.lr_specs.data_specs.col_obs,
            col_total=self.lr_specs.data_specs.col_total
        )
        self.lr_specs.configure_data(df=new_df)
        self.model.detach_specs()
        self.model.attach_specs(self.lr_specs)
        fit_callable(solver=self.solver, data=self.lr_specs.data, **kwargs)
