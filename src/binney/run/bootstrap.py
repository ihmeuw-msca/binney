from copy import deepcopy

import numpy as np
import pandas as pd
from binney.data.data import LRSpecs
from binney.model.model import BinomialModel

from anml.bootstrap.bootstrap import Bootstrap


class BinneyBootstrap(Bootstrap):
    def __init__(self, model: BinomialModel, df: pd.DataFrame, **kwargs):

        super().__init__(model=model, **kwargs)
        self.df = df
        self.lr_specs = None

    def attach_specs(self, lr_specs: LRSpecs):
        self.lr_specs = deepcopy(lr_specs)

    def detach_specs(self):
        self.lr_specs = None

    def _process(self, **kwargs):
        raise NotImplementedError()


class BinomialBootstrap(BinneyBootstrap):
    """
    Non-parametric bootstrap implementation for the BinomRun
    modeling process.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


class BernoulliBootstrap(BinneyBootstrap):
    """
    Non-parametric bootstrap implementation for a dataset with 1's and 0's
    in a logistic regression modeling process.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new data frame by sampling from the binomial distribution
        with p = k / n and n = n from the original data, where n is the sample
        size and k is the number of successes.

        Returns
        -------
        data frame with re-sampled observations
        """
        sample_df = df.copy()
        sample_df = sample_df.sample(n=len(sample_df), replace=True)
        return sample_df

    def _process(self, fit_callable, **kwargs):
        new_df = self._sample(df=self.df)
        self.lr_specs.configure_data(df=new_df)
        self.model.detach_specs()
        self.model.attach_specs(self.lr_specs)
        fit_callable(solver=self.solver, data=self.lr_specs.data, **kwargs)


class BernoulliStratifiedBootstrap(BernoulliBootstrap):
    """
    Non-parametric bootstrap implementation for the BinomRun
    modeling process, but with stratified re-sampling for groups
    when there is bernoulli data.
    """
    def __init__(self, col_group, **kwargs):
        super().__init__(**kwargs)
        self.col_group = col_group

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        sample_df = df.copy()
        sample_df = sample_df.groupby(
            self.col_group, group_keys=False
        ).apply(lambda x: x.sample(len(x), replace=True))
        return sample_df
