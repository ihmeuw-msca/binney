import numpy as np
import pandas as pd

from slime.core import MRData
from slime.model import CovModel, MRModel, CovModelSet

from anml.data.data_specs import DataSpecs

from binney import BinneyException


class ResidualModelError(BinneyException):
    pass


class ResidualModel:
    def __init__(self, data_specs: DataSpecs):

        self.residual_specs = data_specs

        self.mr_data = None
        self.cov_set = None
        self.mr_model = None

        self.gamma = None
        self.params = None

    @property
    def col_group(self):
        return self.residual_specs.col_groups[0]

    def init_model(self, df: pd.DataFrame):

        df['intercept'] = 1
        self.mr_data = MRData(
            df=df,
            col_group=self.col_group,
            col_obs=self.residual_specs.col_obs,
            col_covs=['intercept']
        )
        self.cov_set = CovModelSet([
            CovModel(col_cov='intercept', use_re=True)
        ])
        self.mr_model = MRModel(
            self.mr_data, self.cov_set
        )

    def fit_model(self):

        self.mr_model.fit_model()
        self.params = self.mr_model.result
        self.gamma = np.array(list(self.params.values())).ravel().var()

    @property
    def result(self):
        return self.mr_model.result

    def predict(self, new_df: pd.DataFrame, noise: bool = False):

        if self.col_group not in new_df.columns:
            raise ResidualModelError("New data frame must have "
                                     f"a group column {self.col_group}.")

        groups = new_df[self.col_group].unique()
        missing = [g for g in groups if g not in self.mr_model.result]

        if noise:
            result = self.mr_model.sample_soln()
        else:
            result = self.result
        for k, v in result.items():
            result.update({
                k: v.ravel()[0]
            })

        for m in missing:
            if not noise:
                missing_re = 0
            else:
                missing_re = np.random.normal(loc=0, scale=self.gamma ** 2)
            result.update({
                m: missing_re
            })

        predictions = new_df[self.col_group].apply(result).values
        return predictions

