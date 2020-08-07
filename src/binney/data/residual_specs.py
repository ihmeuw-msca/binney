from typing import Optional
import pandas as pd

from anml.data.data import DataSpecs
from anml.data.data import Data
from anml.parameter.variables import Intercept
from anml.parameter.parameter import Parameter, ParameterSet
from anml.parameter.prior import GaussianPrior
from anml.parameter.processors import process_all


class ResidualSpecs:
    def __init__(self, col_group: str, col_obs_se: Optional[str] = None,
                 re_var: float = 1.):
        super().__init__()

        self.specs = DataSpecs(
            col_obs=self._col_obs,
            col_obs_se=col_obs_se,
            col_groups=[col_group]
        )

        self.re_prior = GaussianPrior(mean=[0], std=[re_var ** 0.5])
        self.intercept = Intercept(re_prior=self.re_prior)
        self.parameter = Parameter(variables=[self.intercept], param_name='mean')
        self.parameter_set = ParameterSet([self.parameter])

        self.data = Data(
            data_specs=self.specs,
            param_set=self.parameter_set
        )

    @property
    def _col_obs(self):
        return 'residual'

    def configure_data(self, df: pd.DataFrame):

        self.data.process_data(df=df)
        self.parameter.variables[0].build_design_matrix_re(df)
        process_all(self.parameter_set, df)
