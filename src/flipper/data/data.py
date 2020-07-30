from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from anml.data.data import Data
from anml.data.data import DataSpecs
from anml.parameter.parameter import Parameter, ParameterSet
from anml.parameter.spline_variable import Spline
from anml.parameter.variables import Variable, Intercept
from anml.parameter.processors import process_all
from flipper import FlipperException
from flipper.utils import expit


# The format for data needs to have two columns
# n_total and n_success, unless it is unit record data
# and can have outcomes of 0's and 1's


class BinomDataError(FlipperException):
    pass


@dataclass
class BinomDataSpecs(DataSpecs):

    col_total: str = None

    def __post_init__(self):
        pass


class LRSpecs:
    def __init__(self, col_success: str, col_total: str,
                 covariates: Optional[List[str]] = None,
                 splines: Optional[Dict[str, Dict[str, Any]]] = None):

        self.data_specs = BinomDataSpecs(
            col_obs=col_success,
            col_total=col_total,
        )

        intercept = [Intercept()]
        if covariates is not None:
            covariate_variables = [
                Variable(covariate=cov) for cov in covariates
            ]
        else:
            covariate_variables = list()
        if splines is not None:
            spline_variables = [
                Spline(
                    covariate=spline,
                    **options
                ) for spline, options in splines.items()
            ]
        else:
            spline_variables = list()

        parameter = Parameter(
            param_name='p',
            variables=intercept + covariate_variables + spline_variables,
            link_fun=lambda x: expit(x)
        )
        self.parameter_set = ParameterSet(
            parameters=[parameter]
        )

        self.data = Data(
            data_specs=self.data_specs,
            param_set=self.parameter_set
        )

    def configure_data(self, df):

        self.data.process_data(df=df)
        process_all(self.parameter_set, df)
