from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np

import pandas as pd

from anml.data.data import Data
from anml.data.data import DataSpecs
from anml.parameter.parameter import Parameter, ParameterSet
from anml.parameter.spline_variable import Spline
from anml.parameter.prior import GaussianPrior
from anml.parameter.variables import Variable, Intercept
from anml.parameter.processors import process_all
from binney import BinneyException
from binney.utils import expit
from binney.data.splines import make_spline_variables


# The format for data needs to have two columns
# n_total and n_success, unless it is unit record data
# and can have outcomes of 0's and 1's


class BinomDataError(BinneyException):
    pass


@dataclass
class BinomDataSpecs(DataSpecs):

    col_total: str = None

    def __post_init__(self):
        pass


class LRSpecs:
    def __init__(self, col_success: str, col_total: str,
                 col_group: Optional[str] = None,
                 covariates: Optional[List[str]] = None,
                 splines: Optional[Dict[str, Dict[str, Any]]] = None,
                 coefficient_priors: Optional[List[float]] = None,
                 coefficient_prior_var: Optional[float] = None):
        """
        Specifications for a logistic regression data set and parameters,
        including splines and spline derivative constraints.

        Parameters
        ----------
        col_success
            The column name of the count outcome, or the number of "successes".
        col_total
            The column name of the total, or the number of trials.
        col_group
            Optional grouping column.
        covariates
            List of covariate names to include as fixed effects.
        splines
            A dictionary with spline specifications. Valid options include
            knots_type, knots_num, degree, r_linear (linear tail on right),
            l_linear (linear tail on left), increasing (monotonic increasing constraint),
            decreasing (monotonic decreasing constraint), concave, and convex.
        """

        self.covariates = covariates
        self.splines = splines
        self.parameter_set = None

        if col_group is not None:
            col_groups = [col_group]
        else:
            col_groups = None
        self.data_specs = BinomDataSpecs(
            col_obs=col_success,
            col_total=col_total,
            col_groups=col_groups
        )
        self.make_parameter_set(
            coefficient_priors=coefficient_priors,
            coefficient_prior_var=coefficient_prior_var
        )
        self.data = Data(
            data_specs=self.data_specs,
            param_set=self.parameter_set
        )

    def make_parameter_set(self, coefficient_priors: Optional[List[float]] = None,
                           coefficient_prior_var: Optional[float] = None):
        if coefficient_priors is not None:
            intercept = [Intercept(
                fe_prior=GaussianPrior(
                    mean=[coefficient_priors.pop(0)],
                    std=[coefficient_prior_var**0.5]
                )
            )]
        else:
            intercept = [Intercept()]
        if self.covariates is not None:
            if coefficient_priors is None:
                covariate_variables = [
                    Variable(covariate=cov) for cov in self.covariates
                ]
            else:
                covariate_variables = [
                    Variable(
                        covariate=cov,
                        fe_prior=GaussianPrior(
                            mean=[coefficient_priors.pop(0)],
                            std=[coefficient_prior_var**0.5]
                        )
                    ) for cov in self.covariates
                ]
        else:
            covariate_variables = list()

        spline_variables = list()
        if self.splines is not None:
            spline_variables = make_spline_variables(self.splines,
                                                     coefficient_priors,
                                                     coefficient_prior_var)

        parameter = Parameter(
            param_name='p',
            variables=intercept + covariate_variables + spline_variables,
            link_fun=lambda x: expit(x)
        )
        self.parameter_set = ParameterSet(
            parameters=[parameter]
        )

    def configure_data(self, df: pd.DataFrame):

        self.data.process_data(df=df)
        for var in self.parameter_set.variables:
            var.build_design_matrix_fe(df=df)
            var.build_constraint_matrix_fe()
        process_all(self.parameter_set, df)

    def configure_new_data(self, df: pd.DataFrame):
        """
        Processes a new data frame so that it will create
        a new design matrix.

        Parameters
        ----------
        df
            Data frame with covariates in the original form
            as before, but with a new design matrix.

        Returns
        -------

        """
        for var in self.parameter_set.variables:
            if isinstance(var, Spline):
                var.build_design_matrix_fe(df=df, create_spline=False)
            else:
                var.build_design_matrix_fe(df=df)
            var.build_constraint_matrix_fe()
        process_all(self.parameter_set, df)
