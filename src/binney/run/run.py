import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from copy import copy

from anml.solvers.interface import Solver
from anml.solvers.base import ScipyOpt, IPOPTSolver
from anml.data.data import Data
from anml.data.data_specs import DataSpecs

from binney.model.bin_model import BinomialModel
from binney.data.data import LRSpecs
from binney.run.bootstrap import BinomialBootstrap, BernoulliBootstrap
from binney import BinneyException
from binney.residuals.residual_model import ResidualModel
from binney.utils import logit, expit


class RunException(BinneyException):
    pass


class BinneyRun:
    def __init__(self, df: pd.DataFrame, col_success: str, col_total: str,
                 covariates: Optional[List[str]] = None,
                 splines: Optional[Dict[str, Dict[str, Any]]] = None,
                 solver_method: str = 'scipy', solver_options: Optional[Dict[str, Any]] = None,
                 data_type: str = 'bernoulli', col_group: Optional[str] = None):
        """
        Create a model run with binney. The model can handle either binomial data or Bernoulli data.
        If you have binomial data, your data will look something like "k successes out
        of n trials" -- binney needs to know both k and n. If you have Bernoulli data,
        your data will look like "individual-record" or "unit-record" data with 1's
        and 0's. The data needs to be in the same form as the bernoulli type, however all
        of the "n trials" will be 1, and then the outcome in "success" is either 1 or 0.
        See the Jupyter notebooks in this repository for an example of Binomial data.
        The model looks like this in either case:

        .. math:: k_i \sim Binomial(n_i, p_i)


        but where :math:`n_i = 1` if you have Bernoulli data.
        The goal is to estimate :math:`p` where :math:`p` is the expit of some linear predictor,
        which may also contain include splines for different covariates. The linear
        predictor will automatically include an intercept, so do not specify one
        in your covariates.

        This run class will create uncertainty with the bootstrap method. The particular
        type of bootstrap re-sampling will depend on whether you have binomial or Bernoulli
        type data. It is not enforced strictly, but **do not mix the two types of data**,
        as it will give inaccurate uncertainty quantification.

        Parameters
        ----------
        df
            A pandas data frame with all of the columns in covariates, splines,
            and col_success and col_total.
        col_success
            The column name of the number of successes (:math:`k`).
        col_total
            The column name of the number of trials (:math:`n`).
        covariates
            A list of column names for covariates to use.
        splines
            A dictionary of spline covariates, each of which is a dictionary
            of spline specifications. For example,

            .. code:: bash

                splines = {
                    'x1': {
                        'knots_type': 'domain',
                        'knots_num': 3,
                        'degree': 3,
                        'convex': True
                    }
                }

            The list of available options for splines is:

            * :code:`knots_type (str)`: type of knots, one of "domain" or "frequency"
            * :code:`knots_num (int)`: number of knots
            * :code:`degree (int)`: degree of the spline
            * :code:`r_linear (bool)`: include linear tails on the right
            * :code:`l_linear (bool)`: include linear tails on the left
            * :code:`increasing (bool)`: impose monotonic increasing constraint on spline shape
            * :code:`decreasing (bool)`: impose monotonic decreasing constraint on spline shape
            * :code:`concave (bool)`: impose concavity constraint on spline shape
            * :code:`convex (bool)`: impose convexity constraint on spline shape

        solver_method
            Type of solver to use, one of "ipopt" (interior point optimizer -- use this if
            you have spline shape constraints), or "scipy".
        solver_options
            A dictionary of options to pass to your desired solver.
        data_type
            The data type: one of "bernoulli" or "binomial"
        col_group
            The group column. Not used. If you want to model with random effects, use :class:`BinneyResidualRun`.

        Attributes
        ----------
        self.params_init
            Initial parameters for the optimization
        self.params_opt
            Optimal parameters found through the fitting process
        self.bootstrap
            Bootstrap class that creates uncertainty. After running the
            :code:`BinneyRun.make_uncertainty()` method you can access
            the parameter estimates across bootstrap replicates in
            :code:`self.bootstrap.parameters`.
        """

        # Check the data type
        if data_type not in ['bernoulli', 'binomial']:
            raise BinneyException(f"Data type must be one of 'bernoulli' or 'binomial'. "
                                   f"Got {data_type}.")
        self.data_type = data_type

        # Configure the data specs
        self.lr_specs = LRSpecs(
            col_success=col_success,
            col_total=col_total,
            col_group=col_group,
            covariates=covariates,
            splines=splines
        )
        self.lr_specs.configure_data(df=df)

        # Set up the model
        self.model = BinomialModel()
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

    def fit(self) -> None:
        """
        Fit the binney model after initialization.
        Optimal parameters are stored in BinneyRun.params_opt.
        """
        self._fit(solver=self.solver, data=self.lr_specs.data)
        self.params_opt = copy(self.solver.x_opt)

    def predict(self, new_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions based on optimal parameter values.

        Parameters
        ----------
        new_df
            A pandas data frame to make predictions for. Must have all of the covariates
            used in the fitting.

        Returns
        -------
        A numpy array of predictions for the data frame.
        """
        if new_df is None:
            return self.model.forward(self.params_opt)
        else:
            self.lr_specs.configure_new_data(df=new_df)
            return self.model.forward(self.params_opt, mat=self.lr_specs.parameter_set.design_matrix_fe)

    def predict_draws(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make draws based on the bootstrap parameters.

        Parameters
        ----------
        df
            A pandas data frame to make predictions for. Must have all of the covariates
            used in the fitting.

        Returns
        -------
        A stacked numpy array of draws for each row in the :code:`df`.
        """
        draw_matrix = []
        for i in range(self.bootstrap.parameters.shape[0]):
            self.bootstrap.lr_specs.configure_new_data(df=df)
            draws = self.model.forward(x=self.bootstrap.parameters[i, :],
                                       mat=self.bootstrap.lr_specs.parameter_set.design_matrix_fe)
            draw_matrix.append(draws)
        draw_matrix = np.vstack(draw_matrix)
        return draw_matrix

    def make_uncertainty(self, n_boots: int = 100):
        """
        Runs bootstrap re-sampling to get uncertainty
        in the parameters. Access parameters in
        :code:`self.bootstrap.parameters`.

        Parameters
        ----------
        n_boots
            Number of bootstrap replicates
        """
        self.bootstrap.run_bootstraps(
            n_bootstraps=n_boots,
            fit_callable=self._fit
        )


class BinneyResidualsRun(BinneyRun):
    def __init__(self, col_group: str, **kwargs):
        super().__init__(**kwargs)

        self.residual_specs = DataSpecs(col_obs=self._col_residual, col_groups=[col_group])
        self.residual_model = ResidualModel(data_specs=self.residual_specs)

    @property
    def _col_residual(self):
        return 'residual'

    @property
    def _col_group(self):
        return self.residual_model.col_group

    @property
    def _logit_p_obs(self) -> np.ndarray:
        return logit(self.lr_specs.data['obs'] / self.lr_specs.data['total'])

    @property
    def _groups(self) -> np.ndarray:
        return self.lr_specs.data['groups']

    @property
    def _logit_p_hat(self) -> np.ndarray:
        return logit(self._predict())

    @property
    def _residuals(self):
        return self._logit_p_obs - self._logit_p_hat

    def predict(self, new_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions based on optimal parameter values.

        Parameters
        ----------
        new_df
            A pandas data frame to make predictions for. Must have all of the covariates
            used in the fitting.

        Returns
        -------
        A numpy array of predictions for the data frame.
        """
        p_hat = self._predict(new_df=new_df)
        p_obs = new_df[self.lr_specs.data_specs.col_obs] / new_df[self.lr_specs.data_specs.col_total]

        residual_df = pd.DataFrame({
            self._col_residual: logit(p_obs) - logit(p_hat),
            self._col_group: new_df[self._col_group].values
        })

        self.residual_model.fit_model()