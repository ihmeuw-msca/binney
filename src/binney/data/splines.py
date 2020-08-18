import numpy as np
from typing import Dict, Union, Optional, Any, List

from anml.parameter.spline_variable import SplineLinearConstr, Spline
from anml.parameter.prior import GaussianPrior
from binney import BinneyException


VALID_SPLINE_OPTIONS = {
    'knots_type': str,
    'knots_num': int,
    'degree': int,
    'r_linear': bool,
    'l_linear': bool,
    'increasing': bool,
    'decreasing': bool,
    'concave': bool,
    'convex': bool
}


VALID_SPLINE_CONSTR_OPTIONS = [
    'increasing', 'decreasing',
    'concave', 'convex',
]


SPLINE_CONSTR_DICT = {
    'increasing': SplineLinearConstr(order=1, y_bounds=[0.0, np.inf], grid_size=20),
    'decreasing': SplineLinearConstr(order=1, y_bounds=[-np.inf, 0.0], grid_size=20),
    'concave': SplineLinearConstr(order=2, y_bounds=[-np.inf, 0.0], grid_size=20),
    'convex': SplineLinearConstr(order=2, y_bounds=[0.0, np.inf], grid_size=20)
}


def make_spline_variables(splines: Dict[str, Dict[str, Any]],
                          coefficient_priors: Optional[List[float]] = None,
                          coefficient_prior_var: Optional[float] = None) -> List[Spline]:
    """
    Creates spline variables with optional coefficient priors and shape constraints.

    Parameters
    ----------
    splines
    coefficient_priors
    coefficient_prior_var

    Returns
    -------
    List of spline variables that can be used in a parameter.
    """
    spline_variables = []
    for spline, options in splines.items():
        spline_constraints = list()
        for option, value in options.copy().items():
            if not type(value) == VALID_SPLINE_OPTIONS[option]:
                raise BinneyException(
                    f"Invalid type of spline option {option}."
                    f"Got type {type(value)}, expected {VALID_SPLINE_OPTIONS[option]}."
                )
            if option in VALID_SPLINE_CONSTR_OPTIONS.copy():
                options.pop(option)
                if value:
                    constraint = SPLINE_CONSTR_DICT[option]
                    spline_constraints.append(constraint)
        spline_variable = Spline(
            covariate=spline,
            **options,
            derivative_constr=spline_constraints
        )

        if coefficient_priors is not None:
            num_fe = spline_variable._count_num_fe()
            priors = coefficient_priors[0:num_fe]
            coefficient_priors = coefficient_priors[num_fe:]
            fe_prior = GaussianPrior(
                mean=priors,
                std=[coefficient_prior_var] * num_fe
            )
            spline_variable.fe_prior = fe_prior
        spline_variables.append(spline_variable)
    return spline_variables
