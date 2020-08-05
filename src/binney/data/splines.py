import numpy as np

from anml.parameter.spline_variable import SplineLinearConstr, Spline
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


def make_spline_variables(splines):
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
        spline_variables.append(spline_variable)
    return spline_variables
