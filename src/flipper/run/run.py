import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

from anml.solvers.base import ScipyOpt

from flipper.model.model import LRBinomModel
from flipper.data.data import LRSpecs


class BinomRun:
    def __init__(self, df: pd.DataFrame, col_success: str, col_total: str,
                 covariates: Optional[List[str]] = None,
                 splines: Optional[List[Dict[str, Any]]] = None):

        self.lr_specs = LRSpecs(
            col_success=col_success,
            col_total=col_total,
            covariates=covariates,
            splines=splines
        )
        self.lr_specs.configure_data(df=df)
        self.model = LRBinomModel(lr_specs=self.lr_specs)
        self.solver = ScipyOpt(self.model)

    def fit(self, x_init: Union[np.ndarray, List[float]]):
        self.solver.fit(
            x_init=x_init, options={'solver_options': {}},
            data=self.lr_specs.data
        )
