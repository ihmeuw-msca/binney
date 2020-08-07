from anml.solvers.composite import CompositeSolver


class ResidualSolver(CompositeSolver):
    def __init__(self):
        super().__init__()

    def fit(self, **kwargs):
        self.solvers.fit(**kwargs)


    def predict(self, **kwargs):
        self.solvers.predict(**kwargs)
