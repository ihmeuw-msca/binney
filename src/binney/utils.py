import numpy as np


def expit(x):
    return np.exp(x) / (1 + np.exp(x))


def logit(x):
    return np.log(x / (1 - x))
