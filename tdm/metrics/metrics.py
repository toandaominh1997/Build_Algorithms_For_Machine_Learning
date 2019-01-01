import autograd.numpy as np

def squared_error(actual, predicted):
    return (actual-predicted)**2

def mean_squared_error(actual, predicted):
    np.mean(squared_error(actual, predicted))
    