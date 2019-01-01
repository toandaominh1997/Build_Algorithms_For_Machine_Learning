import tdm.base import BaseEstimator

class BaseRegression(BaseEstimator):
    def __init__(self, lr=0.01, penatly='None', C=0.01, tolerance=0.0001, max_iters=1000):
        self.C = C
        self.lr = lr
        self.penatly=penatly
        self.max_iters = max_iters
        self.theta = []
        self.errors = []
        self.n_samples, self.n_features = None, None


    def fit(self, X, y=None):
        pass
    def gradient_descent(self):
        theta = self.theta
