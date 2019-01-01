/*
 * @Author: bigkizd(Toan Dao Minh) 
 * @Date: 2019-01-01 22:36:45 
 * @Last Modified by: bigkizd(Toan Dao Minh)
 * @Last Modified time: 2019-01-01 23:06:01
 */
from tdm.base import BaseEstimator
from tdm.metrics.metrics import mean_squared_error 
class BaseRegression(BaseEstimator):
    def __init__(self, lr=0.01, penatly='None', C=0.01, tolerance=0.0001, max_iters=1000):
        self.C = C
        self.lr = lr
        self.penatly=penatly
        self.max_iters = max_iters
        self.theta = []
        self.errors = []
        self.n_samples, self.n_features = None, None

    def init_cost(self):
        raise NotImplementedError()
    def train(self):
        self.theta, self.errors = self.gradient_descent()
    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = X.shape
        self.theta = np.random.normal(size=(self.n_features+1), scale=0.5)
        self.X = self._add_intercept(self.X)
        self.train()
        
    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)
    
    
    def cost(self, X, y, theta):
        prediction = X.dot(theta)
        error = self.cost_func(y, prediction)
        return error
    def loss(self, w):
        raise NotImplementedError()
    def gradient_descent(self):
        theta = self.theta
        errors = [self.cost(self.X, self.y, theta)]
        cost_d = grad(self.loss)
        for i in range(1, self.max_iters+1):
            delta = cost_d(theta)
            theta -=self.lr*delta
            errors.append(self.cost(self.X, self.y, theta))
            error_diff = np.linalg.norm(errors[i-1], errors[i])
            if(error_diff< self.tolerance):
                break
        return theta, errors
    def _add_penalty(self, loss, w):
        if(self.penatly=='l1'):
            loss += self.C*np.abs(w[1:]).sum()
        elif(self.penatly=='l2'):
            loss+=(0.5*self.C)*(w[1:]**2).sum()
        return loss
    
        
class LinearRegression(BasicRegression):
    def loss(self, w):
        loss = self.cost_func(self.y, np.dot(self.X, w))
        return self._add_penalty(loss)
    def init_cost(self):
        self.cost_func = mean_squared_error
    