from scipy.optimize import minimize
import numpy as np


class DRC:
    def __init__(self, lmbda=0.1, m=2):
        self.lmbda = lmbda
        self.m = m
        self.z = None

    def fit(self, X, y):
        W = []
        d = X.shape[1]
        index_array = np.arange(d)*self.m
        for x in X:
            w = np.zeros(self.m*d)
            w[x + index_array] = 1
            W.append(w)
        W = np.array(W)
        C = y/2

        def loss_fun(z):
                return np.mean((W@z - C)**2) + self.lmbda*np.linalg.norm(z)**2

        self.z = minimize(loss_fun, np.ones(W.shape[1])).x

    def predict(self, X):
        prob1, prob2 = [], []
        index_array = np.arange(X.shape[1])*self.m
        for x in X:
            s = np.sum(self.z[index_array + x])
            prob1.append(1/2 - s)
            prob2.append(1/2 + s)

        prob1, prob2 = np.array(prob1), np.array(prob2)
        denom = prob1**2 + prob2**2
        prob2 = prob2**2/denom
        return (np.round(prob2) - 0.5)*2
