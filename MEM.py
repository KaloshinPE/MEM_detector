import numpy as np


class MEM:
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda
        self.alpha = None

    def fit(self, X, y, lr=0.01, lr_decay=300, max_iter=5000, min_diff=1e-5):
        X = np.c_[np.ones((X.shape[0])), X]
        y = y[:, None]

        self.alpha = np.random.rand(X.shape[1])

        last_F = None
        for i in range(max_iter):
            data_multipliers = y*X
            t = (data_multipliers @ self.alpha)[:, None]
            grad_multipliers = np.ones((X.shape[0], 1))
            grad_multipliers[t > -1] = 0.5
            max_values = 1 - grad_multipliers - grad_multipliers*t
            nonzero_mask = max_values > 0

            new_F = np.sum(max_values[nonzero_mask])/X.shape[0] +\
                                self.lmbda*np.linalg.norm(self.alpha)**2
            if last_F is not None and abs(new_F - last_F) < min_diff:
                break
            last_F = new_F

            grad = - np.sum(np.array([elem*m for elem, m in
                                      zip(grad_multipliers*data_multipliers,
                                          nonzero_mask)]), axis=0) +\
                                                    2*self.lmbda*self.alpha

            if i > 0 and i % lr_decay == 0:
                lr /= 2
            self.alpha -= lr*grad

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0])), X]
        return np.sign(X @ self.alpha)

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0])), X]
        z = X @ self.alpha
        t = np.clip(0.5 + z/2, 0, 1)
        return t
