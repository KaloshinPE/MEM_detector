import numpy as np


def minimize_likelihood(X, y, lmbda, lr=0.01, lr_decay=50, max_iter=100, min_diff=1e-3):
    y = y[:, None]
    alpha = np.random.rand(X.shape[1])
    last_F = None
    for i in range(max_iter):
        data_multipliers = y*X
        t = (data_multipliers @ alpha)[:, None]
        grad_multipliers = np.ones((X.shape[0], 1))
        grad_multipliers[t > -1] = 0.5
        max_values = 1 - grad_multipliers - grad_multipliers*t
        nonzero_mask = max_values > 0

        new_F = np.sum(max_values[nonzero_mask])/X.shape[0] + lmbda*np.linalg.norm(alpha)**2
        if last_F is not None:
            print(i, last_F, abs(new_F - last_F))

            if abs(new_F - last_F) < min_diff:
                break
        last_F = new_F

        grad = - np.sum((grad_multipliers *
                         data_multipliers)[nonzero_mask.repeat(X.shape[1],
                                                               axis=1)],
                        axis=0)/X.shape[0] + 2*lmbda*alpha
        if i > 0 and i % lr_decay == 0:
            lr /= 2
        alpha -= lr*grad

    return alpha


def minimize_likelihood_straight(X, y, lmbda, lr=0.01, lr_decay=50, max_iter=100, min_diff=1e-3):
    y = y[:, None]
    alpha = np.random.rand(X.shape[1])
    last_F = None
    for i in range(max_iter):
        data_multipliers = y*X
        t = data_multipliers @ alpha

        t = np.c_[np.zeros(X.shape[0]), 1/2 - t/2, -t]
        mask = np.argmax(t, axis=1)
        t_t = np.array([elem[i] for i, elem in zip(mask, t)])

        new_F = np.sum(t_t)/X.shape[0] + lmbda*np.linalg.norm(alpha)**2
        if last_F is not None:
            print(i, last_F, abs(new_F - last_F))

            if abs(new_F - last_F) < min_diff:
                break
        last_F = new_F

        grad = np.zeros(X.shape[1])
        for i, ind in enumerate(mask):
            if ind == 1:
                grad -= data_multipliers[i]/2
            if ind == 2:
                grad -= data_multipliers[i]

        if i > 0 and i % lr_decay == 0:
            lr /= 2
        alpha -= lr*grad

    return alpha


class MEM:
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda
        self.alpha = None

    def fit(self, X, y, lr=0.01, lr_decay=50, max_iter=100, min_diff=1e-3):
        X = np.c_[np.ones((X.shape[0])), X]
        y = y[:, None]

        self.alpha = np.random.rand(X.shape[1])

        last_F = None
        for i in range(max_iter):
            data_multipliers = y*X
            t = data_multipliers @ self.alpha

            t = np.c_[np.zeros(X.shape[0]), 1/2 - t/2, -t]
            mask = np.argmax(t, axis=1)
            t_t = np.array([elem[i] for i, elem in zip(mask, t)])

            new_F = np.sum(t_t)/X.shape[0] + self.lmbda*np.linalg.norm(self.alpha)**2
            if last_F is not None:
                if abs(new_F - last_F) < min_diff:
                    break
            last_F = new_F

            grad = np.zeros(X.shape[1])
            for i, ind in enumerate(mask):
                if ind == 1:
                    grad -= data_multipliers[i]/2
                if ind == 2:
                    grad -= data_multipliers[i]

            if i > 0 and i % lr_decay == 0:
                lr /= 2
            self.alpha -= lr*grad

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0])), X]
        return np.sign(X @ self.alpha)
