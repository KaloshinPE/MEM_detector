import numpy as np


def minimize_likelihood(X, y, lmbda, lr=0.01, max_iter=100, min_diff=1e-3):
    alpha = np.random.rand(X.shape[1])
    last_F = None
    for i in range(max_iter):
        data_multipliers = y*X
        t = data_multipliers @ alpha
        grad_multipliers = np.ones(X.shape[0])
        grad_multipliers[t > -1] = 0.5
        max_values = 1 - grad_multipliers - grad_multipliers*t
        nonzero_mask = max_values > 0

        new_F = np.sum(max_values[nonzero_mask])
        if last_F is not None and abs(new_F - last_F) < min_diff:
            break

        grad = - np.sum((grad_multipliers * data_multipliers)[nonzero_mask],
                        axis=0)/X.shape[0] + 2*lmbda*alpha
        alpha -= lr*grad

    return alpha
