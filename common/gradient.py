import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not itr.finished:
        idx = itr.multi_index
        tmp = x[idx]
        x[idx] = float(tmp) + h
        fh1 = f(x)

        x[idx] = float(tmp) - h
        fh2 = f(x)

        grad[idx] = (fh1 - fh2) / (2 * h)
        x[idx] = tmp

        itr.iternext()
    return grad
