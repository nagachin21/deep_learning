import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def numerical_diff_no_batch(func, x):
    h = 1e-7
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = float(tmp) + h
        fxh1 = func(x)

        x[idx] = float(tmp) - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2)/(2 * h)
        x[idx] = tmp
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_diff_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numerical_diff_no_batch(f, x)
        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def tangent_line(f, x):
    d = numerical_gradient(f,x)
    print(d)
    y = f(x) - d * x
    return lambda t: d*t + y

if __name__ == "__main__":
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    x = X.flatten()
    y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([x, y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    #plt.quiver(X, Y, -grad[0], -grad[1], color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
