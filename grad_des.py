import numpy as np
import matplotlib.pylab as plt

def gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]
        x[i] = float(tmp) + h
        fxh1 = f(x)
        x[i] = float(tmp) - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp
    return grad


def numerical_grad(f, X):
    if X.ndim == 1:
        grad = gradient(f, X)
    else:
        grad = np.zeros_like(X)
        #for i in range(X.size):
        #    grad[i] = gradient(f, X[i])
        for idx, x in enumerate(X):
            grad[idx] = gradient(f, x)
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_grad(f, x)
        x -= lr * grad
    return x, np.array(x_history)
    #return x

def function_2(x):
    return np.sum(x**2)

if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])

    lr = 0.1
    #lr = 10.0
    #lr = 1e-10
    step_num = 20
    x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()
