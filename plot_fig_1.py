from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def function_2(x):
    return np.sum(x**2)

#plt.xlabel("x0")
#plt.ylabel("x1")
#plt.zlabel("f(x)")

def function_q(x, y):
    return x**2 + y**2

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)

x = X.flatten()
y = Y.flatten()

Z_test = function_2(np.array([x, y]))
#print(Z_test.shape)

#Z = function_q(X,Y)
#print(Z.shape)

#plt.plot(X,Y,Z)
plt.plot(x, y, Z_test)

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("f(x)")

ax.plot_wireframe(x, y, Z_test)
plt.show()
