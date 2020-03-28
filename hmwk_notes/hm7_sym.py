from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp

sp.init_printing()

x, t = sp.symbols('x t')

p = 1 / sp.sqrt(2 * sp.pi * t) * sp.cosh(x) * sp.exp(-1/2 * t) * sp.exp(- 1 / (2 * t) * x**2)
# print('dp/dt')
# print(sp.simplify(sp.diff(p, t)))
#
# print('Left Hand Side')
# lhs = - sp.diff(sp.tanh(x) * p, x) + 1/2 * sp.diff(p, x, 2)
# print(sp.simplify(lhs))

p_num = sp.lambdify((x, t), p)

x_plot = np.linspace(-10, 10, 200)
t_plot = np.linspace(0, 5, 201)[1:]

xx, tt = np.meshgrid(x_plot, t_plot)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_zlim(top=0.2)
ax1.plot_surface(xx, tt, p_num(xx, tt), cmap=plt.cm.get_cmap('viridis'))
