from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp

sp.init_printing()

x, t = sp.symbols('x t')

p = 1 / sp.sqrt(2 * sp.pi * t) * sp.cosh(x) * sp.exp(-1/2 * t) * sp.exp(- 1 / (2 * t) * x**2)
print('dp/dt')
lhs = sp.diff(p, t)
print(lhs)
print(sp.latex(lhs))
print('\nsimplified:')
print(sp.simplify(lhs))
print(sp.latex(sp.simplify(lhs)))

my_lhs = (
    (1 / (sp.sqrt(2 * sp.pi * t)**3))
    * (2 * sp.pi / t * x**2 - 2 * sp.pi * t - 1)
    * (sp.exp(-t / 2) * sp.exp(- x**2 / (2*t)))
    * sp.cosh(x)
)

print('Is my LHS correct?')
print(sp.simplify(sp.simplify(lhs) - sp.simplify(my_lhs)))

print('\n\nRight Hand Side')
print('d tanh(x) * p')
print(sp.diff(sp.tanh(x) * p, x))
print(sp.latex(sp.diff(sp.tanh(x) * p, x)))
print('\nd^2 p / dx^2')
print(sp.diff(p, x, 2))
print(sp.latex(sp.diff(p, x, 2)))

# p_num = sp.lambdify((x, t), p)
#
# x_plot = np.linspace(-10, 10, 200)
# t_plot = np.linspace(0, 5, 201)[1:]
#
# xx, tt = np.meshgrid(x_plot, t_plot)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.set_zlim(top=0.2)
# ax1.plot_surface(xx, tt, p_num(xx, tt), cmap=plt.cm.get_cmap('viridis'))
