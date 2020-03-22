# --- Python imports
from matplotlib import pyplot as plt
import numpy as np


# --- Brownian Motion
def multiple_brownian_motion(end_time=1., num_tsteps=500, n_trials=1000):
    """Creates multiple Brownian motion with time as the row index and
    each column as a separate path of Brownian motion.

    This assumes that all Brownian motion starts at 0. Currently only
    implements one-dimensional Brownian motion. This also assumes all
    step sizes are the same size.

    The steps of Brownian motion, ``dw``, are modeled with a Gaussian
    distribution with mean 0 and variance ``sqrt(dt)``, where ``dt``
    is the constant time step size.

    Parameters
    ----------
    end_time : float

    num_tsteps : int
        The number of steps to take. Will calculate the step
        size dt internally. The number of rows in the output of
        Brownian motion will be num_tsteps + 1.

    n_trials : int
        The number of sample paths to create. This will be the number
        of columns in the output.

    Returns
    -------
    t : ndarray
        One-dimensional time ndarray from 0 to ``end_time`` with
        shape (``num_tsteps``+1,)

    w : ndarray
        Two-dimensional ndarray representing ``n_trials`` number of
        sample paths of one-dimensional Brownian motion.
        This will be of shape (``num_tsteps``+1, ``n_trials``).

    dt : float
        The value indicating the step size of t. This is only implemented
        with constant step size.

    dw : ndarray
        Two-dimensional ndarray representing the steps of Brownian motion.
        The first row is all zeros. Each i-th row of ``dw``, ie. dw[i, :]
        indicates the change in ``w`` from w[i-1, :] to w[i, :]
        This will be the same shape as ``w``, (``num_tsteps``+1, ``n_trials``).

    """

    dt = (end_time - 0) / num_tsteps
    dw = np.random.normal(scale=np.sqrt(dt), size=(num_tsteps+1, n_trials))
    # Brownian motion must start at time 0 with value 0
    dw[0] = np.zeros_like(dw[0])
    w = dw.cumsum(axis=0)
    # t is not used in calculations, but returned to allow user to keep track
    # of points in time
    t = np.linspace(0, end_time, num=num_tsteps+1).reshape((num_tsteps+1, 1))
    assert w.shape[0] == t.shape[0], ('time and position arrays are not the '
                                      'same length. w.shape[0] - t.shape[0] = '
                                      f'{w.shape[0] - t.shape[0]}')
    assert w.shape == dw.shape, ('position and velocity arrays are not the '
                                 'same shape: '
                                 f'w.shape: {w.shape}    dw.shape: {dw.shape}')
    return t, w, dt, dw


# --- Solve SDE
def analytical_truth_linear(b, u, N0, t, w):
    """
    Provides the true (analyatical) solution evaluated at t

    Can be vectorized for multiple trials based on the Brownian motion
    parameter ``w``.

    :param b: shift parameter in SDE
    :param u: drift parameter in SDE
    :param N0: Initial condition
    :param t: time nd-array
    :param w: brownian motion nd-array
    :param dt: step size of time array, float
    :param dw: White noise associated with the Brownian motion, ndarray

    :return: nd-array true solution
    """
    return N0 * np.exp((b - 0.5 * u ** 2) * t + u * w)


def euler_maruyama_linear(b, u, N0, t, dt, dw, M):
    """
    calculates the EM approximation on the linear SDE
    :param b: shift parameter in SDE
    :param u: drift parameter in SDE
    :param N0: Initial condition
    :param t: time nd-array
    :param dt: step size of time array, float
    :param dw: White noise/Gaussian associated with the Brownian motion, nd-array
    :param M: multiple of dt for Euler-Maruyama step size
    :return: time array and solution N array
    """

    Dt = M * dt  # EM step size
    L = (t.shape[0] - 1) / M  # number of EM steps
    assert L.is_integer(), 'Cannot handle Step Size that is not a multiple of M'
    L = int(L)  # needed for range below

    N = [N0]
    T = [0]
    for i in range(1, L+1):
        # DW is the step of Brownian motion for EM step size
        DW = (dw[M * (i - 1) + 1:M * i + 1]).sum(axis=0)
        N.append(N[i-1] + Dt * b * N[i - 1] + u * N[i - 1] * DW)
        T.append(T[i-1] + Dt)

    return np.array(T), np.array(N)


def euler_maruyama_linear_vec(b, u, N0, t, dt, dw, M):
    """
    calculates the EM approximation on the linear SDE, vectorized for
    multiple trials based on the shape of ``dw``.
    :param b: shift parameter in SDE
    :param u: drift parameter in SDE
    :param N0: Initial condition
    :param t: time nd-array
    :param dt: step size of time array, float
    :param dw: White noise associated with the Brownian motion, ndarray
    :param M: multiple of dt for Euler-Maruyama step size. Do not make this too large.
    :return: time array and solution N array
    """

    if M < 1:
        raise ValueError('M must be greater than or equal to 1')

    Dt = M * dt  # EM step size
    L = (t.shape[0] - 1) / M  # number of EM steps

    if not L.is_integer():
        raise ValueError('Cannot handle Step Size that is not a multiple of M')

    L = int(L)  # needed for range below

    N = [np.full((dw.shape[1],), N0)]
    T = [0]
    for i in range(1, L+1):
        # DW is the step of Brownian motion for EM step size
        DW = (dw[M * (i - 1) + 1:M * i + 1, :]).sum(axis=0).reshape(dw.shape[1], )
        N.append(N[i-1] + Dt * b * N[i - 1] + u * N[i - 1] * DW)
        T.append(T[i-1] + Dt)

    return np.array(T), np.array(N)


def plot_on_axis(ax, time, pos, cols, title, color_map, with_mean=False):
    for idx, col in enumerate(cols):
        ax.plot(time, pos[:, col], c=color_map(idx))
    ax.set_title(title)
    if with_mean:
        ax.plot(time, pos.mean(axis=1), color='black',
            label=r'Sample Mean $(n={})$'.format(pos.shape[1]), linewidth=2)
        ax.legend()



# -- SDE
# Parameters for SDE
b = 2
u = 1

# Initial Condition
N0 = 1

# Brownian Motion
END_TIME = 1.
NUM_TSTEPS = 2**8
N_TRIALS = 1000

# Euler-Maruyama
M = 4  # multiple of step size

# Plotting Parameters
N_PATHS = 5
viridis = plt.get_cmap('viridis', lut=N_PATHS)
columns = np.arange(N_PATHS)
np.random.shuffle(columns)
plot_columns = columns[:N_PATHS]

# Truth
t, w, dt, dw = multiple_brownian_motion(end_time=END_TIME,
                                        num_tsteps=NUM_TSTEPS,
                                        n_trials=N_TRIALS)
N_true = analytical_truth_linear(b, u, N0, t, w)

fig, ax = plt.subplots(1, figsize=(7, 6.5))
plot_on_axis(ax, t, N_true, plot_columns, r'$N$ Process Analytical Solution',
             color_map=viridis, with_mean=True)
plt.xlabel('Time')
plt.ylabel('Population')


# # Euler Maruyama
# N_ems = []
# for col in range(N_true.shape[1]):
#     t_em, N_em = euler_maruyama_linear(r, u, N0, t, dt, dw[:, col], R)
#     N_ems.append(N_em)
#
# t_em = np.array(t_em)
# N_ems = np.array(N_ems).transpose()
#
# fig, ax = plt.subplots(1, figsize=(7, 6.5))
# plot_on_axis(ax, t_em, N_ems, plot_columns, r'EM Approx. $N$ Process',
#              color_map=viridis, with_mean=True)
# plt.xlabel('Time')
# plt.ylabel('Population')

# Euler Maruyama vectorized
t_em_v, N_ems_v = euler_maruyama_linear_vec(b, u, N0, t, dt, dw, M)

fig, ax = plt.subplots(1, figsize=(7, 6.5))
plot_on_axis(ax, t_em_v, N_ems_v, plot_columns, r'$N$ Process'+'\nEuler Maruyama Approximation',
             color_map=viridis, with_mean=True)
plt.xlabel('Time')
plt.ylabel('Population')

# Comparison
fig, ax = plt.subplots(1, figsize=(7, 6.5))
for p in range(N_PATHS):
    ax.plot(t, N_true[:, p], c=viridis(p))
    ax.plot(t_em_v, N_ems_v[:, p], c=viridis(p), marker='.', ls='', alpha=0.5)
ax.set_title('Comparison Between Truth and EM Approximation')
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()
