from matplotlib import pyplot as plt
import numpy as np


def plot_on_axis(ax, time, pos, cols, title, with_mean=False):
    ax.plot(time, pos[:, cols])
    ax.set_title(title)
    if with_mean:
        ax.plot(time, pos.mean(axis=1), color='black',
            label='Sample Mean', linewidth=2)
        ax.legend()


def multiple_brownian_motion(end_time=1., num_tsteps=500, n_trials=1000):
    dt = (end_time - 0) / num_tsteps
    dw = np.random.normal(scale=np.sqrt(dt), size=(num_tsteps+1, n_trials))
    # Brownian motion must start at time 0 with value 0
    dw[0] = np.zeros_like(dw[0])
    w = dw.cumsum(axis=0)
    t = np.linspace(0, end_time, num=num_tsteps+1)  # not used in calculations
    assert w.shape[0] == t.shape[0], f'time and position arrays are not the same length. w.shape[0] - t.shape[0] = {w.shape[0] - t.shape[0]}'
    assert w.shape == dw.shape, f'position and velocity arrays are not the same shape: w.shape: {w.shape}    dw.shape: {dw.shape}'
    return t, w, dw


def y_process(t, w):
    """ The stochastic process defined in the exercise.
    This function assumes that the same time axis is used for all Brownian motion,
    can pass the time array output from multiple_brownian_motion.
    """
    assert t.shape[0] == w.shape[0], f'Time and Brownian and motion need to be the same length: time {t.shape[0]} vs {w.shape[0]}'

    return np.exp(t/2).reshape((t.shape[0], 1)) * np.cos(w)

########
END_TIME = 2*np.pi
NUM_TSTEPS = round(END_TIME * 1000)
N_TRIALS = 1000
N_PATHS = 10
########
time, w, dw = multiple_brownian_motion(
    END_TIME, NUM_TSTEPS, N_TRIALS)

position = y_process(time, w)

columns = np.arange(position.shape[1])
np.random.shuffle(columns)
plot_columns = columns[:N_PATHS]

fig, ax = plt.subplots(1, figsize=(7, 6.5))
plot_on_axis(ax, time, position, plot_columns, r'$Y$ Process', with_mean=True)
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()

########
# didn't use the code below. Just exploration


def y2_process(t, w):
    """ The stochastic process defined in the exercise.
    This function assumes that the same time axis is used for all Brownian motion,
    can pass the time array output from multiple_brownian_motion.
    """
    assert t.shape[0] == w.shape[0], f'Time and Brownian and motion need to be the same length: time {t.shape[0]} vs {w.shape[0]}'

    return np.exp(t/2).reshape((t.shape[0], 1)) * w


position2 = y2_process(time, w)
fig, ax = plt.subplots(1, figsize=(7, 6.5))
plot_on_axis(ax, time, position2, plot_columns, r'$Y2$ Process', with_mean=True)
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()