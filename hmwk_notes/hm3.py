

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


def shift_brownian_motion(time_length=1., num_tsteps=500, n_trials=1000, shift=0.5):
    """
    This function calculates the shift with time starting at 0
    and the first value at shift
    """
    dt = time_length / num_tsteps
    end_time = shift + time_length
    assert (end_time / dt).is_integer(), 'adjust end_time and shift so that (end_time / dt).is_integer()'
    total_steps = int(end_time // dt) + 1
    dw = np.random.normal(scale=np.sqrt(dt), size=(total_steps, n_trials))

    # Brownian motion must start at time 0 with value 0
    dw[0] = np.zeros_like(dw[0])
    w = dw.cumsum(axis=0)
    t = np.linspace(0, end_time, num=total_steps)

    # shift occurs here
    w = w[t >= shift]
    w -= w[0]
    t = t[t >= shift]

    assert w.shape[0] == t.shape[0], f'time and position arrays are not the same length. w.shape[0] - t.shape[0] = {w.shape[0] - t.shape[0]}'
    return t, w


def scale_brownian_motion(end_time=1., num_tsteps=500, n_trials=1000, scale=0.5):
    """
    This function calculates the scaling
    """
    dt = (end_time - 0) / num_tsteps / scale**2
    dw = np.random.normal(scale=np.sqrt(dt), size=(num_tsteps + 1, n_trials))
    # Brownian motion must start at time 0 with value 0
    dw[0] = np.zeros_like(dw[0])
    w = dw.cumsum(axis=0) * scale
    t = np.linspace(0, end_time, num=num_tsteps + 1) / scale**2
    assert w.shape[0] == t.shape[0], f'time and position arrays are not the same length. w.shape[0] - t.shape[0] = {w.shape[0] - t.shape[0]}'
    assert w.shape == dw.shape, f'position and velocity arrays are not the same shape: w.shape: {w.shape}    dw.shape: {dw.shape}'
    return t, w, dw


END_TIME = 1.
NUM_TSTEPS = 500
N_TRIALS = 1000
PLOTS = True
N_PATHS = 10


orig_time, orig_position, orig_velocity = multiple_brownian_motion(
    end_time=1., num_tsteps=NUM_TSTEPS, n_trials=N_TRIALS)
shift_time, shift_position = shift_brownian_motion(
    time_length=1., num_tsteps=NUM_TSTEPS, n_trials=N_TRIALS, shift=0.5)
scale_small_time, scale_small_position, scale_small_vel = scale_brownian_motion(
    end_time=1., num_tsteps=NUM_TSTEPS, n_trials=N_TRIALS, scale=2
)
scale_big_time, scale_big_position, scale_big_vel = scale_brownian_motion(
    end_time=1., num_tsteps=NUM_TSTEPS, n_trials=N_TRIALS, scale=0.5
)

if PLOTS:
    # select random paths since showing all N_TRIALS would be visually crowded.
    columns = np.arange(orig_position.shape[1])
    np.random.shuffle(columns)
    plot_columns = columns[:N_PATHS]

    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(7, 6.5))
    plot_on_axis(axs[0], orig_time, orig_position,
                 plot_columns, 'Original Brownian Motion Sample Paths',
                 with_mean=True)
    plot_on_axis(axs[1], shift_time, shift_position,
                 plot_columns, r'Shifted Brownian Motion Sample Paths $s=0.5$',
                 with_mean=True)
    axs[1].set_xlim(0, shift_time.max())
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()

if PLOTS:
    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(7, 10))
    plot_on_axis(axs[0], orig_time, orig_position,
                 plot_columns, 'Original Brownian Motion Sample Paths',
                 with_mean=True)
    plot_on_axis(axs[1], scale_small_time, scale_small_position,
                 plot_columns, r'Scaled Brownian Motion Sample Paths $c=2$',
                 with_mean=True)
    plot_on_axis(axs[2], scale_big_time, scale_big_position,
                 plot_columns, r'Scaled Brownian Motion Sample Paths $c=0.5$',
                 with_mean=True)

    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()
