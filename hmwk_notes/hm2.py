

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

END_TIME = 1.
NUM_TSTEPS = 500
N_TRIALS = 8
CONSISTENT = True
PLOTS = True


def brownian_motion(end_time=1., num_tsteps=500):
    dt = (end_time - 0) / num_tsteps
    dw = np.sqrt(dt) * np.random.normal(size=num_tsteps+1)
    # Brownian motion must start at time 0 with value 0
    dw[0] = 0
    w = dw.cumsum()
    t = np.linspace(0, end_time, num=num_tsteps+1)  # not used in calculations
    assert len(w) == len(t), f'time and position arrays are not the same length. len(t) - len(w) = {len(t) - len(w)}'
    return t, w, dw

if CONSISTENT:
    np.random.seed(0)  # keeps the psuedo-random number
    # generator in the same sequence from run to run

if PLOTS:
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(7,5))

results = []

for i in range(N_TRIALS):
    time, position, velocity = brownian_motion(end_time=END_TIME, num_tsteps=NUM_TSTEPS)
    results.append((position, velocity))

    if PLOTS:
        ax0.plot(time, position, label=f'Trial {i}')

if PLOTS:
    ax0.plot(time, np.zeros_like(time), color='0.0', label='Expectation', linewidth=2)
    ax0.plot(time, 2 * np.sqrt(time), '--', color='0.75')
    ax0.plot(time, -2 * np.sqrt(time) * np.ones_like(time), '--', color='0.75', label='95% Conf. Int.')
    ax0.set_title('Examples of One-Dimensional Brownian Motion')
    box = ax0.get_position()
    ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    mean = np.array([res[0] for res in results]).T.mean(axis=1)

    ax1.plot(time, mean, label='Sample Mean')
    ax1.plot(time, np.zeros_like(time), label='Expectation')
    ax1.set_title('Sample Mean and Mathematical Expectation')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0.)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()

END_TIME = 1.
NUM_TSTEPS = 500
N_TRIALS = 100000
CONSISTENT = True
PLOTS = True
N_PATHS = 4


def multiple_brownian_motion(end_time=1., num_tsteps=500, n_trials=1000):
    dt = (end_time - 0) / num_tsteps
    dw = np.random.normal(scale=np.sqrt(dt), size=(num_tsteps+1, n_trials))
    # Brownian motion must start at time 0 with value 0
    dw[0] = np.zeros_like(dw[0])
    w = dw.cumsum(axis=1)
    t = np.linspace(0, end_time, num=num_tsteps+1)  # not used in calculations
    assert w.shape[0] == t.shape[0], f'time and position arrays are not the same length. w.shape[0] - t.shape[0] = {w.shape[0] - t.shape[0]}'
    assert w.shape == dw.shape, f'position and velocity arrays are not the same shape: w.shape: {w.shape}    dw.shape: {dw.shape}'
    return t, w, dw


def wie_expt_coeff(k):
    return np.math.factorial(2*k) / (2**k * np.math.factorial(k))


def wiener_expectation(k, t):
    return wie_expt_coeff(k) * t**k

K = 2
time, position, velocity = multiple_brownian_motion(end_time=0.2, num_tsteps=500, n_trials=1000)

w4 = np.power(position, 4)
expectation = wiener_expectation(K, time)
variance = wiener_expectation(K*2, time) - wiener_expectation(K, time)**2

columns = np.arange(w4.shape[1])
np.random.shuffle(columns)
plot_columns = columns[:N_PATHS]

# fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
# ax0.errorbar(time, expectation, yerr=2 * np.sqrt(variance), fmt='-o', alpha=0.4)
# ax0.plot(time, expectation, linewidth=4, alpha=0.7)
#
# for col in plot_columns:
#     ax1.plot(time, w4[:, col], alpha=0.4)
# ax1.plot(time, w4.mean(axis=1), linewidth=4, alpha=0.7)
#
# ax2.plot(time, expectation, label='Expectation')
# ax2.plot(time, w4.mean(axis=1), label='Sample Mean')
#
# plt.show()