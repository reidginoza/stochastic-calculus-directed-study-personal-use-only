# ----- Set up
from matplotlib import pyplot as plt
import numpy as np

# ----- Brownian Motion


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


def ito_integral(w, dw):
    return (w[:-1]*dw[1:]).sum(axis=0)


def exact_ito(w, end_time):
    return 0.5 * (w[-1]**2 - end_time)


def approx_w_midpt(w, dt):
    """ Takes w, a vector of length N returns a vector of length N-1
    given W(t_j) and W(t_{j+1}), this approximates W evaluated at the
    midpoint, ie. W( (t_j+t_{j+1}) / 2 )"""
    return 0.5*(w[:-1] + w[1:]) + np.random.normal(scale=dt/4, size=w[:-1].shape)


def stratonovich_integral(w, dw, dt):
    w_mid = approx_w_midpt(w, dt)
    return (w_mid * dw[1:]).sum(axis=0)


def exact_stratonovich(w):
    return 0.5 * w[-1]**2


def plot_results(approx, exact, title=''):
    plt.figure()
    plt.plot(exact, '.', label='Exact')
    plt.plot(approx, '+', alpha=0.5, label='Approx.')
    plt.legend()
    plt.title(title)
    plt.show()


def error_plot(approx, exact, title=''):
    plt.figure()
    plt.hist(approx-exact, bins='auto')
    plt.ylabel('Frequency')
    plt.xlabel('Error')
    plt.title(title)
    plt.show()


# ----- Start


if __name__ == '__main__':
    END_TIME = 1
    NUM_TSTEPS = 500
    N_TRIALS = 1000
    dt = END_TIME / NUM_TSTEPS
    t, w, dw = multiple_brownian_motion(END_TIME, NUM_TSTEPS, N_TRIALS)
    ito_approx = ito_integral(w, dw)
    ito_ex = exact_ito(w, END_TIME)
    ito_error = np.abs(ito_approx-ito_ex)
    plot_results(ito_approx[:20], ito_ex[:20], f'Ito Integration Results\n20 Trials out of $n={N_TRIALS}$')
    error_plot(ito_approx, ito_ex, f'Error of Ito Integral Approximation\n$n={N_TRIALS}$')

    # ----- Stratonovich
    strat_approx = stratonovich_integral(w, dw, dt)
    strat_ex = exact_stratonovich(w)
    strat_error = np.abs(strat_approx-strat_ex)
    plot_results(strat_approx[:20], strat_ex[:20], f'Stratonovich Integration Results\n20 Trials out of $n={N_TRIALS}$')
    error_plot(strat_approx, strat_ex, f'Error of Stratonovich Integral Approximation\n$n={N_TRIALS}$')


