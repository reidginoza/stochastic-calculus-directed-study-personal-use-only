import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

ML_TRIALS = 1000
THETA = 1
SIGMA = 1

r = np.random.normal(scale=SIGMA**2, size=(ML_TRIALS))
sample = THETA + r
print(sample.mean())


def neg_log_llh(theta, sample, T, sigma):
    return - T/2 * np.log(2 * np.pi * sigma**2) + 1 / (2 * sigma) * ((sample - theta)**2).sum()


def nll_vec(theta_plot, sample, T, sigma):
    return np.array([neg_log_llh(theta, sample, T, sigma) for theta in theta_plot])


theta_plot = np.linspace(0, 3, 1000)
likelihood_plot = nll_vec(theta_plot, sample, ML_TRIALS, SIGMA)

fig, ax = plt.subplots()
ax.plot(theta_plot, likelihood_plot,
        label='Likelihood')
ax.vlines(x=THETA, ymin=likelihood_plot.min()-10, ymax=likelihood_plot.max(),
          label='True Parameter')
ax.vlines(x=sample.mean(), ymin=likelihood_plot.min()-10,
          ymax=likelihood_plot.max(), colors='g', ls='dashed',
          label='ML Estimator\n' + r'$\theta_{\mathrm{ML}} = ' + str(round(sample.mean(), 3)) + '$')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Likelihood ' + r'$L(\theta)$')
ax.set_title('Likelihood Function')
ax.legend()
