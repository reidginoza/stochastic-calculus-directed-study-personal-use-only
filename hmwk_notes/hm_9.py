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
estimator_result = minimize_scalar(neg_log_llh, args=(sample, ML_TRIALS, SIGMA))

fig, ax = plt.subplots()
ax.plot(theta_plot, likelihood_plot,
        label='Likelihood')
ax.vlines(x=THETA, ymin=likelihood_plot.min()-50, ymax=likelihood_plot.max(),
          label='True Parameter')
ax.plot(sample.mean(), neg_log_llh(sample.mean(), sample, ML_TRIALS, SIGMA), 'o',
          label='ML Estimator\n' + r'$\theta_{\mathrm{ML}} = ' + str(round(sample.mean(), 3)) + '$')
# ax.plot(estimator_result.x, estimator_result.fun, 'o',
#         label=f'Numerical Minimization: {round(estimator_result.x, 3)},'
#               f' {round(estimator_result.fun, 3)}')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Likelihood ' + r'$L(\theta)$')
ax.set_title('Likelihood Function')
ax.legend()

T = 1000
THETA = 1
SIGMA = 1
LAMBDA = 2


def prior(theta, lamb):
    return 1 / np.sqrt(2 * np.pi * lamb**2) * np.exp( - theta**2 / (2 * lamb**2))


def posterior(theta, lamb, sigma, sample):
    """ f1 * exp(f2 + f3 + f4) """
    f1 = np.sqrt(2 * np.pi * lamb**2)
    f2 = - theta**2 / (2 * lamb**2)
    f3 = sample.size / 2 * np.log(2 * np.pi * sigma**2)
    f4 = - 1 / (2 * sigma**2 ) * np.sum((sample - theta)**2)
    return f1 * np.exp(f2 + f3 + f4)


# run simulation
r = np.random.normal(scale=SIGMA**2, size=(T,))
sample = THETA + r


def this_neg_posterior(theta):
    return -1 * posterior(theta, LAMBDA, SIGMA, sample)


num_map_estimate = minimize_scalar(this_neg_posterior)

def map_estimate(T, sigma, lamb, sample):
    return 1 / (T + sigma**2/lamb**2) * sample.sum()

theta_map = map_estimate(T, SIGMA, LAMBDA, sample)

# plotting
theta_plot = np.linspace(-5, 5, 5000)
prior_plot = prior(theta_plot, LAMBDA)
posterior_plot = np.array([posterior(th, LAMBDA, SIGMA, sample) for th in theta_plot])

def low_buff(y_values):
    span = y_values.max() - y_values.min()
    return y_values.min() - 0.1 * span

fig2, axes2 = plt.subplots(nrows=2)
# prior plot
axes2[0].plot(theta_plot, prior_plot, label='Prior')
axes2[0].vlines(x=THETA, ymin=low_buff(prior_plot), ymax=prior_plot.max(),
          label='True Parameter')
axes2[0].set_title('Prior Distribution')
axes2[0].legend()

# posterior plot
axes2[1].plot(theta_plot, prior_plot, label='Prior', ls='dashed', alpha=0.8)
axes2[1].plot(theta_plot, posterior_plot, label='Posterior')
axes2[1].vlines(x=THETA, ymin=low_buff(posterior_plot), ymax=posterior_plot.max(),
          label=f'True Parameter: {THETA}')
axes2[1].vlines(x=theta_map, ymin=low_buff(posterior_plot), ymax=posterior_plot.max(),
          label=f'MAP Estimator: {round(theta_map, 3)}')
axes2[1].vlines(x=num_map_estimate.x, ymin=low_buff(posterior_plot), ymax=posterior_plot.max(),
          label=f'Numerical MAP\nEstimator: {round(num_map_estimate.x, 3)}')
axes2[1].set_xlim(left=0, right=2)
axes2[1].set_title('Posterior Distribution')
axes2[1].legend()
fig2.tight_layout()
plt.show()
