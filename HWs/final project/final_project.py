import emcee
import corner
import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt


def log_likelihood(theta, x, y, yerr, fitting_type):
    """
    Likelihood function in log scale for MCMC process
    :param theta: fitting parameters at current step
    :param x: input data for x-axis
    :param y: input data for y-axis
    :param yerr: input error for y-axis
    :param fitting_type: fitting mode for linear or non-linear
    :return: log(likelihood) based on current parameter
    """
    # KZ construct fitting function for linear and non-linear mode
    if fitting_type == "linear":
        m, b = theta
        model = m * x + b
    else:
        a_0, a_1, a_2 = theta
        model = a_2 * x**2 + a_1 * x + a_0

    # KZ return the log(chi^2) value given the current parameters
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))


def log_prior(theta, fitting_type):
    """
    Prior information in log scale for MCMC process
    :param theta: fitting parameters at current step
    :param fitting_type: fitting mode for linear or non-linear
    :return: prior information based on current parameter
    """
    # KZ construct prior info for linear and non-linear mode
    if fitting_type == "linear":
        m, b = theta
        if -10 < m < 10.0 and -10 < b < 10:
            return 0.0
    else:
        a_0, a_1, a_2 = theta
        if -10 < a_2 < 10.0 and -10 < a_1 < 10 and -10 < a_0 < 20:
            return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr, fitting_type):
    """
    Probability in log scale for MCMC process
    :param theta: fitting parameters at current step
    :param x: input data for x-axis
    :param y: input data for y-axis
    :param yerr: input error for y-axis
    :param fitting_type: fitting mode for linear or non-linear
    :return: log(probability) based on current parameter
    """
    lp = log_prior(theta, fitting_type)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, fitting_type)


def postprocess_result(samples, burn_in):
    """
    Calculate the 16th, 50th and 84th percentiles for parameter
    :param samples: sampling distribution for parameter
    :param burn_in: discard initial burn_in steps
    :return: array of the 16th, 50th and 84th percentiles for parameter
    """
    # KZ discard the burn-in phase can improve the result statistics
    return np.percentile(samples[burn_in:], [16, 50, 84])


def mcmc_linear(x, y, yerr):
    """
    MCMC process for the linear fitting
    :param x: input data for x-axis
    :param y: input data for y-axis
    :param yerr: input error for y-axis
    """
    # KZ start 32 walkers at some initial pos
    nwalkers, ndim = 32, 2
    pos = np.array([3, 5.5]) + 1e-4 * np.random.randn(32, 2)

    # KZ start MCMC with linear log_probability function and 5000 steps
    print("start MCMC process")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(x, y, yerr, "linear")
    )
    sampler.run_mcmc(pos, 5000, progress=True)

    # KZ extract parameters statistics with 500 burn-in steps
    samples = sampler.get_chain()
    m = postprocess_result(samples[:, :, 0], 500)
    b = postprocess_result(samples[:, :, 1], 500)
    
    print(f"the 16th, 50th and 84th percentiles for m is {m}")
    print(f"the 16th, 50th and 84th percentiles for b is {b}")

    # KZ Make plots to show the fitting results
    print("plot the corner plot for parameter distribution")
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    labels = ["m", "b"]
    fig = corner.corner(flat_samples, labels=labels, truths=[m[1], b[1]])
    plt.show()
    
    print("plot the data along with the best-fit line")
    x_data = np.linspace(np.min(x), np.max(x), 100)
    plt.errorbar(x, y, yerr, fmt='s', label='data')
    plt.plot(x_data, m[1] * x_data + b[1], label='fitting')
    plt.legend()
    plt.show()
    return


def mcmc_non_linear(x, y, yerr):
    """
    MCMC process for the non-linear fitting
    :param x: input data for x-axis
    :param y: input data for y-axis
    :param yerr: input error for y-axis
    """
    # KZ start 32 walkers at some initial pos
    nwalkers, ndim = 32, 3
    pos = np.array([3, 5.5, 3]) + 1e-4 * np.random.randn(32, 3)

    # KZ start MCMC with non-linear log_probability function and 5000 steps
    print("start MCMC process")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(x, y, yerr, "non_linear")
    )
    sampler.run_mcmc(pos, 5000, progress=True)

    # KZ extract parameters statistics with 500 burn-in steps
    samples = sampler.get_chain()
    a_0 = postprocess_result(samples[:, :, 0], 500)
    a_1 = postprocess_result(samples[:, :, 1], 500)
    a_2 = postprocess_result(samples[:, :, 2], 500)
    
    print(f"the 16th, 50th and 84th percentiles for a0 is {a_0}")
    print(f"the 16th, 50th and 84th percentiles for a1 is {a_1}")
    print(f"the 16th, 50th and 84th percentiles for a2 is {a_2}")

    # KZ Make plots to show the fitting results
    print("plot the corner plot for parameter distribution")
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    labels = ["a0", "a1", "a2"]
    fig = corner.corner(flat_samples, labels=labels, truths=[a_0[1], a_1[1], a_2[1]])
    plt.show()
    
    print("plot the data along with the best-fit line")
    x_data = np.linspace(np.min(x), np.max(x), 100)
    plt.errorbar(x, y, yerr, fmt='s', label='data')
    plt.plot(x_data, a_2[1] * x_data**2 + a_1[1] * x_data + a_0[1], label='fitting')
    plt.legend()
    plt.show()
    return


def main():
    """Command-line interface for MCMC fitting process."""
    data = Table.read('/d/scratch/ASTR5160/final/dataxy.fits')
    x = data['x']
    y = data['y']
    yerr = data['yerr']
    
    print("First try linear fitting")
    mcmc_linear(x, y, yerr)
    
    print(" ")
    
    print("Then try quadratic fitting")
    mcmc_non_linear(x, y, yerr)
    
    print(" ")
    
    print("The best-fit a2 ~ 0.06, quite small, thus I feel like the linear fit should be sufficient at this x range.")
    return


if __name__ == "__main__":
    main()
