"""
Provides tools for computing Monte Carlo standard errors (MCSE) in Markov chain Monte Carlo (MCMC).
A python/numpy implementation of mcmcse.r, extended to support numpy broadcasting.

By Žiga Sajovic
"""
import numpy as np
from scipy.stats import gaussian_kde
from numba import jit

# @jit(nopython=True)
def ess(x):
    lambda_ = np.var(x, ddof=1, axis=0)

    n = len(x)
    shape = x.shape
    b = int(np.floor(np.sqrt(n)))
    a = int(np.floor(n/b))
    n_ = int(a*b)
    y = np.mean(np.reshape(x[:n_], (a, b, *shape[1:])), axis=1)
    mu_hat = np.mean(x, axis=0)
    var_hat = b*np.sum((y-mu_hat)**2, axis=0)/(a-1)
    se = np.sqrt(var_hat/n)

    sigma = se**2*n
    return n*lambda_/sigma, se