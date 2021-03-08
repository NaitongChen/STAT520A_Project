import numpy as np
from numpy.lib.function_base import diff
from scipy import stats as stat
import pickle as pk
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
import helpers
import mcmcse
sys.path.insert(1, os.path.join(sys.path[0], '../mcmc_sampler'))
from gibbs import gibbs
from metropolis_within_gibbs import metropolis_within_gibbs
sys.path.insert(1, os.path.join(sys.path[0], '../data_generation'))
from gaussian_mean_shift import generate_gaussian_mean_shift
import matplotlib.pyplot as plt

np.seterr(all='raise')

# params that won't change throughout
sd = 1
alpha = 1
beta = 1
rep = 1
i = np.array([0,1,2,3,4])

M = np.array([2, 2, 4, 4, 3])
n = np.array([50, 20000, 60, 100, 100])
n_MCMC = np.array([1, 3, 3, 3, 3])
diff_ind = np.array([2, 5])

for m in np.arange(M.shape[0]):
    print(m)
    helpers.plot_kl(M[m], n[m], n_MCMC[m], diff_ind[0], i)