import numpy as np
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

# params that won't change throughout
sd = 1
alpha = 1
beta = 1
rep = 1

### 4 of 9
print("4 of 9")
problem=4
for i in np.arange(rep): 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=100, 1 changepoint, small mean diff, runs for 30s
    ######
    M = 2 # number of segments
    n = 100
    n_MCMC = 0.5
    diff_ind = 2
    helpers.plot_ess(M, n, n_MCMC, diff_ind, i)
    ###### 
    # n=100, 1 changepoint, large mean diff, runs for 30s
    ######
    diff_ind = 5
    helpers.plot_ess(M, n, n_MCMC, diff_ind, i)

### 5 of 9
print("5 of 9")
problem=5
for i in np.arange(rep): 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=100, 2 changepoints, small mean diff, runs for 3 min
    ######
    M = 3 # number of segments
    n = 100
    n_MCMC = 3
    diff_ind = 2
    helpers.plot_ess(M, n, n_MCMC, diff_ind, i)
    ###### 
    # n=100, 2 changepoints, large mean diff, runs for 3 min
    ######
    diff_ind = 5
    helpers.plot_ess(M, n, n_MCMC, diff_ind, i)

### 6 of 9
print("6 of 9")
problem=6
for i in np.arange(rep): 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=100, 3 changepoints, small mean diff, runs for 5 min
    ######
    M = 4 # number of segments
    n = 100
    n_MCMC = 5
    diff_ind = 2
    helpers.plot_ess(M, n, n_MCMC, diff_ind, i)
    ###### 
    # n=100, 3 changepoints, large mean diff, runs for 5 min
    ######
    diff_ind = 5
    helpers.plot_ess(M, n, n_MCMC, diff_ind, i)
