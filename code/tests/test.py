import numpy as np
from scipy import stats as stat
import pickle as pk
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
import helpers
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
rep = np.array([1,2,3,4])
seed = 218

print("1 of 5")
n = 50
diff_ind = 2
means = np.array([2, 4])
locs = np.array([25])
seq1,_,_ = helpers.get_sequence(2, n, 1, diff_ind, 218)

problem=1
for i in rep: 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=50, 1 changepoint, small mean diff, runs for 1 min
    ######
    diff_ind = 2
    M = 2 # number of segments
    n_MCMC = 1
    mus = np.ones(M) * np.mean(seq1)
    vs = np.ones(M)

    # samples M-1 (M=mus.shape[0]) values from 0 to n-2 last index can't be a changepoint
    locs = helpers.sample_combinations(n-1, M-1, None) - 1
    # intialized using empirical segment means based on the sampled locations
    seg_means, _ = helpers.compute_seg_means(seq1, locs)

    a,b,c,d = metropolis_within_gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)
    e,f,g = gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)

print("2 of 5")
n = 20000
diff_ind = 2
means = np.array([2, 4])
locs = np.array([10000])
seq1,_,_ = helpers.get_sequence(2, n, 3, diff_ind, 218)

problem=2
for i in rep: 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=20000, 1 changepoint, small mean diff, runs for 3 min
    ######
    diff_ind = 2
    M = 2 # number of segments
    n_MCMC = 3
    mus = np.ones(M) * np.mean(seq1)
    vs = np.ones(M)

    # samples M-1 (M=mus.shape[0]) values from 0 to n-2 last index can't be a changepoint
    locs = helpers.sample_combinations(n-1, M-1, None) - 1
    # intialized using empirical segment means based on the sampled locations
    seg_means, _ = helpers.compute_seg_means(seq1, locs)

    a,b,c,d = metropolis_within_gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)
    e,f,g = gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)

print("3 of 5")
n = 60
diff_ind = 2
means = np.array([4, 6, 2, 4])
locs = np.array([15, 30, 45])
seq1,_,_ = helpers.get_sequence(4, n, 3, diff_ind, 218)

problem=3
for i in rep: 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=60, 3 changepoints, small mean diff, runs for 3 min
    ######
    diff_ind = 2
    M = 4 # number of segments
    n_MCMC = 3
    mus = np.ones(M) * np.mean(seq1)
    vs = np.ones(M)

    # samples M-1 (M=mus.shape[0]) values from 0 to n-2 last index can't be a changepoint
    locs = helpers.sample_combinations(n-1, M-1, None) - 1
    # intialized using empirical segment means based on the sampled locations
    seg_means, _ = helpers.compute_seg_means(seq1, locs)

    a,b,c,d = metropolis_within_gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)
    e,f,g = gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)

print("4 of 5")
n = 100
diff_ind = 2
means = np.array([4, 6, 2, 4])
locs = np.array([25, 50, 75])
seq1,_,_ = helpers.get_sequence(4, n, 3, diff_ind, 218)

problem=4
for i in rep: 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=100, 3 changepoints, small mean diff, runs for 5 min
    ######
    diff_ind = 2
    M = 4 # number of segments
    n_MCMC = 3
    mus = np.ones(M) * np.mean(seq1)
    vs = np.ones(M)

    # samples M-1 (M=mus.shape[0]) values from 0 to n-2 last index can't be a changepoint
    locs = helpers.sample_combinations(n-1, M-1, None) - 1
    # intialized using empirical segment means based on the sampled locations
    seg_means, _ = helpers.compute_seg_means(seq1, locs)

    a,b,c,d = metropolis_within_gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)
    e,f,g = gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)

print("5 of 5")
n = 100
diff_ind = 2
means = np.array([4, 6, 2])
locs = np.array([30, 60])
seq1,_,_ = helpers.get_sequence(3, n, 3, diff_ind, 218)

problem=5
for i in rep: 
    print(i)
    np.random.seed(i*problem)
    ###### 
    # n=100, 2 changepoints, small mean diff, runs for 3 min
    ######
    diff_ind = 2
    M = 3 # number of segments
    n_MCMC = 3
    mus = np.ones(M) * np.mean(seq1)
    vs = np.ones(M)

    # samples M-1 (M=mus.shape[0]) values from 0 to n-2 last index can't be a changepoint
    locs = helpers.sample_combinations(n-1, M-1, None) - 1
    # intialized using empirical segment means based on the sampled locations
    seg_means, _ = helpers.compute_seg_means(seq1, locs)

    a,b,c,d = metropolis_within_gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)
    e,f,g = gibbs(seq1, n_MCMC, n, M, mus, vs, alpha, beta, i, sd, locs, seg_means, diff_ind)