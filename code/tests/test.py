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

# load data
M = 2
n = 50
sd = 5
seed = 1
file_name = "gaussian_mean_shift" + "_M" + str(M) + "_N" + str(n) + "_seed" + str(seed)
path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'sequences', file_name))
seq, locs, seg_means = pk.load(open(path, 'rb'))

n_MCMC = 2000
alpha = 1
beta = 1
mus = np.average(seq) * np.ones(M)
vs = np.ones(M)
seed = None
a,b,c,d = metropolis_within_gibbs(seq, n_MCMC, n, M, mus, vs, alpha, beta, seed)
e,f,g = gibbs(seq, n_MCMC, n, M, mus, vs, alpha, beta, seed)

print("hola")