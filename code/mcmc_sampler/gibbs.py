import numpy as np
from scipy import stats as stat
import pickle as pk
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
import helpers
from itertools import combinations

def gibbs(seq=None, n_MCMC=None, n=None, M=None,
        mus=None, vs=None, alpha=None, beta=None, # priors
        seed=None):
    # initialization
    np.random.seed(seed)
    sample_locs = []
    sample_seg_means = []
    sample_seq_var = np.zeros(0)

    # samples M-1 (M=mus.shape[0]) values from 0 to n-2
    # last index can't be a changepoint
    locs = helpers.sample_combinations(n-1, M-1, seed) - 1

    # intialized using empirical segment means based on the sampled locations
    seg_means, _ = helpers.compute_seg_means(seq, locs)
    
    # beta is rate param, but np.random.gamma takes scale as input (1 / seg_var)
    # gam = np.random.gamma(alpha, 1/beta)
    gam = 1

    while sample_seq_var.shape[0] < n_MCMC:
        # no need to find combinations with the last observation as it 
        # won't be a change point
        combs = combinations(np.arange(seq.shape[0]-1), seg_means.shape[0]-1)
        combs = np.array(list(combs))
        locs_new = helpers.sample_locs(seq, seg_means, gam, seed, combs)
        seg_means_new, _ = helpers.sample_seg_means(seq, locs_new, mus, vs, gam, seed)
        # gam_new = helpers.sample_gam(seq, locs_new, seg_means_new, alpha, beta, seed)
        gam_new = 1

        sample_locs.append(locs_new)
        sample_seg_means.append(seg_means_new)
        sample_seq_var = np.hstack([sample_seq_var, np.exp(-np.log(gam_new))])
        locs = locs_new
        seg_means = seg_means_new
        gam = gam_new

        if sample_seq_var.shape[0] % 10 == 0:
            print(str(sample_seq_var.shape[0]) + " / " + str(n_MCMC))

    file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(seed)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    fi = open(path, 'wb')
    pk.dump((sample_locs, sample_seg_means, sample_seq_var), fi)
    fi.close()

    return sample_locs, sample_seg_means, sample_seq_var