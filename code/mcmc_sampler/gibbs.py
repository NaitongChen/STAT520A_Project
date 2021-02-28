import numpy as np
from scipy import stats as stat
import pickle as pk
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
import helpers
from itertools import combinations
from time import perf_counter

def gibbs(seq=None, n_MCMC=None, n=None, M=None,
        mus=None, vs=None, alpha=None, beta=None, # priors
        seed=None, 
        sd=None,
        locs=None, seg_means=None, # init
        diff_ind=None):
    # initialization
    sample_locs = []
    sample_seg_means = []
    sample_time = []

    # beta is rate param, but np.random.gamma takes scale as input (1 / seg_var)
    # gam = np.random.gamma(alpha, 1/beta)
    gam = 1/(sd**2)

    # no need to find combinations with the last observation as it 
    # won't be a change point
    combs = combinations(np.arange(seq.shape[0]-1), seg_means.shape[0]-1)
    combs = np.array(list(combs))

    curr_time = 0
    # while sample_seq_var.shape[0] < n_MCMC:
    while np.sum(sample_time) < n_MCMC * 60:
        start_time = perf_counter()

        locs_new = helpers.sample_locs(seq, seg_means, gam, seed, combs)
        seg_means_new, _ = helpers.sample_seg_means(seq, locs_new, mus, vs, gam, seed)

        end_time = perf_counter()
        # gam_new = helpers.sample_gam(seq, locs_new, seg_means_new, alpha, beta, seed)

        sample_locs.append(locs_new)
        sample_seg_means.append(seg_means_new)
        locs = locs_new
        seg_means = seg_means_new
        sample_time.append(end_time - start_time)

        if sum(sample_time) - curr_time > 5:
            print(sum(sample_time))
            curr_time = sum(sample_time)

    file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(seed) + "_diffint" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    fi = open(path, 'wb')
    pk.dump((sample_locs, sample_seg_means, sample_time), fi)
    fi.close()

    return sample_locs, sample_seg_means, sample_time