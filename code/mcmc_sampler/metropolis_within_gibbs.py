import numpy as np
from scipy import stats as stat
import pickle as pk
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../util'))
import helpers
from time import perf_counter

def metropolis_within_gibbs(seq=None, n_MCMC=None, n=None, M=None,
                            mus=None, vs=None, alpha=None, beta=None, # priors
                            seed=None,
                            sd=None,
                            locs=None, seg_means=None, # init
                            diff_ind=None):
    # initialization
    accept_count = 0
    sample_locs = []
    sample_seg_means = []
    sample_time = []
    
    # beta is rate param, but np.random.gamma takes scale as input (1 / seg_var)
    # gam = np.random.gamma(alpha, 1/beta)
    gam = 1/(sd**2)

    # while sample_seq_var.shape[0] < n_MCMC:
    while sum(sample_time) < n_MCMC * 60:
        start_time = perf_counter()
        # proposal
        locs_new = helpers.sample_combinations(n-1, M-1, seed) - 1
        
        # acceptance prob
        log_new = helpers.sequence_log_likelihood(seq, locs_new, seg_means, gam)
        log_old = helpers.sequence_log_likelihood(seq, locs, seg_means, gam)

        accept_prob = np.exp(log_new - log_old)
        unif = np.random.uniform(size=1)

        if unif <= accept_prob:
            sample_locs.append(locs_new)
            locs = locs_new
            accept_count += 1
        else:
            sample_locs.append(locs)
        
        # sampling the rest parameters using Gibbs
        seg_means_new, _ = helpers.sample_seg_means(seq, locs_new, mus, vs, gam, seed)

        end_time = perf_counter()
        # gam_new = helpers.sample_gam(seq, locs_new, seg_means_new, alpha, beta, seed)

        sample_seg_means.append(seg_means_new)
        seg_means = seg_means_new
        sample_time.append(end_time - start_time)

        # if len(sample_time) % 1000 == 0:
        #     print(sum(sample_time))
    
    accept_prop = accept_count / n_MCMC

    file_name = "MWG" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(seed) + "_diffint" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    fi = open(path, 'wb')
    pk.dump((sample_locs, sample_seg_means, accept_prop, sample_time), fi)
    fi.close()

    return sample_locs, sample_seg_means, accept_prop, sample_time