import numpy as np
from scipy import stats as stat
from itertools import combinations
from numba import jit
import mcmcse
import matplotlib.pyplot as plt
import pickle as pk
import os

"""
samples uniformly a combination of m values from 1 to n
"""
def sample_combinations(n=None, m=None, seed=None):
    ret = np.zeros(0)
    for j in np.arange(n-m+1, n+1):
        t = np.random.randint(1, j+1)
        if not np.isin(t, ret):
            ret = np.hstack([ret, t])
        else:
            ret = np.hstack([ret, j])
    ret = np.sort(ret)

    return ret.astype(int)

"""
computes segment means given the sequence and changepoint locations;
also returns number of observations in each segment
"""
@jit(nopython=True)
def compute_seg_means(seq=None, locations=None):
    means = np.zeros(locations.shape[0] + 1)
    sizes = np.zeros(locations.shape[0] + 1)
    if locations.shape[0] == 0:
        means[0] = np.mean(seq)
        sizes[0] = seq.shape[0]
    else:
        for i in range(means.shape[0]):
            if i == 0:
                means[i] = np.mean(seq[:locations[i] + 1])
                temp = seq[:locations[i] + 1]
                sizes[i] = temp.shape[0]
            elif i == means.shape[0] - 1:
                means[i] = np.mean(seq[locations[i - 1] + 1:])
                temp = seq[locations[i - 1] + 1:]
                sizes[i] = temp.shape[0]
            else:
                means[i] = np.mean(seq[locations[i - 1] + 1: locations[i] + 1])
                temp = seq[locations[i - 1] + 1: locations[i] + 1]
                sizes[i] = temp.shape[0]
    return means, sizes

"""
samples from posterior segment means;
also returns number of observations in each segment
"""
def sample_seg_means(seq=None, locs_new=None, 
                        mus=None, vs=None, # priors
                        gam=None, seed=None):
    empirical_means, seg_sizes = compute_seg_means(seq, locs_new)

    means = np.zeros(locs_new.shape[0] + 1)
    for i in range(means.shape[0]):
        post_mean, post_sd = compute_post_mean_dist(vs, seg_sizes, gam, i, mus, empirical_means)

        means[i] = np.random.normal(post_mean, post_sd)

    return means, seg_sizes

@jit(nopython=True)
def compute_post_mean_dist(vs, seg_sizes, gam, i, mus, empirical_means):
    # post_var = (1 / ((1 / vs[i]) + (seg_sizes[i] * gam)))
    # post_sd = np.sqrt(post_var)
    # post_mean = ( ((mus[i] / vs[i]) + (gam * empirical_means[i] * seg_sizes[i])) /
    #                 ((1 / vs[i]) + (seg_sizes[i] * gam)) )

    # using log trick for numerical stability
    post_var = np.exp(-np.log( np.exp(-np.log(vs[i])) + np.exp(np.log(seg_sizes[i]) + np.log(gam))))
    post_sd = np.exp(0.5 * np.log(post_var))
    post_mean = np.exp(np.log(np.exp(np.log(mus[i]) - np.log(vs[i])) + 
                np.exp(np.log(gam) + np.log(empirical_means[i]) + np.log(seg_sizes[i]))) + np.log(post_var))
    return post_mean, post_sd

"""
computes sequence log likelihood
"""
@jit(nopython=True)
def sequence_log_likelihood(seq=None, locations=None, seg_means_new=None, gam_new=None):
    # gam_new**-0.5
    sd = np.exp(-0.5 * np.log(gam_new))
    var = np.exp(-np.log(gam_new))
    log_likelihood = 0

    if locations.shape[0] == 0:
        lognormpdf = -np.log(sd) - 0.5 * np.log(2*np.pi) - (0.5 / var) * np.power(seq - seg_means_new[0], 2) 
        log_likelihood = np.sum(lognormpdf)
        # log_likelihood = np.sum(stat.norm.logpdf(seq, seg_means_new[0], sd))
    else:
        for i in range(seg_means_new.shape[0]):
            if i == 0:
                mean = seg_means_new[i]
                temp = seq[:locations[i] + 1]
            elif i == seg_means_new.shape[0] - 1:
                mean = seg_means_new[i]
                temp = seq[locations[i - 1] + 1:]
            else:
                mean = seg_means_new[i]
                temp = seq[locations[i - 1] + 1: locations[i] + 1]

            lognormpdf = -np.log(sd) - 0.5 * np.log(2*np.pi) - (0.5 / var) * np.power(temp - mean, 2) 
            sumlognormpdf = np.sum(lognormpdf)
            log_likelihood = log_likelihood + sumlognormpdf
            # log_likelihood = log_likelihood + np.sum(stat.norm.logpdf(temp, mean, sd))
    return log_likelihood

"""
samples changepoint locations by computing its conditional distribution
"""
def sample_locs(seq=None, seg_means=None, gam=None, seed=None, combs=None):
    probs = compute_probs(combs, seq, seg_means, gam)
    # probs = np.exp(np.log(probs) - np.log(np.sum(probs)))

    # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
    probs_shifted = probs - np.min(probs)
    probs_shifted2 = probs_shifted - np.max(probs_shifted)

    eps = 10e-16
    n = seq.shape[0]
    cap = np.log(eps / n)

    truncated_prob = np.zeros(combs.shape[0])
    truncated_prob[probs_shifted2 >= cap] = np.exp(probs_shifted2[probs_shifted2 >= cap])
    
    probs_normed = truncated_prob / np.sum(truncated_prob)

    selection = np.random.choice(np.arange(combs.shape[0]), size=1, p=probs_normed)[0]
    return combs[selection,:]

@jit(nopython=True)
def compute_probs(combs=None, seq=None, seg_means=None, gam=None):
    probs = np.zeros(combs.shape[0])
    for i in np.arange(combs.shape[0]):
        # probs[i] = np.exp(sequence_log_likelihood(seq, combs[i], seg_means, gam))
        probs[i] = sequence_log_likelihood(seq, combs[i], seg_means, gam)
        # print(i)
    return probs

def generate_means(diff=None, sd=None, num_seg=None):
    means = np.zeros(num_seg)
    for i in np.arange(num_seg):
        means[i] = np.random.normal(10, np.sqrt(diff))
    return np.sort(means)

# @jit(nopython=True)
def compute_ess(x, every):
    tot_length = x[::every].shape[0]
    esses = np.zeros((tot_length, x.shape[1]))
    ses = np.zeros((tot_length, x.shape[1]))
    for i in np.arange(tot_length):
        print(str(i) + "/" + str(tot_length))
        esses[i,:], ses[i,:] = mcmcse.ess(x[:(i+1)*every, :])
    return esses, ses

def process_data(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    # MWG
    file_name = "MWG" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffint" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    MWG = pk.load(open(path, 'rb'))

    locs_mwg = np.array(MWG[0])
    times_mwg = np.array(MWG[3])
    times_mwg = np.cumsum(times_mwg)

    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name+ "_locs.csv"))
    np.savetxt(path, locs_mwg, delimiter=",")
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name+ "_cumtime"))
    fi = open(path, 'wb')
    pk.dump((times_mwg), fi)
    fi.close()

    # Gibbs
    file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffint" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    Gibbs = pk.load(open(path, 'rb'))

    locs_gibbs = np.array(Gibbs[0])
    times_gibbs = np.array(Gibbs[2])
    times_gibbs = np.cumsum(times_gibbs)

    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name+ "_locs.csv"))
    np.savetxt(path, locs_gibbs, delimiter=",")
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name+ "_cumtime"))
    fi = open(path, 'wb')
    pk.dump((times_gibbs), fi)
    fi.close()

def plot_trace(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    # MWG
    file_name = "MWG" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffint" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    MWG = pk.load(open(path, 'rb'))

    locs_mwg = np.array(MWG[0])
    times_mwg = np.array(MWG[3])
    times_mwg = np.cumsum(times_mwg)

    locs_mwg = locs_mwg[locs_mwg.shape[0]-5000:,:]
    plt.plot(np.arange(locs_mwg.shape[0]), locs_mwg, alpha=0.7, marker='o')
    plt.show()

    # Gibbs
    # file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffint" + str(diff_ind)
    # path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    # Gibbs = pk.load(open(path, 'rb'))

    # locs_gibbs = np.array(Gibbs[0])
    # times_gibbs = np.array(Gibbs[2])
    # times_gibbs = np.cumsum(times_gibbs)

    # plt.plot(np.arange(locs_gibbs.shape[0]), locs_gibbs)
    # plt.show()