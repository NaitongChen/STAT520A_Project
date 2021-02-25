import numpy as np
from scipy import stats as stat
from itertools import combinations
from numba import jit

"""
samples uniformly a combination of m values from 1 to n
"""
def sample_combinations(n=None, m=None, seed=None):
    np.random.seed(seed)
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
def compute_seg_means(seq=None, locations=None):
    means = np.zeros(locations.shape[0] + 1)
    sizes = np.zeros(locations.shape[0] + 1)
    if locations.shape[0] == 0:
        means[0] = np.average(seq)
        sizes[0] = seq.shape[0]
    else:
        for i in range(means.shape[0]):
            if i == 0:
                means[i] = np.average(seq[:locations[i] + 1])
                temp = seq[:locations[i] + 1]
                sizes[i] = temp.shape[0]
            elif i == means.shape[0] - 1:
                means[i] = np.average(seq[locations[i - 1] + 1:])
                temp = seq[locations[i - 1] + 1:]
                sizes[i] = temp.shape[0]
            else:
                means[i] = np.average(seq[locations[i - 1] + 1: locations[i] + 1])
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
    np.random.seed(seed)
    empirical_means, seg_sizes = compute_seg_means(seq, locs_new)

    means = np.zeros(locs_new.shape[0] + 1)
    for i in range(means.shape[0]):
        # post_var = (1 / ((1 / vs[i]) + (seg_sizes[i] * gam)))
        # post_sd = np.sqrt(post_var)
        # post_mean = ( ((mus[i] / vs[i]) + (gam * empirical_means[i] * seg_sizes[i])) /
        #                 ((1 / vs[i]) + (seg_sizes[i] * gam)) )

        # using log trick for numerical stability
        post_var = np.exp(-np.log( np.exp(-np.log(vs[i])) + np.exp(np.log(seg_sizes[i]) + np.log(gam))))
        post_sd = np.exp(0.5 * np.log(post_var))
        post_mean = np.exp(np.log(np.exp(np.log(mus[i]) - np.log(vs[i])) + 
                    np.exp(np.log(gam) + np.log(empirical_means[i]) + np.log(seg_sizes[i]))) + np.log(post_var))

        means[i] = np.random.normal(post_mean, post_sd)

    return means, seg_sizes

"""
samples from posterior segment variance
"""
def sample_gam(seq=None, locations=None, seg_means_new=None,
                alpha=None, beta=None, # priors
                seed=None):
    np.random.seed(seed)
    sum_of_squares = 0

    if locations.shape[0] == 0:
        sum_of_squares = np.sum((seq - seg_means_new[0])**2)
        # sum_of_squares = np.sum(np.exp(2 * np.log(np.absolute(seq - seg_means_new[0]))))
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
            
            seg_sum_of_squares = np.sum((temp - mean)**2)
            # seg_sum_of_squares = np.sum(np.exp(2 * np.log(np.absolute(temp - mean))))
            sum_of_squares = sum_of_squares + seg_sum_of_squares
    
    alpha_new = alpha + (seq.shape[0] / 2)
    beta_new = beta + (sum_of_squares / 2)
    # gam_new = np.random.gamma(alpha_new, 1/beta_new)
    gam_new = np.random.gamma(alpha_new, np.exp(-np.log(beta_new)))
    return gam_new

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