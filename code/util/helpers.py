import numpy as np
from scipy import stats as stat
from scipy.spatial.distance import jensenshannon
from itertools import combinations
from numba import jit
from scipy.special.basic import ellipk
import mcmcse
import matplotlib.pyplot as plt
import pickle as pk
import os
import pandas as pd

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

def evidences(seq=None, locs_new=None, 
                        mus=None, vs=None, # priors
                        gam=None, seed=None):
    empirical_means, seg_sizes = compute_seg_means(seq, locs_new)

    log_liks = np.zeros(locs_new.shape[0] + 1)
    for i in range(log_liks.shape[0]):
        post_mean, post_sd = compute_post_mean_dist(vs, seg_sizes, gam, i, mus, empirical_means)

        log_liks[i] = stat.norm.logpdf(1, post_mean, post_sd)

    log_likelihoods = sequence_log_likelihood_vec(seq, locs_new, np.ones(mus.shape[0]), gam)
    pmus = stat.norm.logpdf(np.ones(mus.shape[0]), mus[0], vs[0])

    return np.sum(log_likelihoods + pmus - log_liks)

@jit(nopython=True)
def compute_post_mean_dist(vs, seg_sizes, gam, i, mus, empirical_means):
    post_var = (1 / ((1 / vs[i]) + (seg_sizes[i] * gam)))
    post_sd = np.sqrt(post_var)
    post_mean = ( ((mus[i] / vs[i]) + (gam * empirical_means[i] * seg_sizes[i])) /
                    ((1 / vs[i]) + (seg_sizes[i] * gam)) )

    # using log trick for numerical stability
    # post_var = np.exp(-np.log( np.exp(-np.log(vs[i])) + np.exp(np.log(seg_sizes[i]) + np.log(gam))))
    # post_sd = np.exp(0.5 * np.log(post_var))
    # post_mean = np.exp(np.log(np.exp(np.log(mus[i]) - np.log(vs[i])) + 
    #             np.exp(np.log(gam) + np.log(empirical_means[i]) + np.log(seg_sizes[i]))) + np.log(post_var))
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

def sequence_log_likelihood_vec(seq=None, locations=None, seg_means_new=None, gam_new=None):
    # gam_new**-0.5
    sd = np.exp(-0.5 * np.log(gam_new))
    var = np.exp(-np.log(gam_new))
    log_likelihoods = np.zeros(seg_means_new.shape[0])

    if locations.shape[0] == 0:
        lognormpdf = -np.log(sd) - 0.5 * np.log(2*np.pi) - (0.5 / var) * np.power(seq - seg_means_new[0], 2) 
        log_likelihoods[0] = np.sum(lognormpdf)
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
            log_likelihoods[i] = sumlognormpdf
            # log_likelihood = log_likelihood + np.sum(stat.norm.logpdf(temp, mean, sd))
    return log_likelihoods

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

def get_MWG(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    file_name = "MWG" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    MWG = pk.load(open(path, 'rb'))

    locs_mwg = np.array(MWG[0])
    times_mwg = np.array(MWG[3])
    times_mwg = np.cumsum(times_mwg)
    return locs_mwg, times_mwg

def get_Gibbs(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    Gibbs = pk.load(open(path, 'rb'))

    locs_gibbs = np.array(Gibbs[0])
    times_gibbs = np.array(Gibbs[2])
    times_gibbs = np.cumsum(times_gibbs)
    return locs_gibbs, times_gibbs

def process_data(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    # MWG
    file_name = "MWG" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
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
    file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
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
    file_name = "MWG" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    MWG = pk.load(open(path, 'rb'))

    locs_mwg = np.array(MWG[0])
    times_mwg = np.array(MWG[3])
    times_mwg = np.cumsum(times_mwg)

    # Gibbs
    file_name = "Gibbs" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
    Gibbs = pk.load(open(path, 'rb'))

    locs_gibbs = np.array(Gibbs[0])
    times_gibbs = np.array(Gibbs[2])
    times_gibbs = np.cumsum(times_gibbs)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text('MWG')
    ax1.plot(np.arange(locs_mwg.shape[0]), locs_mwg)
    ax2.title.set_text('Gibbs')
    ax2.plot(np.arange(locs_gibbs.shape[0]), locs_gibbs)

    file_name = "Trace" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(i) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.savefig(path)
    plt.clf()

def get_sequence(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    file_name = "gaussian_mean_shift" + "_M" + str(M) + "_N" + str(n) + "_seed" + str(218) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'sequences', file_name))
    dat = pk.load(open(path, 'rb'))

    seq = np.array(dat[0])
    ks = np.array(dat[1])
    means = np.array(dat[2])

    return seq, ks, means

def plot_sequence(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    seq,ks,means = get_sequence(M, n, n_MCMC, diff_ind, i)
    ms = np.zeros(seq.shape[0])
    for i in range(M):
        if i == 0:
            ms[:ks[i]+1] = means[i]
        elif i == M-1:
            ms[ks[i-1]+1:] = means[i]
        else:
            ms[ks[i-1]+1:ks[i]+1] = means[i]
    plt.clf()

    plt.plot(np.arange(seq.shape[0]), seq, alpha=0.7, marker='o', color='#1f77b4')
    plt.plot(np.arange(seq.shape[0]), ms, color='#ff7f0e')

    file_name = "sequence" + "_M" + str(M) + "_N" + str(n) + "_seed" + str(i) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.savefig(path)

def brute_force_est(seq, M, combs, gam=1):
    probs = np.zeros(combs.shape[0])
    mus = np.ones(M) * np.mean(seq)
    vs = np.ones(M)
    for i in np.arange(combs.shape[0]):
        log_liks = evidences(seq, combs[i], mus, vs, gam=1, seed=None)
        probs[i] = log_liks

        # seg_means,_ = compute_seg_means(seq, combs[i])
        # probs[i] = sequence_log_likelihood(seq, combs[i], seg_means, gam)

    probs_shifted = probs - np.min(probs)
    probs_shifted2 = probs_shifted - np.max(probs_shifted)

    eps = 10e-16
    n = seq.shape[0]
    cap = np.log(eps / n)

    truncated_prob = np.zeros(combs.shape[0])
    truncated_prob[probs_shifted2 >= cap] = np.exp(probs_shifted2[probs_shifted2 >= cap])
    
    probs_normed = truncated_prob / np.sum(truncated_prob)

    return probs_normed

def compare_posteriors(M=None, n=None, n_MCMC=None, diff_ind=None, i=None):
    seq,_,_ = get_sequence(M, n, n_MCMC, diff_ind, i)
    combs = combinations(np.arange(seq.shape[0]-1), M-1)
    combs = np.array(list(combs))
    post = brute_force_est(seq, M, combs)

    locs_MWG,_ = get_MWG(M, n, n_MCMC, diff_ind, i)
    locs_Gibbs,_ = get_Gibbs(M, n, n_MCMC, diff_ind, i)

    MWG_unique, counts = np.unique(locs_MWG, axis=0, return_counts=True)
    post_MWG = np.zeros(combs.shape[0])
    for i in np.arange(MWG_unique.shape[0]):
        if MWG_unique.shape[1] == 1:
            arg = np.argwhere(combs == MWG_unique[i])
        else:
            arg = np.arange(combs.shape[0])
            for j in np.arange(MWG_unique.shape[1]):
                arg = np.intersect1d(np.argwhere(combs[:,j] == MWG_unique[i][j]), arg)
        if arg.ravel().shape[0] > 1:
            post_MWG[arg.ravel()[0]] = counts[i]
        else:
            post_MWG[arg[0]] = counts[i]
    post_MWG = post_MWG / np.sum(post_MWG)

    Gibbs_unique, counts = np.unique(locs_Gibbs, axis=0, return_counts=True)
    post_Gibbs = np.zeros(combs.shape[0])
    for i in np.arange(Gibbs_unique.shape[0]):
        if Gibbs_unique.shape[1] == 1:
            arg = np.argwhere(combs == Gibbs_unique[i])
        else:
            arg = np.arange(combs.shape[0])
            for j in np.arange(Gibbs_unique.shape[1]):
                arg = np.intersect1d(np.argwhere(combs[:,j] == Gibbs_unique[i][j]), arg)
        if arg.ravel().shape[0] > 1:
            post_Gibbs[arg.ravel()[0]] = counts[i]
        else:
            post_Gibbs[arg[0]] = counts[i]
    post_Gibbs = post_Gibbs / np.sum(post_Gibbs)

    plt.clf()
    plt.plot(np.arange(post.shape[0]), post, 'o-', alpha=0.7, label="BF")
    plt.plot(np.arange(post.shape[0]), post_Gibbs, 'o-', alpha=0.7, label="Gibbs")
    plt.plot(np.arange(post.shape[0]), post_MWG, 'o-', alpha=0.7, label="MWG")
    plt.legend()

    file_name = "Posterior" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(0) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.savefig(path)

def brute_force_marginal(seq, M, combs, post):
    post_margs = np.zeros((M-1, seq.shape[0]))
    for j in np.arange(M-1):
        for i in np.arange(seq.shape[0]):
            combs_sub = combs[:,j]
            args = np.argwhere(combs_sub == i)
            post_margs[j, i] = np.sum(post[args])
    return post_margs

def compare_posteriors_marginal(M=None, n=None, n_MCMC=None, diff_ind=None, seed=None):
    seq,_,_ = get_sequence(M, n, n_MCMC, diff_ind, 218)
    combs = combinations(np.arange(seq.shape[0]-1), M-1)
    combs = np.array(list(combs))
    post = brute_force_est(seq, M, combs)
    post_marginal = brute_force_marginal(seq, M, combs, post)

    locs_MWG,_ = get_MWG(M, n, n_MCMC, diff_ind, seed)
    locs_Gibbs,_ = get_Gibbs(M, n, n_MCMC, diff_ind, seed)

    post_MWG = np.zeros((M-1, seq.shape[0]))
    for j in np.arange(M-1):
        for i in np.arange(seq.shape[0]):
            combs_sub = locs_MWG[:,j]
            args = np.argwhere(combs_sub == i)
            post_MWG[j, i] = args.shape[0]
        post_MWG[j,:] = post_MWG[j,:] / np.sum(post_MWG[j,:])

    post_Gibbs = np.zeros((M-1, seq.shape[0]))
    for j in np.arange(M-1):
        for i in np.arange(seq.shape[0]):
            combs_sub = locs_Gibbs[:,j]
            args = np.argwhere(combs_sub == i)
            post_Gibbs[j, i] = args.shape[0]
        post_Gibbs[j,:] = post_Gibbs[j,:] / np.sum(post_Gibbs[j,:])

    plt.clf()
    fig, axs = plt.subplots(M-1, sharex=True)
    for j in np.arange(M-1):
        if M-1 > 1:
            axs[j].plot(np.arange(seq.shape[0]), post_marginal[j,:], 'o-', alpha=0.5, label="BF", color='#2ca02c')
            axs[j].plot(np.arange(seq.shape[0]), post_Gibbs[j,:], 'o-', alpha=0.5, label="Gibbs", color='#1f77b4')
            axs[j].plot(np.arange(seq.shape[0]), post_MWG[j,:], 'o-', alpha=0.5, label="MWG", color='#ff7f0e')
        else:
            axs.plot(np.arange(seq.shape[0]), post_marginal[j,:], 'o-', alpha=0.5, label="BF", color='#2ca02c')
            axs.plot(np.arange(seq.shape[0]), post_Gibbs[j,:], 'o-', alpha=0.5, label="Gibbs", color='#1f77b4')
            axs.plot(np.arange(seq.shape[0]), post_MWG[j,:], 'o-', alpha=0.5, label="MWG", color='#ff7f0e')

    file_name = "Posterior" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(seed) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.legend()
    plt.savefig(path)

def plottime(M, n, n_MCMC, diff_ind, inds):
    min_len_MWG = np.inf
    min_len_Gibbs = np.inf
    for i in inds:
        _, times_MWG = get_MWG(M, n, n_MCMC, diff_ind, i)
        _, times_Gibbs = get_Gibbs(M, n, n_MCMC, diff_ind, i)
        if times_MWG.shape[0] < min_len_MWG:
            min_len_MWG = times_MWG.shape[0]
        if times_Gibbs.shape[0] < min_len_Gibbs:
            min_len_Gibbs = times_Gibbs.shape[0]

    times_MWGs_array = np.zeros((inds.shape[0], min_len_MWG))
    times_Gibbss_array = np.zeros((inds.shape[0], min_len_Gibbs))

    for i in inds:
        _, times_MWG = get_MWG(M, n, n_MCMC, diff_ind, i)
        _, times_Gibbs = get_Gibbs(M, n, n_MCMC, diff_ind, i)
        times_MWGs_array[i,:] = times_MWG[:min_len_MWG]
        times_Gibbss_array[i,:] = times_Gibbs[:min_len_Gibbs]

    times_MWGs_up = np.percentile(times_MWGs_array, 75, axis=0)
    times_MWGs_mid = np.median(times_MWGs_array, axis=0)
    times_MWGs_low = np.percentile(times_MWGs_array, 25, axis=0)

    times_Gibbss_up = np.percentile(times_Gibbss_array, 75, axis=0)
    times_Gibbss_mid = np.median(times_Gibbss_array, axis=0)
    times_Gibbss_low = np.percentile(times_Gibbss_array, 25, axis=0)

    plt.clf()

    plt.plot(times_Gibbss_mid, np.arange(times_Gibbss_mid.shape[0]) + 1, label="Gibbs", color='#1f77b4')
    plt.plot(times_Gibbss_up, np.arange(times_Gibbss_up.shape[0]) + 1, color='#1f77b4', linestyle='dashed')
    plt.plot(times_Gibbss_low, np.arange(times_Gibbss_low.shape[0]) + 1, color='#1f77b4', linestyle='dashed')
    plt.fill_betweenx(np.arange(times_Gibbss_low.shape[0]) + 1, times_Gibbss_low, times_Gibbss_up, color='#1f77b4', alpha=0.5)

    plt.plot(times_MWGs_mid, np.arange(times_MWGs_mid.shape[0]) + 1, label="MWG", color='#ff7f0e')
    plt.plot(times_MWGs_up, np.arange(times_MWGs_up.shape[0]) + 1, color='#ff7f0e', linestyle='dashed')
    plt.plot(times_MWGs_low, np.arange(times_MWGs_low.shape[0]) + 1, color='#ff7f0e', linestyle='dashed')
    plt.fill_betweenx(np.arange(times_MWGs_low.shape[0]) + 1, times_MWGs_low, times_MWGs_up, color='#ff7f0e', alpha=0.5)

    plt.yscale("log")
    plt.legend(loc='lower right')
    plt.ylabel("# sample")
    plt.xlabel("time(s)")

    file_name = "SampleTime" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(0) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.savefig(path)

def approx_post(locs_MWG, combs):
    MWG_unique, counts = np.unique(locs_MWG, axis=0, return_counts=True)
    post_MWG = np.zeros(combs.shape[0])
    for i in np.arange(MWG_unique.shape[0]):
        if MWG_unique.shape[1] == 1:
            arg = np.argwhere(combs == MWG_unique[i])
        else:
            arg = np.arange(combs.shape[0])
            for j in np.arange(MWG_unique.shape[1]):
                arg = np.intersect1d(np.argwhere(combs[:,j] == MWG_unique[i][j]), arg)
        if arg.ravel().shape[0] > 1:
            post_MWG[arg.ravel()[0]] = counts[i]
        else:
            post_MWG[arg[0]] = counts[i]
    post_MWG = post_MWG / np.sum(post_MWG)
    return post_MWG

def compute_kl(post, locs, combs):
    post_approx = approx_post(locs, combs)
    return jensenshannon(post, post_approx)

def plot_kl(M, n, n_MCMC, diff_ind, inds):
    seq,_,_ = get_sequence(M, n, n_MCMC, diff_ind, inds)
    combs = combinations(np.arange(seq.shape[0]-1), M-1)
    combs = np.array(list(combs))
    post = brute_force_est(seq, M, combs)

    max_len_MWG = 0
    max_len_Gibbs = 0
    for i in inds:
        _, times_MWG = get_MWG(M, n, n_MCMC, diff_ind, i)
        _, times_Gibbs = get_Gibbs(M, n, n_MCMC, diff_ind, i)
        if times_MWG.shape[0] > max_len_MWG:
            max_len_MWG = times_MWG.shape[0]
        if times_Gibbs.shape[0] > max_len_Gibbs:
            max_len_Gibbs = times_Gibbs.shape[0]

    skip_mwg = int(np.ceil(max_len_MWG / 200))
    skip_gibbs = int(np.ceil(max_len_Gibbs / 200))

    times_MWGs_array = []
    times_Gibbss_array = []
    kls_MWGs_array = []
    kls_Gibbss_array = []
    locs_MWGs_array = []
    locs_Gibbss_array = []

    for i in inds:
        locs_MWG, times_MWG = get_MWG(M, n, n_MCMC, diff_ind, i)
        locs_Gibbs, times_Gibbs = get_Gibbs(M, n, n_MCMC, diff_ind, i)
        locs_MWGs_array.append(locs_MWG)
        locs_Gibbss_array.append(locs_Gibbs)
        times_MWGs_array.append(times_MWG[::skip_mwg])
        times_Gibbss_array.append(times_Gibbs[::skip_gibbs])

    for j in inds:
        locs_mwg = locs_MWGs_array[j]
        locs_gibbs = locs_Gibbss_array[j]
        kls_MWG = np.zeros(0)
        kls_Gibbs = np.zeros(0)
        for i in np.arange(times_MWGs_array[j].shape[0]):
            print(str(i) + "/" + str(times_MWGs_array[j].shape[0]))
            kls_MWG = np.hstack([kls_MWG, compute_kl(post, locs_mwg[:(i+1)*skip_mwg,:], combs)])
        for i in np.arange(times_Gibbss_array[j].shape[0]):
            print(str(i) + "/" + str(times_Gibbss_array[j].shape[0]))
            kls_Gibbs = np.hstack([kls_Gibbs, compute_kl(post, locs_gibbs[:(i+1)*skip_gibbs,:], combs)])

        kls_MWGs_array.append(kls_MWG)
        kls_Gibbss_array.append(kls_Gibbs)

    plt.clf()

    for i in inds:
        if i == 0:
            plt.plot(times_Gibbss_array[i], kls_Gibbss_array[i], label="Gibbs", color='#1f77b4', alpha=0.5)
            plt.plot(times_MWGs_array[i], kls_MWGs_array[i], label="MWG", color='#ff7f0e', alpha=0.5)
        else:
            plt.plot(times_Gibbss_array[i], kls_Gibbss_array[i], color='#1f77b4', alpha=0.5)
            plt.plot(times_MWGs_array[i], kls_MWGs_array[i], color='#ff7f0e', alpha=0.5)

    plt.yscale("log")
    plt.legend(loc='lower right')
    plt.ylabel("JS divergence")
    plt.xlabel("time(s)")

    file_name = "KL" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(0) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.savefig(path)

def plot_ess_se(M, n, n_MCMC, diff_ind, inds):
    ess_mwg = np.zeros((inds.shape[0], M-1))
    mcse_mwg = np.zeros((inds.shape[0], M-1))
    ess_gibbs = np.zeros((inds.shape[0], M-1))
    mcse_gibbs = np.zeros((inds.shape[0], M-1))
    for i in inds:
        file_name = "seed" + str(i) + "_ess_se.csv"
        path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'posterior_samples', file_name))
        dat = pd.read_csv(path)
        if M == 2 and n == 50:
            ess_mwg[i,0] = dat["ess first cp"][3]
            ess_gibbs[i,0] = dat["ess first cp"][2]
            mcse_mwg[i,0] = dat["mcse first cp"][3]
            mcse_gibbs[i,0] = dat["mcse first cp"][2]
        elif M == 2 and n == 20000:
            ess_mwg[i,0] = dat["ess first cp"][1]
            ess_gibbs[i,0] = dat["ess first cp"][0]
            mcse_mwg[i,0] = dat["mcse first cp"][1]
            mcse_gibbs[i,0] = dat["mcse first cp"][0]
        elif M == 3 and n == 100:
            ess_mwg[i,0] = dat["ess first cp"][5]
            ess_gibbs[i,0] = dat["ess first cp"][4]
            mcse_mwg[i,0] = dat["mcse first cp"][5]
            mcse_gibbs[i,0] = dat["mcse first cp"][4]

            ess_mwg[i,1] = dat["ess second cp"][5]
            ess_gibbs[i,1] = dat["ess second cp"][4]
            mcse_mwg[i,1] = dat["mcse second cp"][5]
            mcse_gibbs[i,1] = dat["mcse second cp"][4]
        elif M == 4 and n == 100:
            ess_mwg[i,0] = dat["ess first cp"][7]
            ess_gibbs[i,0] = dat["ess first cp"][6]
            mcse_mwg[i,0] = dat["mcse first cp"][7]
            mcse_gibbs[i,0] = dat["mcse first cp"][6]

            ess_mwg[i,1] = dat["ess second cp"][7]
            ess_gibbs[i,1] = dat["ess second cp"][6]
            mcse_mwg[i,1] = dat["mcse second cp"][7]
            mcse_gibbs[i,1] = dat["mcse second cp"][6]

            ess_mwg[i,2] = dat["ess third cp"][7]
            ess_gibbs[i,2] = dat["ess third cp"][6]
            mcse_mwg[i,2] = dat["mcse third cp"][7]
            mcse_gibbs[i,2] = dat["mcse third cp"][6]
        else:
            ess_mwg[i,0] = dat["ess first cp"][9]
            ess_gibbs[i,0] = dat["ess first cp"][8]
            mcse_mwg[i,0] = dat["mcse first cp"][9]
            mcse_gibbs[i,0] = dat["mcse first cp"][8]

            ess_mwg[i,1] = dat["ess second cp"][9]
            ess_gibbs[i,1] = dat["ess second cp"][8]
            mcse_mwg[i,1] = dat["mcse second cp"][9]
            mcse_gibbs[i,1] = dat["mcse second cp"][8]

            ess_mwg[i,2] = dat["ess third cp"][9]
            ess_gibbs[i,2] = dat["ess third cp"][8]
            mcse_mwg[i,2] = dat["mcse third cp"][9]
            mcse_gibbs[i,2] = dat["mcse third cp"][8]
    plt.clf()
    fig, axs = plt.subplots(2, M-1, sharex=True)
    for i in np.arange(2):
        for j in np.arange(M-1):
            if i == 0: # ess
                if M-1 > 1:
                    axs[i, j].scatter(np.zeros(inds.shape[0]), ess_mwg[:,j], label="MWG", color='#ff7f0e')
                    axs[i, j].scatter(np.ones(inds.shape[0]), ess_gibbs[:,j], label="Gibbs", color='#1f77b4')
                    axs[i, j].title.set_text("ESS of changepoint" + str(j+1))
                else:
                    axs[i].scatter(np.zeros(inds.shape[0]), ess_mwg[:,j], label="MWG", color='#ff7f0e')
                    axs[i].scatter(np.ones(inds.shape[0]), ess_gibbs[:,j], label="Gibbs", color='#1f77b4')
                    axs[i].title.set_text("ESS of changepoint" + str(j+1))
            else: # mcse
                if M-1 > 1:
                    axs[i, j].scatter(np.zeros(inds.shape[0]), mcse_mwg[:,j], label="MWG", color='#ff7f0e')
                    axs[i, j].scatter(np.ones(inds.shape[0]), mcse_gibbs[:,j], label="Gibbs", color='#1f77b4')
                    axs[i, j].title.set_text("MCSE of changepoint" + str(j+1))
                else:
                    axs[i].scatter(np.zeros(inds.shape[0]), mcse_mwg[:,j], label="MWG", color='#ff7f0e')
                    axs[i].scatter(np.ones(inds.shape[0]), mcse_gibbs[:,j], label="Gibbs", color='#1f77b4')
                    axs[i].title.set_text("MCSE of changepoint" + str(j+1))
    file_name = "ess_se" + "_M" + str(M) + "_N" + str(n) + "_NMCMC" + str(n_MCMC) + "_seed" + str(0) + "_diffind" + str(diff_ind)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'plots', file_name))
    plt.legend()
    plt.savefig(path)