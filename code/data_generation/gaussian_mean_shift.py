import numpy as np
import pickle as pk
import os

def generate_gaussian_mean_shift(ks=None, n=None, means=None, sd=None, seed=None):
    # assumes the list of changepoint locations ks are in an increasing order
    # changepoints are stopping times
    if ks.shape[0] != 0:
        if ks is None or n is None or ks[ks.shape[0]-1] > n-1:
            raise Exception("ks and n need to be defined such that all values in ks are <= n")
    if means is None or sd is None or means.shape[0] != ks.shape[0]+1:
        raise Exception("means and sd need to be defined so the means and")

    seq = np.zeros(n)
    M = means.shape[0] # number of segments

    if M == 1:
        temp = np.copy(np.random.normal(means[0], sd, seq.shape[0]))
        seq = temp
    else:
        for i in range(M):
            if i == 0:
                temp = np.copy(np.random.normal(means[i], sd, ks[i]+1))
                seq[:ks[i]+1] = temp
            elif i == M-1:
                temp = np.copy(np.random.normal(means[i], sd, n-ks[i-1]-1))
                seq[ks[i-1]+1:] = temp
            else:
                temp = np.copy(np.random.normal(means[i], sd, ks[i]-ks[i-1]))
                seq[ks[i-1]+1:ks[i]+1] = temp

    file_name = "gaussian_mean_shift" + "_M" + str(M) + "_N" + str(n) + "_seed" + str(seed)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'data', 'sequences', file_name))
    fi = open(path, 'wb')
    pk.dump((seq, ks, means), fi)
    fi.close()
    return seq, ks, means