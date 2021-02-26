import numpy as np
from gaussian_mean_shift import generate_gaussian_mean_shift
import matplotlib.pyplot as plt

ks = np.array([30, 50, 70])
n = 100
means = np.array([5, 7, 9, 11])
sd = 1
seed = 1
diff_ind = 1
np.random.seed(seed)
seq,_,_ = generate_gaussian_mean_shift(ks, n, means, sd, seed, diff_ind)
plt.plot(np.arange(n)+1, seq)
plt.show()