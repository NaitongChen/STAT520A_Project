import numpy as np
from gaussian_mean_shift import generate_gaussian_mean_shift
import matplotlib.pyplot as plt

ks = np.array([10, 25, 40])
n = 200
means = np.array([5, 7, 10, 15])
sd = 1
seed = 1

seq,_,_ = generate_gaussian_mean_shift(ks, n, means, sd, seed)
plt.plot(np.arange(n)+1, seq)
plt.show()