import numpy as np
from gaussian_mean_shift import generate_gaussian_mean_shift
import matplotlib.pyplot as plt

ks = np.array([25])
n = 50
means = np.array([40, 60])
sd = 5
seed = 1

seq,_,_ = generate_gaussian_mean_shift(ks, n, means, sd, seed)
plt.plot(np.arange(n)+1, seq)
plt.show()