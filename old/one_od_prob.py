import random
import numpy as np

seed = 0
np.random.seed(seed)
random.seed(seed)

N = 100  # users
M = 6   # available toll paths
T = np.random.uniform(size=M)*100   # toll vector
eps = 10**(-12)
C = 10000

p_old = np.ones(M) / M
print('Initial p_old =', p_old)
print('Initial p_old total probability =', p_old.sum())

p_new = p_old

h_old = C - T - N*p_old
h_new = h_old

star = False

while (np.abs(h_old - h_new) > eps).any() or not star:
    p_old = p_new
    h_old = h_new
    star = True

    # average payoff
    h_average = (h_old * p_old).sum()

    # updated probabilities
    p_new = p_old*h_old/h_average

    # updated payoff
    h_new = C - T - N*p_new

print('Toll vector T =', T)
print('p_new =', p_new)
print('p_new total probability =', p_new.sum())
