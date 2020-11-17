import numpy as np
from numpy.linalg import norm
def ecp_ll(data = None, gamma = 0.01, max_m = 3, alpha = 0.9, t_start = 2, t_min = 1e-5):

    y = []
    x = data
    x_avg = np.mean(data, axis= 0)
    y.append(x_avg)
    T = t_start
    m = len(y)

    def d(x, y):
        return norm(x-y)

    while(T >= t_min):
        temp = np.array([[-(d(x_, y_) + gamma * sum(d(y_, _y_) for _y_ in y))/T  for y_ in y] for x_ in x])
        c = np.max(temp, axis=1).reshape((-1, 1))
        temp -= c
        p_yx = np.exp(temp)
        Z = np.sum(p_yx, axis=1).reshape((-1, 1))
        p_yx = p_yx / Z

        eps = np.random.random(y[0].shape)
        y = [y_ - eps for y_ in y] + [y_ + eps for y_ in y]


