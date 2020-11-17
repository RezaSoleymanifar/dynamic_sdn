import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv

def ecp_ll(data = None, gamma = 0.01, max_m = 3, alpha = 0.9, t_start = 2, t_min = 1e-5):

    y = []
    x = data
    x_f = x.flatten()
    n = x.shape(0)
    x_avg = np.mean(data, axis= 0)
    y.append(x_avg)
    T = t_start
    m = len(y)
    i2 = np.eye(2)


    def d(x, y):
        return norm(x-y)**2

    def calc_theta(m):
        eta = gamma * (m - 1) + 1
        theta = np.zeros((2 * m, 2 * m))

        for i in range(m):
            for j in range(m):
                i_start = i * 2
                i_end = i_start + 2
                j_start = j * 2
                j_end = j_start + 2
                if i == j:
                    theta[i_start:i_end, j_start:j_end] = np.kron(eta, i2)
                else:
                    theta[i_start:i_end, j_start:j_end] = np.kron(-gamma, i2)

    def exists():
        pass

    while(T >= t_min):
        #update associations
        temp = np.array([[-(d(x_, y_) + gamma * sum(d(y_, _y_) for _y_ in y))/T  for y_ in y] for x_ in x])
        c = np.max(temp, axis=1).reshape((-1, 1))
        temp -= c
        p_yx = np.exp(temp)
        Z = np.sum(p_yx, axis=1).reshape((-1, 1))
        p_yx = p_yx / Z

        #update centroids
        p_x_temp = (np.ones(n) * (1 / n)).reshape((-1, 1))
        p_x = np.diag(np.ones(n) * (1 / n))
        p_y = np.diag(np.sum(p_yx * p_x_temp, axis=0))
        p_xy = p_x @ p_yx @ inv(p_y)
        p_xy_ = np.kron(p_xy, i2)
        theta = calc_theta(len(y))
        y = inv(theta) @ p_xy_.T @ x_f

        #merge centroids
        if len(len(y) > 1):
            temp = []
            for y_ in y:
                if exists(y_, y):


        #perturb centroids
        if (len(y)< max_m)
        eps = np.random.random(y[0].shape)
        y = [y_ - eps for y_ in y] + [y_ + eps for y_ in y]


