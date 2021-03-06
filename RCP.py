import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv

def rcp(seed = 5, spread = 0.2, k_0 = 5e-5, alpha = 0.8, gamma = 0.4):
    Y = np.load('y_hist.npy', allow_pickle = True)
    time_series = np.load('time_series.npy', allow_pickle = True)

    X = time_series[0]
    y_0 = np.array(Y[0])
    m = len(Y[-1])

    np.random.seed(seed)
    eps = np.random.randn(m, 2) * spread
    y_0 = y_0 + eps

    horizon = len(X)
    T = 5
    alpha = alpha
    k_0 = k_0
    n = X[0].shape[0]
    eta = gamma*(m-1) + 1
    i2 = np.eye(2)

    theta = np.zeros((2*m, 2*m))
    for i in range(m):
        for j in range(m):
            i_start = i*2
            i_end = i_start + 2
            j_start = j*2
            j_end = j_start + 2
            if i == j:
                theta[i_start:i_end, j_start:j_end] = np.kron(eta, i2)
            else:
                theta[i_start:i_end, j_start:j_end] = np.kron(-gamma, i2)

    def d(x, y):
        return norm(x-y) ** 2

    y = y_0
    rcp_hist = []
    for t in range(horizon - 1):
        rcp_hist.append(y)
        x = X[t]
        phi = X[t+1] - X[t]
        phi_f = phi.flatten()
        temp = np.array([[-(d(x_, y_) + gamma * sum(d(y_, _y_) for _y_ in y)) / T for y_ in y] for x_ in x])
        c = np.max(temp, axis= 1).reshape((-1, 1))
        temp -= c
        p_yx = np.exp(temp)
        Z = np.sum(p_yx, axis = 1).reshape((-1, 1))
        p_yx = p_yx / Z
        p_yx_ = np.kron(p_yx, i2)

        p_x_temp = (np.ones(n)*(1/n)).reshape((-1, 1))
        p_x = np.diag(np.ones(n)*(1/n))
        p_y = np.diag(np.sum(p_yx * p_x_temp, axis = 0))
        p_y_ = np.kron(p_y, i2)

        p_xy = p_x @ p_yx @ inv(p_y)
        p_xy_ = np.kron(p_xy, i2)

        x_f = x.flatten()
        y_f = y.flatten()

        y_ = n * theta @ (y_f - inv(theta) @ p_xy_.T @ x_f)

        term1 = x_f - y_f @ p_yx_.T
        term2 = y_ @ p_y_ @ y_
        u = -(k_0 + (term1 @ phi_f)/term2) * y_

        y_f += u
        y = y_f.reshape((m, 2))
        T *= alpha

    np.save('rcp_hist', rcp_hist, allow_pickle = True)
    return rcp_hist