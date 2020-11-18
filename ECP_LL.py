import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

def ecp_ll(data = None, gamma = 0.01, max_m = 3, alpha = 0.9, t_start = 2, t_min = 1e-5, show_plot = True):

    y = []
    x = data
    x_f = x.flatten()
    n = x.shape[0]
    x_avg = np.mean(data, axis= 0)
    y.append(x_avg)
    T = t_start
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
        return theta

    def exists(item, list):
        for item_ in list:
            if np.allclose(item, item_):
                return True
        return False

    def make_list(arr):
        temp = []
        for i in range(0, len(arr), 2):
            temp.append(arr[i:i+2])
        return temp

    while(T >= t_min):
        #update associations
        temp = np.array([[-(d(x_, y_) + gamma * sum(d(y_, _y_) for _y_ in y))/T  for y_ in y] for x_ in x])
        c = np.max(temp, axis=1).reshape((-1, 1))
        temp -= c
        p_yx = np.exp(temp)
        Z = np.sum(p_yx, axis=1).reshape((-1, 1))
        p_yx = p_yx / Z

        #Plot algorithm
        if show_plot == True:
            rels = np.argmax(p_yx, axis= 1)
            clusters = [[x[i] for i in range(n) if rels[i] == int(j)] for j in range(len(y))]
            colors = cm.rainbow(np.linspace(0, 1, len(clusters)))
            ctr = 0
            for y_, cluster, color in zip(y, clusters, colors):
                ctr += 1
                if len(cluster) != 0:
                    cluster = np.array(cluster)
                    x_cord, y_cord = cluster[:, 0], cluster[:, 1]
                    plt.scatter(x_cord, y_cord, c=[color], edgecolors=None, s=30, alpha=0.15,
                                label='Cluster {0}'.format(ctr))
                y_x_cord, y_y_cord = y_[0], y_[1]
                plt.scatter(y_x_cord, y_y_cord, marker="s", edgecolors='black', c=[color], alpha=1)
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LL Clustering, gamma={0}, N={1}'.format(gamma, n))
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.pause(1e-5)
            plt.clf()

        #update centroids
        p_x_temp = (np.ones(n) * (1 / n)).reshape((-1, 1))
        p_x = np.diag(np.ones(n) * (1 / n))
        p_y = np.diag(np.sum(p_yx * p_x_temp, axis=0))
        p_xy = p_x @ p_yx @ inv(p_y)
        p_xy_ = np.kron(p_xy, i2)
        theta = calc_theta(len(y))
        temp_y = inv(theta) @ p_xy_.T @ x_f
        y = make_list(temp_y)

        #merge centroids
        if len(y) > 1:
            temp = []
            for y_ in y:
                if len(temp) == 0:
                    temp.append(y_)
                elif not exists(y_, temp) and len(temp) < max_m: temp.append(y_)
            y = temp.copy()

        #perturb centroids
        if len(y) < max_m:
            eps = 1e-3 * np.random.random(y[0].shape)
            y = [y_ - eps for y_ in y] + [y_ + eps for y_ in y]

        T *= alpha
    return y


