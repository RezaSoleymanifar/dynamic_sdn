# %load gen_dataset.py
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt 
import numpy as np
import sys
from numpy.linalg import norm

# n_samples, centers, random_state, T, case, shuffle = sys.argv[1:]
def run_sim(seed1=20, seed2=22):
    n_samples, centers, T, case =  300, 3, 100, 'exp'

    X, labels = make_blobs(int(n_samples), centers=int(centers),
                           n_features=2, random_state = seed1,
                           center_box = (-0.7, 0.35), cluster_std = 0.07)

    X_, labels_ = make_blobs(int(n_samples), centers=int(centers),
                             n_features=2, random_state = seed2,
                             center_box = (-0.35, 0.7), cluster_std = 0.07,
                             shuffle = 'yes')

    label_vals = np.unique(labels)
    n_labels = len(labels)
    n, d = np.shape(X)
    K = np.zeros((n,d))

    ##K = 0.015 * np.random.rayleigh(0.5, size = (n,d)) + 0.015
    for label in label_vals:
        indices = np.array(np.argwhere(labels == label)).flatten()
        size = len(indices)
        K1 = 0.015 * np.random.rayleigh(0.5, size = (size,1)) + 0.015
    ##    K2 = 0.03 * np.random.rayleigh(0.5, size = (size,1)) + 0.01
        K2 = 2 * np.random.uniform(0.85, 1) * K1
        K_ = np.concatenate((K1, K2), axis = 1)
        temp = K_.T.copy()
        np.random.shuffle(temp)
        K_ = temp.T
        K[indices] = K_.copy()


    indices = {label:[labels == label] for label in label_vals}
    indices_ = {label:[labels_ == label] for label in label_vals}

    history = []


    def rand_one():
        return 1 if np.random.random() > 0.5 else -1


    temp = X_.copy()
    for label in label_vals:
        X_[indices[label]] = temp[indices_[label]]



    x_0 = X[:, 0]
    y_0 = X[:, 1]
    x_1 = X_[:, 0]
    y_1 = X_[:, 1]

    def exp_func(t, x_0, x_1, k, T):
        x_t = (x_0 - x_1) * np.exp(-k * t) + x_1
        return x_t

    def exp_grad(t, x_0, x_1, k, T):
        grad_t = (-k) * (x_0 - x_1) * np.exp(-k * t) + 10 * np.sin(t) / T

    def lin_func(t, x_0, x_1, k, T):
        x_t = ((x_1 - x_0) / T) * t + x_0
        return x_t

    def loc_func(t, idx):
        result = calc_func(t, X[idx], X_[idx], K[idx], T)
        return result

    def plot_system(X_new, t, T):
        x = X_new[:, 0]
        y = X_new[:, 1]
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.title('2D dynamical system (' + str(t) + '/' + str(T) + ')')
        plt.scatter(x, y, color = 'turquoise', alpha=0.25, edgecolors = 'black')
        plt.scatter(x_0, y_0, color = 'cyan', alpha = 0.02, edgecolors = 'purple')
        plt.scatter(x_1, y_1, color = 'purple', alpha = 0.03, edgecolors = 'cyan')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='black', linestyle='--', alpha = 0.15)
        plt.grid(b=True, which='minor', color='black', linestyle='-', alpha = 0.01)
        plt.pause(0.3)

    def move(t):
        X_new = np.zeros((n, d))
        for idx in range(n):
            X_new[idx, :] = loc_func(t, idx)
        return X_new

    if case == 'lin':
        calc_func = lin_func
    elif case == 'exp':
        calc_func = exp_func

    for t in range(0, T):
        X_new = move(t)
        history.append(X_new)
        np.save('time_series', [history, labels, K], allow_pickle = True)
        plot_system(X_new, t, T)
        plt.clf()
    return None
