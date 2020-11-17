# %load testAlg.py
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def show_sim():
    y_indices = np.load('y_hist.npy', allow_pickle = True)
    time_series = np.load('time_series.npy', allow_pickle = True)
    rcp_hist = np.array(np.load('rcp_hist.npy', allow_pickle= True))

    t_series, labels = time_series[0], time_series[1]
    T = len(y_indices)

    data_start = t_series[0]
    data_final = t_series[-1]


    def plot_nodes(data, color, alpha, edgecolors, zorder, label):
        x = data[:,0]
        y = data[:, 1]
        plt.scatter(x, y, color = color, alpha=alpha,
                edgecolors = edgecolors, zorder = 2, label = label)

    def plot_ctrls(y_indices, t_series, labels, color, marker, alpha, label):
        label_vals = np.unique(labels)
        fac_ctrls = {label:[data[idx]
                for ctrl_indices, data in zip(y_indices, t_series)
                for idx in ctrl_indices
                if labels[idx] == label]
                for label in label_vals}
        ctr = 0
        for key in fac_ctrls:
            path = fac_ctrls[key]
            x = np.array(path)[:, 0]
            y = np.array(path)[:, 1]
            if ctr == 0:
                plt.plot(x, y, color=color, marker = marker,
                         alpha = alpha, label = label)
            else:
                plt.plot(x, y, color=color, marker = marker,
                         alpha = alpha)
            ctr += 1

    def plot_rcp(hist):
        m = hist.shape[1]
        paths = []
        for path_idx in range(m):
            path = hist[:, path_idx, :]
            paths.append(path)
        ctr = 0
        for path in paths:
            x = path[:, 0]
            y = path[:, 1]
            if ctr == 0:
                plt.plot(x, y, marker = '.', c = 'orange', alpha = 0.6, label = 'RCP')
            else:
                plt.plot(x, y, marker='.', c='orange', alpha=0.6)
            ctr += 1



    def find_match(new_pnt, history):
        memory = []
        for pnt in history:
            dist = norm(new_pnt - pnt)
            memory.append(dist)
        return np.argmin(memory)



    for idx in range(T-1):

        plot_rcp(rcp_hist[:idx+1, :, :])

        plot_ctrls(y_indices = y_indices[0:idx+1], t_series = t_series[0:idx+1],
                   labels = labels, color='darkorchid',
                   marker='.', alpha = 0.6, label = 'optimal')
        data = t_series[idx]

    ##    plt.scatter(d_x, d_y, color = 'turquoise', alpha=0.25,
    ##                edgecolors = 'black',
    ##                zorder = 2, label = 'data')
        plot_nodes(data, color='turquoise', alpha = 0.25,
                   edgecolors='black', zorder = 2, label='network nodes')
    ##    plt.scatter(d_start_x, d_start_y, color = 'cyan',
    ##                alpha = 0.02, edgecolors = 'purple',
    ##                zorder = 1, label = 'start')
        plot_nodes(data_start, color='cyan', alpha = 0.04,
               edgecolors='purple', zorder = 1, label='start')
    ##    plt.scatter(d_final_x, d_final_y, color = 'purple',
    ##                alpha = 0.03, edgecolors = 'cyan',
    ##                zorder = 1, label = 'end')


        plot_nodes(data_final, color='purple', alpha = 0.04,
               edgecolors='cyan', zorder = 1, label='end')


        plt.minorticks_on()
        plt.grid(b=True, which='major', color='black', linestyle='--', alpha = 0.15)
        plt.grid(b=True, which='minor', color='black', linestyle='-', alpha = 0.01)
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.title('2D dynamical system (' + str(idx) + '/' + str(T) + ')')
        plt.legend(loc="upper left")
        plt.pause(0.1)
        plt.savefig('final_fig.pdf')
        plt.clf()
    return None

show_sim()
