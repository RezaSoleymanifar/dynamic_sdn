import numpy as np
from ECP_LL import plt
plt.rcParams.update({'font.size': 12})


def plot_nodes(data, color, alpha, edgecolors, zorder, label):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color=color, alpha=alpha,
                edgecolors=edgecolors, zorder=2, label=label)


def plot_paths(hist, label, color='orange', alpha=0.6, edge=None, ):
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
            plt.plot(x, y, marker='.', c=color, markeredgecolor=edge, alpha=alpha, label=label)
        else:
            plt.plot(x, y, marker='.', c=color, markeredgecolor=edge, alpha=alpha)
        ctr += 1

def show_sim(title = '2D dynamical system', col_opt = 'darkorchid', col_rcp = 'orange'):
    y = np.load('y_hist.npy', allow_pickle = True)
    time_series = np.load('time_series.npy', allow_pickle = True)
    rcp_hist = np.array(np.load('rcp_hist.npy', allow_pickle= True))

    t_series, labels = time_series[0], time_series[1]
    T = len(y)

    data_start = t_series[0]
    data_final = t_series[-1]

    while(True):
        for idx in range(T-1):

            data = t_series[idx]

            plot_paths(rcp_hist[:idx+1, :, :], edge = 'grey', color=col_rcp, label = 'RCP')

            plot_paths(y[:idx + 1, :, :], color=col_opt, label = 'Optimal')

            plot_nodes(data, color='turquoise', alpha = 0.25,
                       edgecolors='black', zorder = 2, label='network nodes')

            plot_nodes(data_start, color='cyan', alpha = 0.04,
                   edgecolors='purple', zorder = 1, label='start')

            plot_nodes(data_final, color='purple', alpha = 0.04,
                   edgecolors='cyan', zorder = 1, label='end')

            plt.minorticks_on()
            plt.grid(b=True, which='major', color='black', linestyle='--', alpha = 0.15)
            plt.grid(b=True, which='minor', color='black', linestyle='-', alpha = 0.01)
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            # plt.title('2D dynamical system (' + str(idx) + '/' + str(T) + ')')
            plt.title(title)
            plt.legend(loc="upper left")
            plt.pause(0.1)
            plt.savefig('sim.pdf')
            plt.clf()

    return None
