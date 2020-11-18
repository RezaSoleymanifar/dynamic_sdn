# %load testAlg.py
import numpy as np
import os
from ECP_LL import ecp_ll


#sample input: 'data.npy', 10, 0.05, 3, 'lb'
#data, max_m, gamma, interactive, mode = sys.argv[1:]

def find_opt(gamma = 0.06, t_min = 1e-4, t_reset = 1, alpha = 0.9):
    file = np.load('time_series.npy', allow_pickle = True)
    t_series = file[0]
    labels = file[1]
    clusters = len(np.unique(labels))
    T = len(t_series)
    ctr = 0
    y_hist = []

    #Removes previous y_hist.npy
    try:
        os.remove('y_hist.npy')
    except:
        pass

    for dataset in t_series:
        ctr += 1
        print('{0}/{1}'.format(ctr, T))
        data = dataset

        try:
            temp = np.load('y_hist.npy')
            y_0 = list(temp[-1, :, :])
            t_start = t_reset
        except:
            y_0 = None
            t_start = 2

        y = ecp_ll(data = data, max_m = int(clusters),
                   gamma = float(gamma), y_0 = y_0,
                   t_min = t_min, t_start = t_start, alpha = alpha)
        y_hist.append(y)
        np.save('y_hist', y_hist)
    return None