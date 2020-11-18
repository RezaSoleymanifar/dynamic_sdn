# %load testAlg.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from ECP_LL import ecp_ll

y_hist = []

#sample input: 'data.npy', 10, 0.05, 3, 'lb'
#data, max_m, gamma, interactive, mode = sys.argv[1:]

def find_opt(gamma = 0.06):
    file = np.load('time_series.npy', allow_pickle = True)
    t_series = file[0]
    labels = file[1]
    clusters = len(np.unique(labels))
    T = len(t_series)
    ctr = 0

    max_m, gamma, interactive = clusters, gamma, 2

    for dataset in t_series:
        ctr += 1
        print('{0}/{1}'.format(ctr, T))
        data = dataset
        obj = ecp_ll(data = data, max_m = int(max_m), gamma = float(gamma))
        y = obj.train()
        y_hist.append(y)
        np.save('y_hist', y_hist)
    return None
