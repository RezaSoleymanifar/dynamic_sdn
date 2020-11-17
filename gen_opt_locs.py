# %load testAlg.py
import numpy as np
import matplotlib.pyplot as plt
import sys
from DALL import DA

y_hist = []

#sample input: 'data.npy', 10, 0.05, 3, 'lb'
#data, max_m, gamma, interactive, mode = sys.argv[1:]

def find_opt():
    file = np.load('time_series.npy', allow_pickle = True)
    t_series = file[0]
    labels = file[1]
    clusters = len(np.unique(labels))
    T = len(t_series)
    ctr = 0

    max_m, gamma, interactive = clusters, 0.06, 2

    for dataset in t_series:
        ctr += 1
        print('{0}/{1}'.format(ctr, T))
        data = dataset
        obj = DA(data = data, max_m = int(max_m), gamma = float(gamma), interactive = int(interactive))
        indices = obj.train()
        y_hist.append(indices)
        np.save('y_hist', y_hist)
    return None

##data, max_m, gamma, interactive = t_series[4], clusters, 0.06, 2
##obj = DA(data = data, max_m = max_m, gamma = float(gamma), interactive = interactive)   
##indices = obj.train()
