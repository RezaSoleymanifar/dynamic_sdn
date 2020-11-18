from ECP_LL import ecp_ll
import numpy as np

file = np.load('time_series.npy', allow_pickle=True)
t_series = file[0]
data = t_series[0]
max_m, gamma = 3, 2
y = ecp_ll(data=data, gamma=gamma, max_m = max_m, show_plot= True)
