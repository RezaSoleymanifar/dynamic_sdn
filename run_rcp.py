from RCP import rcp
from plot_sims import show_sim

alpha = 0.7
k_0 = 2e-4
rcp(seed = 2, spread = 0.3, k_0 = k_0, alpha= alpha, gamma = 0.1)
show_sim('K_0 = {0}'.format(k_0), 'slateblue', 'peru')
