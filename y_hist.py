# %load testAlg.py
import numpy as np
import matplotlib.pyplot as plt

y_hist = np.load('y_hist.npy', allow_pickle = True)
T = len(y_hist)

##for ctrds in y_hist:
for idx in range(T):
    ctrds = y_hist[idx]
    c_x = ctrds[:, 0]
    c_y = ctrds[:, 1]

    plt.scatter(c_x, c_y, c= 'b')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='black', linestyle='--', alpha = 0.15)
    plt.grid(b=True, which='minor', color='black', linestyle='-', alpha = 0.01)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('2D dynamical system (' + str(idx) + '/' + str(T) + ')')
    plt.pause(0.5)

