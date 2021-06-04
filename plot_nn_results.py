#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

def plot_it(x_, x_dot, theta_, theta_dot, num):
    plt.clf()
    y = [i for i in range(len(x_))]
    plt.plot(y, x_dot, label='x_dot')
    plt.plot(y, theta_dot, label="theta_dot")
    plt.plot(y, x_, label="x")
    plt.plot(y, theta_, label="theta")
    plt.legend()
    plt.savefig('nn_cartpole_results_with_dots_run{:02d}'.format(num))

if __name__ == "__main__":
    results_file = "cart-pole-data-20210603-173326.npy"

    data = np.load(results_file)

    for i in range(len(data)):
        entry = data[i]
        x, xdot, theta, thetadot = np.hsplit(entry, 4)

        plot_it(x, xdot, theta, thetadot, i)
