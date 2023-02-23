"""
source: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter23.03-Finite-Difference-Method.html

"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from sklearn.metrics import mean_squared_error

N_TRUE = 1000


def generate_func(n, x_max, x_min=0):
    interval_width = (x_max - x_min) / n

    # Get stencil matrix
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = 1
        A[i, i] = -2
        A[i, i + 1] = 1

    t = np.linspace(x_min, x_max, n)

    # Get b (is equal to u'')
    b = np.zeros(n)
    for idx, val in enumerate(t):
        b[idx] = (-4 * pi ** 2 * sin(2 * pi * val)) / ((n) ** 2)

        # get y_true to compare and calculate error
    t_true = np.linspace(x_min, x_max, N_TRUE)
    y_true = np.zeros(N_TRUE)
    for idx, val in enumerate(t_true):
        y_true[idx] = sin(2 * pi * val)

    # solve the linear equations
    y = np.linalg.solve(A, b)

    # We now fit points between every approximated point (as shown in the final product) and its neighbors for the mse
    y_fitted = np.zeros(N_TRUE)
    for idx in range(len(t) - 1):
        section_size = int(np.floor(N_TRUE / (n - 1)))
        new_y = np.interp(t_true[section_size * idx:section_size * (idx + 1)], [t[idx], t[idx + 1]],
                          [y[idx], y[idx + 1]])
        for idx_2, val_y in enumerate(new_y):
            y_fitted[section_size * idx + idx_2] = val_y

    mse = mean_squared_error(y_true, y_fitted)

    plt.figure(figsize=(10, 8))
    plt.plot(t, y, label="Approximated Function")
    plt.plot(t_true, y_true, label="Reference Function")
    plt.xlabel('y')
    plt.ylabel('x')
    plt.legend(loc='lower left')
    plt.text(0.7, 0.9, f"MSE: {mse:.5f}")
    plt.title(f"sine function, evaluated at {n} points")
    plt.show()


generate_func(5, 1)
generate_func(10, 1)
generate_func(200, 1)
generate_func(500, 1)