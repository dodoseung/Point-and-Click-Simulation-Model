"""
Motor Control Module
Basic Unit of Motor Production (BUMP) model

Reference:
1) "The BUMP model of response planning: Variable horizon predictive control accounts for the speed–accuracy tradeoffs and velocity profiles of aimed movement", Robin T. Bye and Peter D. Neilson
https://www.sciencedirect.com/science/article/pii/S0167945708000377

This code was written by Seungwon Do (dodoseung)
"""
import numpy as np
import math
import csv
import os

# Optimal trajectory generation
# Pos0 and Vel0 are the position and velocity of initial state.
# PosN and VelN are the position and velocity of final state.
def otg(pos_0, vel_0, pos_n, vel_n, th, _interval, num):
    # Generate the OTG (Position and velocity)
    n = th
    x_0 = np.mat([[pos_0], [vel_0]])
    x_n = np.mat([[pos_n], [vel_n]])
    g = np.mat([[1, _interval], [0, 1]])
    h = np.mat([[_interval * _interval / 2], [_interval]])

    g_n = g ** n
    inv_gram_n = np.linalg.inv(grammian(n, n, g, h))
    x = np.zeros((2, num + 1))
    for k in range(0, num + 1):
        g_k = g ** k
        gram_k = grammian(k, n, g, h)
        x[:, [k]] = (g_k * x_0) + gram_k * inv_gram_n * (x_n - g_n * x_0)

    return x


# 2 by 2 discrete time Grammian matrix
def grammian(k, n, g, h):
    h_t = h.transpose()
    g_t = g.transpose()
    mat = np.zeros((2, 2))
    for j in range(0, k):
        g_j = g ** j
        g_t_j = g_t ** (n-k+j)
        mat = mat + g_j * h * h_t * g_t_j
    return mat


def fixedhorizon(_th, _tp):
    _th -= _tp

    if _th < _tp:
        _th = _tp

    return _th


def boundary(size, pos, vel, interval, bound, radius):
    if size == 0:
        return pos, vel

    upper_bound = bound - radius
    lower_bound = radius
    for i in range(size):
        next_pos = pos + (interval * vel)
        if upper_bound >= next_pos >= lower_bound:
            pos = next_pos
        elif next_pos > upper_bound:
            pos = (2 * upper_bound) - next_pos
            vel *= -1
        elif next_pos < lower_bound:
            pos = (2 * lower_bound) - next_pos
            vel *= -1

    return pos, vel
