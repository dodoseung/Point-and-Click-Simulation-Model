"""
Visual Perception Module

1) Perceiving cursor position and velocity
2) Perceiving target position and velocity

Reference:
1) "Noise characteristics and prior expectations in human visual speed perception"
https://www.nature.com/articles/nn1669

This code was written by Seungwon Do (dodoseung)
"""

import math
import numpy as np

### Perceiving target position and velocity
dist_head_to_monitor = 0.63
sample = 1

def visual_speed_noise(vel_x, vel_y):
    vel = (vel_x ** 2 + vel_y ** 2) ** 0.5
    if vel <= 0: vel = np.finfo(float).tiny

    v_set = []
    for i in range(sample):      
        v = 2 * math.degrees(math.atan(vel / (2 * dist_head_to_monitor)))
        v_0 = 0.3
        v_hat = math.log(1 + v / v_0)
        sigma = 0.15 * 1
        v_prime = np.random.lognormal(v_hat, sigma)
        v_final = (v_prime - 1) * v_0
        v_final = dist_head_to_monitor * 2 * math.tan(math.radians(v_final) / 2)
        if v_final < 0: v_final = 0
        v_set.append(v_final)

    v_final = sum(v_set) / sample
    ratio_x = vel_x / vel
    ratio_y = vel_y / vel
    vel_x = v_final * ratio_x
    vel_y = v_final * ratio_y

    return vel_x, vel_y