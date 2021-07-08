"""
Click Action Module
Intermittent Click Planing (ICP) model

Reference:
1) "An Intermittent Click Planning Model", Eunji Park and Byungjoo Lee
https://dl.acm.org/doi/abs/10.1145/3313831.3376725

This code was written by Seungwon Do (dodoseung)
"""
import numpy as np


# The length of the line where the movement trajectory of the pointer intersects with the target. A, B, C, and radius.
def w_intersect(cursor_pos, cursor_vel, target_pos, target_vel, target_radius):
    # Calculate the distance between the trajectory line of the cursor and the center of the target
    # result = root((target_radius)^2 - (the distance calculated above)^2)
    # return the 2 * result

    cursor_pos_end = cursor_pos + (cursor_vel - target_vel)

    nominator = np.linalg.norm(np.cross(cursor_pos_end - cursor_pos, cursor_pos - target_pos))
    denominator = np.linalg.norm(cursor_pos_end - cursor_pos)
    center_to_line = nominator / denominator
    
    dist_cursor_target = np.linalg.norm(target_pos - cursor_pos)
    if dist_cursor_target <= target_radius:
        value = np.sqrt(target_radius ** 2 - center_to_line ** 2)
        bot = np.sqrt(dist_cursor_target**2 - center_to_line**2)
        vec1 = cursor_pos_end - cursor_pos
        vec2 = cursor_pos - target_pos
        if vec1.dot(vec2) > 0:
            return value - bot
        else:
            return value + bot

    if center_to_line <= target_radius:
        value = np.sqrt(target_radius**2 - center_to_line**2)
        w_intersect_value = 2 * value
    else:
        w_intersect_value = 0

    return w_intersect_value


def w_t(w_intersect_value, cursor_vel, target_vel):
    vel_relative = cursor_vel - np.array(target_vel)
    wt = w_intersect_value / np.linalg.norm(vel_relative)

    return wt


def mean(c_mu, wt):
    click_timing_mean = c_mu * wt

    return click_timing_mean


def variance(c_sigma, _p, _nu, tc, _delta):
    nominator = (c_sigma ** 2) * (_p ** 2)
    denominator = (1 + (_p / (1 / (np.exp(_nu * tc) - 1) + _delta)) ** 2)
    click_timing_variance = np.sqrt(nominator / denominator)

    return click_timing_variance


# The click timing
def model(cursor_pos_init, cursor_vel, target_pos, target_vel, target_radius, tc, c_mu, c_sigma, _nu, _delta, _p):
    w_intersect_value = w_intersect(cursor_pos_init, cursor_vel, target_pos, target_vel, target_radius)
    w_t_value = w_t(w_intersect_value, cursor_vel, target_vel)
    tc -= (w_t_value / 2)
    click_timing_mean = mean(c_mu, w_t_value)
    click_timing_variance = variance(c_sigma, _p, _nu, tc, _delta)
    click_timing = np.random.normal(click_timing_mean, click_timing_variance, 1)

    return click_timing[0]


def index_of_difficulty(cursor_pos_init, cursor_vel, target_pos, target_vel,
                        target_radius, tc, _nu, _delta, _p):
    w_intersect_value = w_intersect(cursor_pos_init, cursor_vel, target_pos, target_vel, target_radius)
    w_t_value = w_t(w_intersect_value, cursor_vel, target_vel)

    tc -= (w_t_value / 2)
    nominator = _p
    denominator = np.sqrt(1 + (_p / (1 / (np.exp(_nu * tc) - 1) + _delta)) ** 2)
    d_t_value = nominator / denominator

    if w_t_value == 0:
        w_t_value = np.finfo(float).tiny

    idx_of_diff = np.log2(d_t_value / w_t_value)

    return idx_of_diff


def total_click_time(click_timing, tc, cursor_pos, cursor_vel, target_pos, target_vel, target_radius, c_mu, c_sigma, _p, _nu, _delta):
    w_intersect_value = w_intersect(cursor_pos, cursor_vel, target_pos, target_vel, target_radius)
    w_t_value = w_t(w_intersect_value, cursor_vel, target_vel)

    time_enter = tc - (w_t_value / 2)
    time_total = time_enter + click_timing

    if time_enter < 0 and click_timing >= tc:
        time_total = tc - 0.00001
    elif time_total < 0:
        time_total = 0
    elif w_t_value == 0:
        time_total = tc - 0.00001
    elif time_total > tc:
        time_total = tc - 0.00001

    return time_total
