"""
Point-and-Click model

Reference:
1) "A Simulation Model of Intermittently Controlled Point-and-Click Behavior"

This code was written by Seungwon Do (dodoseung)
"""

import numpy as np

import modules.click_action_module as click
import modules.motor_control_module as motor
import modules.visual_perception_module as visual
import modules.mouse_module as mouse
import modules.upper_limb_module as limb


def model(state_true, state_cog, para_bump, para_icp, para_env):

    # States
    cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y = state_cog
    c_true_pos_x, c_true_pos_y, target_pos_x, target_pos_y, target_vel_x, target_vel_y, target_radius, h_pos_x, h_pos_y = state_true

    # Parameters for the models
    Th, Tp, nc, fixed = para_bump
    click_decision, c_mu, c_sigma, _nu, _delta, _p = para_icp

    # Parameters for the environment
    interval, boundary_width, boundary_height, forearm = para_env

    # Default click timing
    time_total = 99

    target_true = [target_pos_x, target_pos_y, target_vel_x, target_vel_y]

    # Target information at SA
    target_pos_x, target_vel_x = motor.boundary(Tp, target_pos_x, -target_vel_x, interval, boundary_width, target_radius)
    target_pos_y, target_vel_y = motor.boundary(Tp, target_pos_y, -target_vel_y, interval, boundary_height, target_radius)
    target_vel_x *= -1
    target_vel_y *= -1

    # Target information with the visual noise
    target_vel_x, target_vel_y = visual.visual_speed_noise(target_vel_x, target_vel_y)

    # Target information at RP
    target_pos_x, target_vel_x = motor.boundary(Tp, target_pos_x, target_vel_x, interval, boundary_width, target_radius)
    target_pos_y, target_vel_y = motor.boundary(Tp, target_pos_y, target_vel_y, interval, boundary_height, target_radius)
    target_info = [target_pos_x, target_pos_y, target_vel_x, target_vel_y]

    # Predicted target information after Th
    target_pos_x, target_vel_x = motor.boundary(Th, target_pos_x, target_vel_x, interval, boundary_width, target_radius)
    target_pos_y, target_vel_y = motor.boundary(Th, target_pos_y, target_vel_y, interval, boundary_height, target_radius)

    # Set the prediction horizon
    pred_horizon = Th if click_decision else Tp

    # Generating the optimal trajectory
    otg_ideal_x = motor.otg(cursor_pos_x, cursor_vel_x, target_pos_x, target_vel_x, Th, interval, pred_horizon)
    otg_ideal_y = motor.otg(cursor_pos_y, cursor_vel_y, target_pos_y, target_vel_y, Th, interval, pred_horizon)

    # Correct the trajectory which is out of the bound
    for idx in range(otg_ideal_x.shape[1]):
        if otg_ideal_x[0, idx] <= 0:
            otg_ideal_x[0, idx] = np.finfo(float).tiny
            otg_ideal_x[1, idx] = 0
        elif otg_ideal_x[0, idx] >= boundary_width:
            otg_ideal_x[0, idx] = boundary_width
            otg_ideal_x[1, idx] = 0

        if otg_ideal_y[0, idx] <= 0:
            otg_ideal_y[0, idx] = np.finfo(float).tiny
            otg_ideal_y[1, idx] = 0
        elif otg_ideal_y[0, idx] >= boundary_height:
            otg_ideal_y[0, idx] = boundary_height
            otg_ideal_y[1, idx] = 0

    # Get the position, velocity, and acceleration information of the cursor
    c_vel_x = otg_ideal_x[1, :pred_horizon + 1].copy()
    c_vel_y = otg_ideal_y[1, :pred_horizon + 1].copy()

    # Set the mouse noise
    mouse_gain = mouse.gain_func_can((c_vel_x ** 2 + c_vel_y ** 2) ** (1 / 2))
    for idx in range(pred_horizon + 1):
        if mouse_gain[idx] == 0: mouse_gain[idx] = np.finfo(float).tiny

    # Get the acceleration
    acc_x = ((c_vel_x / mouse_gain)[1:] - (c_vel_x / mouse_gain)[:-1]) / interval
    acc_y = ((c_vel_y / mouse_gain)[1:] - (c_vel_y / mouse_gain)[:-1]) / interval

    # Get the ideal hand pos
    _, _, h_pos_x_ideal, h_pos_y_ideal = limb.mouse_noise(c_vel_x / mouse_gain, c_vel_y / mouse_gain, h_pos_x, h_pos_y, forearm, mouse_gain, interval)
    h_pos_delta_x = h_pos_x_ideal - h_pos_x
    h_pos_delta_y = h_pos_y_ideal - h_pos_y

    # Set the motor noise
    _, _, c_vel_x, c_vel_y = limb.motor_noise(c_vel_x / mouse_gain, c_vel_y / mouse_gain, pred_horizon, 0, 0, nc, interval)

    # Set the mouse noise
    pos_dx_mouse, pos_dy_mouse, h_pos_x, h_pos_y = limb.mouse_noise(c_vel_x.copy(), c_vel_y.copy(), h_pos_x, h_pos_y, forearm, mouse_gain, interval)
    c_pos_dx = ((c_vel_x * mouse_gain)[1:] + (c_vel_x * mouse_gain)[:-1]) / 2 * interval
    c_pos_dy = ((c_vel_y * mouse_gain)[1:] + (c_vel_y * mouse_gain)[:-1]) / 2 * interval
    vel_mouse_noise_x = (sum(pos_dx_mouse) - sum(c_pos_dx)) / (pred_horizon * interval)
    vel_mouse_noise_y = (sum(pos_dy_mouse) - sum(c_pos_dy)) / (pred_horizon * interval)
    c_pos_dx, c_pos_dy, c_vel_x, c_vel_y = limb.motor_noise(c_vel_x * mouse_gain, c_vel_y * mouse_gain, pred_horizon, vel_mouse_noise_x, vel_mouse_noise_y, [0, 0], interval)

    # Ideal last executed cursor position and velocity for next BUMP
    c_pos_delta_x = otg_ideal_x[0, pred_horizon] - otg_ideal_x[0, 0]
    c_pos_delta_y = otg_ideal_y[0, pred_horizon] - otg_ideal_y[0, 0]

    # Ideal cursor velocity
    c_vel_ideal_x = otg_ideal_x[1, pred_horizon]
    c_vel_ideal_y = otg_ideal_y[1, pred_horizon]

    # For Click Timing
    # Get the click timing
    if click_decision:
        target_next_pos_x, _ = motor.boundary(Th, target_true[0], target_true[2], interval, boundary_width, target_radius)
        target_next_pos_y, _ = motor.boundary(Th, target_true[1], target_true[3], interval, boundary_height, target_radius)

        tc = Th * interval
        cursor_pos = np.array([c_true_pos_x, c_true_pos_y])
        cursor_vel = np.array([np.sum(c_pos_dx) / tc, np.sum(c_pos_dy) / tc])
        target_pos = np.array([target_true[0], target_true[1]])
        target_vel = np.array([(target_next_pos_x - target_true[0]) / tc, (target_next_pos_y - target_true[1]) / tc])

        click_timing = click.model(cursor_pos, cursor_vel, target_pos, target_vel, target_radius, tc, c_mu, c_sigma, _nu, _delta, _p)
        time_total = click.total_click_time(click_timing, tc, cursor_pos, cursor_vel, target_pos, target_vel, target_radius, c_mu, c_sigma, _p, _nu, _delta)

    # Acceleration
    acc_sum = sum((acc_x**2 + acc_y**2) ** (1/2))

    # Correct the trajectory which is out of the bound
    for idx in range(len(c_pos_dx)):
        if c_true_pos_x + sum(c_pos_dx[:idx + 1]) <= 0:
            c_pos_dx[idx] = -(c_true_pos_x + sum(c_pos_dx[:idx]))
            c_vel_x[idx + 1] = 0
        elif c_true_pos_x + sum(c_pos_dx[:idx + 1]) >= boundary_width:
            c_pos_dx[idx] = boundary_width - (c_true_pos_x + sum(c_pos_dx[:idx]))
            c_vel_x[idx + 1] = 0

        if c_true_pos_y + sum(c_pos_dy[:idx + 1]) <= 0:
            c_pos_dy[idx] = -(c_true_pos_y + sum(c_pos_dy[:idx]))
            c_vel_y[idx + 1] = 0
        elif c_true_pos_y + sum(c_pos_dy[:idx + 1]) >= boundary_height:
            c_pos_dy[idx] = boundary_height - (c_true_pos_y + sum(c_pos_dy[:idx]))
            c_vel_y[idx + 1] = 0

    # Summary of the outputs
    cursor_info = c_pos_dx, c_pos_dy, c_vel_x, c_vel_y, c_pos_delta_x, c_pos_delta_y, c_vel_ideal_x, c_vel_ideal_y
    target_info = target_info
    hand_info = h_pos_x, h_pos_y, h_pos_delta_x, h_pos_delta_y
    click_info = time_total
    effort_info = acc_sum

    return cursor_info, target_info, hand_info, click_info, effort_info