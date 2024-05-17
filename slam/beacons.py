import numpy as np
import matplotlib.pyplot as plt
import yaml
import control as ct
import scipy
from qbstyles import mpl_style

# plt.style.use('dark_background')

mpl_style(True, False)


def load_beacon_positions(yaml_file_path: str):
    x, y = [], []

    with open(yaml_file_path, "r") as file:
        data_dict = yaml.load(file, yaml.SafeLoader)
        for pose in data_dict["poses"]:
            x.append(pose["position"]["x"])
            y.append(pose["position"]["y"])

    return np.vstack((x, y))


def plot_path(path, ax: plt.Axes = None, *symbol, **kwargs):
    x, y = path
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(x, y, *symbol, **kwargs)
    ax.axis("equal")
    ax.legend()


def simple_model(t, x, u, params={}):
    noisy = params.get("noisy", False)

    v, w = u
    x, y, theta = x

    if noisy:
        v = np.random.normal(v, params.get("std_v", 0.1))
        w = np.random.normal(w, params.get("std_w", 0.1))
    xdot = v * np.cos(theta)
    ydot = v * np.sin(theta)
    theta = w
    return [xdot, ydot, theta]


def interpolate_path(points, n=1000):
    x, y = points
    new_x = np.linspace(x[0], x[-1], n)
    new_y = np.interp(new_x, x, y)
    return np.vstack((new_x, new_y))


def simulate_model(initial_pose, speed=1.0, noisy=False, total_sim_time=120):
    model = ct.NonlinearIOSystem(
        simple_model,
        params={"noisy": noisy, "std_w": 0.1},
        states=["x", "y", "theta"],
        inputs=["v", "w"],
    )
    t = np.linspace(0, total_sim_time, 100)
    u = np.vstack((np.ones_like(t) * speed, np.zeros_like(t)))
    u[1][u.shape[1] // 2:] = -0.01
    sol = ct.input_output_response(model, t, u, initial_pose)

    x, y, theta = sol.y
    return np.vstack((x, y))


def roll_dice(p_true=0.2):
    if p_true > 1:
        raise AttributeError(f"p_true must be a value between 0 and 1. Got: {p_true}")
    return np.random.choice([True, False], p=[p_true, 1 - p_true])


def find_closest_beacon(xy, beacons):
    distances = np.linalg.norm(beacons.T - xy, axis=1)
    return np.argmin(distances)


def plot_detection_correspondence(xy1, xy2, ax: plt.Axes, mode=""):
    x1, y1 = xy1
    x2, y2 = xy2
    if mode == "est":
        ax.plot([x1, x2], [y1, y2], "--", color="#FFFFFFAA")
    elif mode == "match":
        ax.plot([x1, x2], [y1, y2], "-", linewidth=0.2, color="#FFFFFFBB")
    else:
        ax.plot([x1, x2], [y1, y2], color="#FFFFFFAA")


def pp_follow(t, x, y, params={}):
    lookahead_dist = params.get("lookahead_dist", 3.0)
    speed = params.get("speed", 3.0)
    path = params.get("path", None)
    if not path:
        raise "Path is empty"

    rx, ry, rtheta = x

    distances = np.linalg.norm(path.T - np.array([rx, ry]))
    closest_point_idx = np.argmin(distances)

    closest_point = path.T[closest_point_idx]

    trimmed_path = path.T[closest_point_idx:]
    trimmed_distances = distances[closest_point_idx]

    intersect_point_idx = np.where(trimmed_distances > lookahead_dist)[0]
    carrot_point = trimmed_path[intersect_point_idx]
    carrot_dist = trimmed_distances[intersect_point_idx]

    v = speed * np.array([np.cos(rtheta), np.sin(rtheta)])
    curv = (v - carrot_point) / carrot_dist

    # todo(ejalaa) finish this


# %%

odom_beacons = load_beacon_positions("slam/beacon_odom.yaml")
true_beacons = load_beacon_positions("slam/beacon_truth.yaml")
xy_real_car_path = simulate_model([-1, 0, 0.11], speed=1.2, noisy=True)
xy_dead_reckoning_path = simulate_model([1, -2, 0.11], speed=1.2)

path_to_follow = interpolate_path(odom_beacons)

# %%
fig, ax = plt.subplots()
plot_path(odom_beacons, ax, "+", label="beacons_odom")
plot_path(true_beacons, ax, "o", mfc="none", label="beacons")

plot_path(xy_real_car_path, ax, "-", label="real car path")
plot_path(xy_dead_reckoning_path, ax, "--", label="estimated car path")

corrections = np.zeros_like(xy_dead_reckoning_path)
xy_corrected_path = xy_dead_reckoning_path.copy()
corr = np.zeros(2)
n_corrections = 0
max_corrections = 100

for i, (xy, xy_dead_reckon, xy_corr) in enumerate(
        zip(xy_real_car_path.T, xy_dead_reckoning_path.T, xy_corrected_path.T)
):
    # random chance of detecting a beacon
    if not roll_dice(0.3):
        corrections.T[i] = corrections.T[i - 1]
        continue

    if n_corrections == max_corrections:
        continue

    # for simplicity always detect beacon closest to true pose
    idx_truth = find_closest_beacon(xy, true_beacons)
    plot_detection_correspondence(xy, true_beacons.T[idx_truth], ax)

    # Compute the pose relative to the beacon
    xy_relative_to_beacon = xy - true_beacons.T[idx_truth]

    # Identification method 1:
    #   Once a marker was detected, assume that it is necessary the one that is closest
    #   to the currently estimated position
    idx_closest = find_closest_beacon(xy_corr, odom_beacons)
    plot_detection_correspondence(xy_corr, odom_beacons.T[idx_closest], ax, "est")

    # plot matching points on true and corrected path
    plot_detection_correspondence(xy, xy_corr, ax, "match")

    # Correct the position based on the relative beacon pose
    prev_xy_corr = xy_corrected_path.T[i]
    xy_corr = odom_beacons.T[idx_closest] + xy_relative_to_beacon
    corr = xy_corr - prev_xy_corr
    # save the correction
    corrections.T[i] = corr
    # and apply it
    xy_corrected_path.T[i + 1:] += corr

    n_corrections += 1

plot_path(xy_corrected_path, ax, "*-", label="corrected path")
