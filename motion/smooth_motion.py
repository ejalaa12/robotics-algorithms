import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from functools import cache

np.set_printoptions(
    precision=3,
)


@cache
def quint_poly(t0, tf, s0, sf, sd0=0, sdf=0, sdd0=0, sddf=0):
    # matrix of bounding conditions
    bound_mat = np.array(
        [
            [t0**5, t0**4, t0**3, t0**2, t0, 1],
            [tf**5, tf**4, tf**3, tf**2, tf, 1],
            [5 * t0**4, 4 * t0**3, 3 * t0**2, 2 * t0, 1, 0],
            [5 * tf**4, 4 * tf**3, 3 * tf**2, 2 * tf, 1, 0],
            [20 * t0**3, 12 * t0**2, 6 * t0, 2, 0, 0],
            [20 * tf**3, 12 * tf**2, 6 * tf, 2, 0, 0],
        ]
    )
    if np.linalg.det(bound_mat) < 1e-3:
        raise Exception(
            f"Singular Matrix:", t0, tf, s0, sf, sd0, sdf, sdd0, sddf, "\n{bound_mat}"
        )
    bound = np.array([s0, sf, sd0, sdf, sdd0, sddf])
    coeffs = np.linalg.solve(bound_mat, bound)

    return np.polynomial.Polynomial(np.flip(coeffs))


def sixt_poly(t0, ti, tf, s0, si, sf, sd0=0, sdf=0, sdd0=0, sddf=0):
    bound_mat = np.array(
        [
            [t0**6, t0**5, t0**4, t0**3, t0**2, t0, 1],
            [ti**6, ti**5, ti**4, ti**3, ti**2, ti, 1],
            [tf**6, tf**5, tf**4, tf**3, tf**2, tf, 1],
            [6 * t0**5, 5 * t0**4, 4 * t0**3, 3 * t0**2, 2 * t0, 1, 0],
            [6 * tf**5, 5 * tf**4, 4 * tf**3, 3 * tf**2, 2 * tf, 1, 0],
            [30 * t0**4, 20 * t0**3, 12 * t0**2, 6 * t0, 2, 0, 0],
            [30 * tf**4, 20 * tf**3, 12 * tf**2, 6 * tf, 2, 0, 0],
        ]
    )
    bound = np.array([s0, si, sf, sd0, sdf, sdd0, sddf])
    coeffs = np.linalg.solve(bound_mat, bound)

    return np.polynomial.Polynomial(np.flip(coeffs))


quint_poly_v = np.vectorize(quint_poly)


def test_interp_quint_poly(t0, tf, s0, sf, sd0=0, sdf=0, sdd0=0, sddf=0):
    p = quint_poly(t0, tf, s0, sf, sd0, sdf, sdd0, sddf)

    t = np.linspace(t0, tf, 1000)
    s = p(t)
    sd = p.deriv()(t)
    sdd = p.deriv().deriv()(t)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(t, s)
    ax1.grid()

    ax2.plot(t, sd)
    ax2.grid()

    ax3.plot(t, sdd)
    ax3.grid()


def test_interp_sixt_poly(t0, ti, tf, s0, si, sf, sd0=0, sdf=0, sdd0=0, sddf=0):
    p = sixt_poly(t0, ti, tf, s0, si, sf, sd0, sdf, sdd0, sddf)

    t = np.linspace(t0, tf, 1000)
    s = p(t)
    sd = p.deriv()(t)
    sdd = p.deriv().deriv()(t)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(t, s)
    ax1.grid()

    ax2.plot(t, sd)
    ax2.grid()

    ax3.plot(t, sdd)
    ax3.grid()


def show_points(points):
    fig, ax = plt.subplots()
    ax.plot(via[:, 0], via[:, 1])


def linear_func(a, b, t):
    return (1 - t) * a + b * t


def slerp_func(t0, tf, s0, sf):
    return np.polynomial.Polynomial(
        [s0 - t0 * (sf - s0) / (tf - t0), (sf - s0) / (tf - t0)]
    )


# %%
def save_and_increment_time(inc_time, time_list=None):
    if time_list is None:
        time_list = []

    prev_time = time_list[-1]
    new_time = prev_time + inc_time
    time_list.append(new_time)

    return time_list


def create_segments(via):
    return np.diff(via, axis=0)


class Trajectory:
    def __init__(self, times, poses, speeds=None, accs=None):
        if isinstance(times, list):
            times = np.array(times)
        if isinstance(poses, list):
            poses = np.array(poses)
        if isinstance(speeds, list):
            speeds = np.array(speeds)
        if isinstance(accs, list):
            accs = np.array(accs)

        if speeds is None:
            speeds = np.zeros_like(poses)

        if accs is None:
            accs = np.zeros_like(poses)

        if not (times.size == poses.shape[0] == speeds.shape[0] == accs.shape[0]):
            raise AttributeError(
                "times, poses and speeds should have the number of elements"
            )

        self.times = times
        self.n = self.times.size

        self.poses = poses.reshape(self.n, -1)
        self.speeds = speeds.reshape(self.n, -1)
        self.accs = accs.reshape(self.n, -1)
        self.dim = self.poses.shape[1]
        self.polys = []

    def plot(self, *args, **kwargs):
        ax = kwargs.get("ax", None)
        for i in range(self.dim):
            if ax is None:
                plt.plot(self.times, self.poses[:, i], *args, label=f"x_{i}")
            else:
                ax.plot(self.times, self.poses[:, i], *args, label=f"x_{i}")
        plt.legend()

    def plot_speed(self, *args, **kwargs):
        ax = kwargs.get("ax", None)
        for i in range(self.dim):
            if ax is None:
                plt.plot(self.times, self.speeds[:, i], *args, label=f"v_{i}")
            else:
                ax.plot(self.times, self.speeds[:, i], *args, label=f"v_{i}")
        plt.legend()


def show_grid(dx, xmax, dy, ymax):
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.xticks(np.arange(0, xmax, dx))
    plt.yticks(np.arange(0, ymax, dy))


def show_grid_on_points(xpoints, ypoints, ax=None):
    if ax is None:
        artist = plt
    else:
        artist = ax
    artist.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.xticks(xpoints)
    plt.yticks(ypoints)


# %%


def compute_timing(
    via, vmax, tacc, t0=0, v0=None, verbose=False, v_nominal=None, max_acc=None
):
    if isinstance(vmax, float) or isinstance(vmax, int):
        vmax = np.array([vmax]).reshape(-1, 1)
    if v0 is None:
        v0 = np.zeros_like(vmax)
    if v_nominal is None:
        # if vnom is not defined, make sure it is greater than norm of vmax
        v_nominal = 2 * np.linalg.norm(vmax)
    if max_acc is None:
        max_acc = np.ones_like(vmax) * 3

    # compute segments
    segments = np.diff(via, axis=0)
    # compute time per segments
    # compute the time of the slowest axis
    t_slowest_axis = np.max(np.abs(segments / vmax), axis=1)
    # Compute the time it would take if we move at v_nominal
    t_nominal = np.linalg.norm(segments, axis=1) / v_nominal
    # Time per segment should either be the max between the two previous one
    t_segments = np.max((t_slowest_axis, t_nominal), axis=0)
    # deduce speed per segments
    speed_segments = np.divide(segments.T, t_segments.T).T
    # compute blend duration based on max acceleration
    blend_durations = np.max(
        np.abs(np.diff(np.vstack([v0, speed_segments]), axis=0) / max_acc), axis=1
    )

    # Compute transition points
    blending_times = [t0]
    blending_points = [via[0]]
    blending_speeds = [v0]
    polys = [[] for _ in range(via.shape[1])]

    # save via points time as well
    via_points_times = np.concatenate([[0], np.cumsum(t_segments)])

    blend_start_time = t0
    blend_start_point = via[0]
    blend_start_speed = v0

    minimal_blend_duration = 0.1
    # Iterate through pre-calculated start, segment, time and speed per segments
    for i, (start, seg, t_seg, speed_seg, blend_duration) in enumerate(
        zip(via[:-1, :], segments, t_segments, speed_segments, blend_durations)
    ):
        tacc_for_seg = tacc
        tacc_for_seg = blend_duration
        tacc_for_n_seg = blend_durations[i+1]
        # if blend_duration < 1e-3:
        #     blend_duration = minimal_blend_duration
        if blend_duration > t_seg:
            print('ahem')

        start_time = via_points_times[i]
        end = via[i + 1, :]
        end_time = via_points_times[i + 1]

        # end of blend
        blend_end_time = start_time + tacc_for_seg / 2
        blend_end_point = start + speed_seg * tacc_for_seg / 2
        blend_end_speed = speed_seg

        blending_times.append(blend_end_time)
        blending_points.append(blend_end_point)
        blending_speeds.append(blend_end_speed)

        # start of next blend
        next_blend_time = end_time - tacc_for_n_seg / 2
        dt = next_blend_time - start_time
        next_blend_point = start + speed_seg * dt
        next_blend_speed = speed_seg

        print(f"blending from {blend_start_time} -> {blend_end_time} (={blend_duration})")
        print(f"\t{blend_start_point} ---> {blend_end_point}")
        print(f"and then linear from {blend_end_time} -> {next_blend_time}")
        print(f"\t{blend_end_point} ---> {next_blend_point}")

        for axis in range(via.shape[1]):
            polys[axis].append(
                quint_poly(
                    blend_start_time,
                    blend_end_time,
                    blend_start_point[axis],
                    blend_end_point[axis],
                    blend_start_speed[axis],
                    blend_end_speed[axis],
                )
            )
            polys[axis].append(
                slerp_func(
                    blend_end_time,
                    next_blend_time,
                    blend_end_point[axis],
                    next_blend_point[axis],
                )
            )

        blend_start_time = next_blend_time
        blend_start_point = next_blend_point
        blend_start_speed = next_blend_speed

        if verbose:
            print(
                f"# {blending_times[-1]:.1f} .. {start} --> {end} : {seg} \t || {t_seg}s @ {speed_seg}"
            )
            print(f"\t blend_end_time = t_prev + {tacc_for_seg / 2} = {blend_end_time}")
            print(
                f"\t blend_end_point = {start} + {speed_seg * tacc_for_seg / 2} = {blend_end_point}"
            )
            print("\t .")
            print(f"\t duration_until_next_blend = {t_seg - tacc_for_seg / 2=}")
            print(
                f"\t next_blend_point = {start + speed_seg * dt=} = {blend_end_point}"
            )
            print()

        blending_times.append(next_blend_time)
        blending_points.append(next_blend_point)
        blending_speeds.append(next_blend_speed)

    # TODO: add last blend to finish
    blending_times.append(via_points_times[-1])
    blending_points.append(via[-1])
    blending_speeds.append(v0)
    for axis in range(via.shape[1]):
        polys[axis].append(
            quint_poly(
                blend_start_time,
                via_points_times[-1],
                blend_start_point[axis],
                via[-1][axis],
                blend_start_speed[axis],
                0,
            )
        )
    blending_traj = Trajectory(blending_times, blending_points, blending_speeds)
    blending_traj.polys = polys
    via_traj = Trajectory(via_points_times, via)
    return via_traj, blending_traj

def compute_timing3(via, vmax, tacc, t0=0, v0=None, verbose=False, v_nominal=None, max_acc=None):
    if isinstance(vmax, float) or isinstance(vmax, int):
        vmax = np.array([vmax]).reshape(-1, 1)
    if v0 is None:
        v0 = np.zeros_like(vmax)
    if v_nominal is None:
        # if vnom is not defined, make sure it is greater than norm of vmax
        v_nominal = 2 * np.linalg.norm(vmax)
    if max_acc is None:
        max_acc = np.ones_like(vmax) * 3

    # compute segments
    segments = np.diff(via, axis=0)

    time = t0

    for i in range(segments.shape[0]):
        seg = segments[i]

        t_slowest_axis = np.max(np.abs(seg / vmax))
        t_nominal = np.linalg.norm(seg) / v_nominal
        t_seg = np.max((t_slowest_axis, t_nominal))

        max_sync_speed = seg / t_seg
def compute_timing2(via, vmax, tacc, t0=0, v0=None, verbose=False, v_nominal=None):
    if isinstance(vmax, float) or isinstance(vmax, int):
        vmax = np.array([vmax]).reshape(-1, 1)
    if v0 is None:
        v0 = np.zeros_like(vmax)
    if v_nominal is None:
        # if vnom is not defined, make sure it is greater than norm of vmax
        v_nominal = 2 * np.linalg.norm(vmax)

    # compute segments
    segments = np.diff(via, axis=0)
    # compute time per segments
    # compute the time of the slowest axis
    t_slowest_axis = np.max(np.abs(segments / vmax), axis=1)
    # Compute the time it would take if we move at v_nominal
    t_nominal = np.linalg.norm(segments, axis=1) / v_nominal
    # Time per segment should either be the max between the two previous one
    t_segments = np.max((t_slowest_axis, t_nominal), axis=0)
    # deduce speed per segments
    speed_segments = np.divide(segments.T, t_segments.T).T

    # save via points time as well
    via_points_times = np.concatenate([[0], np.cumsum(t_segments)])

    # Compute transition points
    blending_times = [t0]
    blending_points = [via[0]]
    blending_speeds = [v0]

    blending_times1 = via_points_times[:-1] + tacc / 2
    blending_points1 = via[:-1, :] + speed_segments * tacc / 2
    blending_speeds1 = speed_segments

    blending_times2 = via_points_times[1:] - tacc / 2
    blending_points2 = via[:-1, :] + speed_segments * (
        blending_times2 - via_points_times[:-1]
    )
    blending_speeds2 = speed_segments

    times = np.concatenate((blending_times1, blending_times2))
    blending_times = times.sort()
    blending_points = np.concatenate((blending_points1, blending_points2))[
        np.argsort(times)
    ]
    blending_speeds = np.concatenate((blending_speeds1, blending_speeds2))[
        np.argsort(times)
    ]

    # # TODO: add start and last blend to finish
    # blending_times.append(via_points_times[-1])
    # blending_points.append(via[-1])
    # blending_speeds.append(v0)
    blending_traj = Trajectory(blending_times, blending_points, blending_speeds)
    via_traj = Trajectory(via_points_times, via)
    return via_traj, blending_traj


def compute_command(time, via_traj, blend_traj, vmax, verbose=False):
    dim = via_traj.dim

    for i in range(blend_traj.n - 1):
        start_time = blend_traj.times[i]
        start_pose = blend_traj.poses[i]
        start_speed = blend_traj.speeds[i]

        end_time = blend_traj.times[i + 1]
        end_pose = blend_traj.poses[i + 1]
        end_speed = blend_traj.speeds[i + 1]

        if not (start_time <= time <= end_time):
            continue

        # if verbose:
        #     print(f"{time} between {start_time} and {end_time} at {i=}", end="..")

        if i % 2 == 0:
            print(
                f"Blending from ({start_time}, [{start_pose}]) ---> ({end_time}, [{end_pose}]) "
            ) if verbose else ""
            x = []
            for j in range(dim):
                fj = quint_poly(
                    start_time,
                    end_time,
                    start_pose[j],
                    end_pose[j],
                    start_speed[j],
                    end_speed[j],
                )
                xj = [fj(time), fj.deriv()(time), fj.deriv().deriv()(time)]
                x.append(xj)
        else:
            print(
                f"Linear from ({start_time}, [{end_time}]) ---> ({start_pose}, [{end_pose}])"
            ) if verbose else ""
            x = []
            for j in range(len(start_pose)):
                fj = slerp_func(start_time, end_time, start_pose[j], end_pose[j])
                xj = [fj(time), fj.deriv()(time), fj.deriv().deriv()(time)]
                # s = (time - start_time) / (end_time - start_time)
                # xj = [
                #     linear_func(start_pose[j], end_pose[j], s),
                #     np.mean([start_speed[j], end_speed[j]]),
                #     np.zeros_like(vmax)[j],
                # ]
                x.append(xj)

        return x
    return [[via_traj.poses[-1, 0], 0, 0]]


# %%
def main1d():
    t0 = 0
    via = np.array([5, 2, 4, 8, 3, 2, 6, 7, 4, 9]).reshape(-1, 1)
    vmax = np.array([1]).reshape(-1, 1)
    v0 = np.zeros_like(vmax)
    tacc = 1

    via_traj, blending_traj = compute_timing(via, vmax, tacc)

    via_traj.plot("o")
    blending_traj.plot("+")

    times = np.arange(0, np.max(via_traj.times), 0.01)
    poses = []
    speeds = []
    accs = []
    for t in times:
        pose, speed, acc = compute_command(t, via_traj, blending_traj, vmax)[0]
        poses.append(pose)
        speeds.append(speed)
        accs.append(acc)

    traj = Trajectory(times, poses, speeds, accs)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    traj.plot(ax=ax1)
    ax2.plot(traj.times, traj.speeds[:, 0])

    show_grid(0.5, np.max(times), 0.5, np.max(poses))
    ax2.plot(
        blending_traj.times,
        np.ones_like(blending_traj.times) * vmax[0],
        "--",
        label="vmax_0",
    )
    ax2.plot(
        blending_traj.times,
        -np.ones_like(blending_traj.times) * vmax[0],
        "--",
        label="vmax_0",
    )


# %%


def main2d(tacc=5):
    t0 = 0
    via = np.array([[0, 1, 3, 5, 12], [0, 2, 6, 8, 10]]).T * 10
    via = np.array([[0, 5, 10, 12, 20, 15, 17], [0, 0, 0, 12, 20, 20, 18]]).T
    vmax = np.array([1, 1])
    v0 = np.zeros_like(vmax)

    via_traj, blending_traj = compute_timing(via, vmax, tacc, v_nominal=2)

    times = np.arange(0, np.max(via_traj.times), 0.1)
    cmds = []
    for t in times:
        cmd = compute_command(t, via_traj, blending_traj, vmax, verbose=False)
        cmds.append(cmd)

    cmds = np.array(cmds)
    traj = Trajectory(times, cmds[:, :, 0], cmds[:, :, 1], cmds[:, :, 2])
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"Trajectory with {tacc=}s")
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title("x and y traj VS time")
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title("speed traj VS time")
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title("XY path")
    traj.plot(ax=ax1)
    via_traj.plot("o", ax=ax1)
    blending_traj.plot("+", ax=ax1)
    show_grid_on_points(blending_traj.times, blending_traj.poses[:, 0], ax=ax1)
    show_grid_on_points(blending_traj.times, blending_traj.poses[:, 1], ax=ax1)

    traj.plot_speed(ax=ax2)
    via_traj.plot_speed("o", ax=ax2)
    blending_traj.plot_speed("+", ax=ax2)
    ax2.plot(
        blending_traj.times,
        np.ones_like(blending_traj.times) * vmax[0],
        "--",
        label="vmax_0",
    )
    ax2.plot(
        blending_traj.times,
        np.ones_like(blending_traj.times) * vmax[1],
        "--",
        label="vmax_1",
    )
    ax2.plot(
        blending_traj.times,
        -np.ones_like(blending_traj.times) * vmax[0],
        "--",
        label="vmax_0",
    )
    ax2.plot(
        blending_traj.times,
        -np.ones_like(blending_traj.times) * vmax[1],
        "--",
        label="vmax_1",
    )

    ax3.plot(via[:, 0], via[:, 1], "o")
    ax3.plot(traj.poses[:, 0], traj.poses[:, 1])
    plt.grid()


def plot2d(traj: Trajectory, via: Trajectory):
    fig, ax = plt.subplots()
    ax.plot(traj.poses[:, 0], traj.poses[:, 1])
    ax.plot(via.poses[:, 0], via.poses[:, 1], "+")


# %%


def main2dprime(tacc=5.0):
    t0 = 0
    via = np.array(
        [[0, 5, 10, 12, 20, 15, 17, 17, 17], [0, 0, 0, 12, 20, 20, 18, 12, 3]]
    ).T
    vmax = np.array([1, 1])
    v0 = np.zeros_like(vmax)

    via_traj, blending_traj = compute_timing(
        via, vmax, tacc, v_nominal=2, max_acc=[0.5, 0.5]
    )

    times = np.arange(0, np.max(via_traj.times), 0.01)
    cmds = []
    for t in times:
        seg = 0
        while blending_traj.times[1:][seg] <= t < via_traj.times[-1]:
            seg += 1
        row = []
        for axis in range(blending_traj.dim):
            row.append(blending_traj.polys[axis][seg](t))
        cmds.append(row)
    cmds = np.array(cmds)

    plt.plot(cmds[:, 0], cmds[:, 1])
    plt.plot(via[:, 0], via[:, 1], "o")

    print(f"Final time: {via_traj.times[-1]:.2f} seconds.")


# %%
if __name__ == "__main__":
    # main1d()
    # main2d(1)
    main2dprime(10.0)

#%%
@dataclasses.dataclass
class SegmentData:
    polynomial: np.polynomial.Polynomial
    start_time: float
    end_time: float

#%%
via = np.array(
    [[0, 5, 10, 12, 20, 15, 17, 17, 17], [0, 0, 0, 12, 20, 20, 18, 12, 3]]
).T
t0 = 0
v_nominal = 3
vmax = np.array([1.3, 1.7])
max_acc = np.array([0.2, 0.3])
v0 = np.zeros((2,))

# compute segments
segments = np.diff(via, axis=0)
# compute time per segments
# compute the time of the slowest axis
t_slowest_axis = np.max(np.abs(segments / vmax), axis=1)
# Compute the time it would take if we move at v_nominal
t_nominal = np.linalg.norm(segments, axis=1) / v_nominal
# Time per segment should either be the max between the two previous one
t_segments = np.max((t_slowest_axis, t_nominal), axis=0)
# deduce speed per segments
speed_segments = np.divide(segments.T, t_segments.T).T
# compute blend duration based on max acceleration
blend_durations = np.max(
    np.abs(np.diff(np.vstack([v0, speed_segments]), axis=0) / max_acc), axis=1
)
# compute expected timing of each via points
via_points_times = np.concatenate([[0], np.cumsum(t_segments)])

for i in range(segments.shape[0]):
    seg = segments[i]

    t_slowest_axis = np.max(np.abs(seg / vmax))
    t_nominal = np.linalg.norm(seg) / v_nominal
    t_seg = np.max((t_slowest_axis, t_nominal))

    max_sync_speed = seg / t_seg

    # compute the acceleration time
    t_acc = np.abs(np.abs((max_sync_speed - prev_speed) / max_acc))

    # if tacc is smaller than tsegment

