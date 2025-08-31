import numpy as np
from robotics_algorithms import control as ct
from robotics_algorithms.utils.math import wrap_to_pi


class AckermanCar(ct.NonlinearIOSystem):
    def __init__(
        self,
        wheel_separation,
        wheel_base,
        front_wheel_radius,
        rear_wheel_radius,
        **kwargs
    ):
        car_params = dict(
            w=wheel_separation,
            l=wheel_base,
            rf=front_wheel_radius,
            rr=rear_wheel_radius,
        )
        super().__init__(
            self.update_fn,
            self.output_fn,
            inputs=["speed", "steer"],
            outputs=["x", "y", "theta"],
            states=["x", "y", "theta"],
            name="ackerman_car",
            params=car_params,
            **kwargs
        )

    def update_fn(self, t, state, commands, params):
        if params is None:
            params = {}

        x, y, theta = state
        v, delta = commands

        l = params.get("l", 1)
        w = params.get("w", 0.7)

        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)
        theta_dot = v * np.tan(delta) / l
        delta_dot = 0

        return [xdot, ydot, theta_dot]

    def output_fn(self, t, state, u, params):
        x, y, theta = state
        theta = wrap_to_pi(theta)
        return [theta]


# %%

# if __name__ == "__main__":
ackerman_car = AckermanCar(0.6, 1, 0.3, 0.3)
t = np.arange(0, 100, 0.1)

#%%
u = np.vstack((0 * t + 1, np.sin(t / 10) * 0.12))
sol = ct.input_output_response(ackerman_car, t, u)

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_title('XY')
ax1.plot(sol.states[0], sol.states[1])
ax1.axis('equal')
ax2.set_title('speed')
ax2.plot(sol.t, sol.inputs[0])
ax3.set_title('steer')
ax3.plot(sol.t, sol.inputs[1])
plt.show()

# %%
kp = 0.15
heading_controller = ct.ss(
    lambda t, state, command, params: np.array([[0]]),
    lambda t, state, command, params: np.array(
        ([[wrap_to_pi(kp * (command[0] - command[1]))]])
    ),
    inputs=("theta_ref", "theta"),
    outputs=("delta"),
    states=('x'),
    name="heading_controller",
)
controlled_car = ct.InterconnectedSystem(
    [ackerman_car, heading_controller],
    connections=[
        ("ackerman_car.steer", "heading_controller.delta"),
        ("heading_controller.theta", "ackerman_car.theta"),
    ],
    inplist=["heading_controller.theta_ref", "ackerman_car.speed"],
    inputs=["desired_theta", "speed"],
    outlist=["ackerman_car.x", "ackerman_car.y", "ackerman_car.theta"],
    outputs=("x", "y", "theta"),
)

half_time_index = len(t) // 3
half_time_index2 = len(t) // 2
u = np.vstack((np.zeros_like(t), np.zeros_like(t) + 4))
u[0][half_time_index:] = 0.3
u[0][half_time_index2:] = 0.5

sol = ct.input_output_response(controlled_car, t, u, [1, 1, -0.3])

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_title('XY')
ax1.plot(sol.outputs[0][0], sol.outputs[1][0], 'k*')
ax1.plot(sol.outputs[0], sol.outputs[1])
ax1.axis('equal')

ax2.set_title('theta')
ax2.plot(sol.t, sol.outputs[2], label="theta")
ax2.plot(sol.t, sol.inputs[0], label="theta_ref")
ax2.legend()

ax3.set_title('speed')
ax3.plot(sol.t, sol.inputs[1])

plt.show()
