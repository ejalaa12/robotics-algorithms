import numpy as np
import control as ct


class DubinsCar(ct.NonlinearIOSystem):
    def __init__(self):
        self.car_params = {}
        super().__init__(
            self._update_fn,
            self._output_fn,
            states=["x", "y", "theta"],
            inputs=["v", "w"],
            outputs=["theta"],
            params=self.car_params,
            name="dubins",
        )

    def _update_fn(self, time, state, command, params=None):
        if params is None:
            params = {}

        x, y, theta = state
        v, w = command

        return [v * np.cos(theta), v * np.sin(theta), w]

    def _output_fn(self, time, state, command, params=None):
        if params is None:
            params = {}

        x, y, theta = state

        # todo(ejalaa) add noise
        return [theta]




if __name__ == "__main__":
    car = DubinsCar()
    dt = 0.1
    t = np.arange(0, 100, dt)
    u = np.vstack((1 + t * 0, np.sin(t / 10) * 0.1))
    sol = ct.input_output_response(car, t, u, [0, 0, 0])

    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(sol.states[0], sol.states[1])
    ax2.plot(sol.t, sol.inputs[0])
    ax3.plot(sol.t, sol.inputs[1])
    plt.show()
