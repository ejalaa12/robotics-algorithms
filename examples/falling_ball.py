import numpy as np


"""
@brief FallingBallModel modelises a ball that is falling subject to gravity

State:
    X = (z, vz)

Observations:
    Y = (z)

Model:
    Xdot = (vz, az) = (dz/dt, dv/dt)
    Xn = Xp + dt * Xdot
       = ( zp )      ( vp )
         ( vp ) + dt ( g  )
       = I X + 
       = ( 1 dt )     ( 0 )
         ( 0  1 ) X + ( 1 ) g


"""
class FallingBallModel(object):
    def __init__(self, gravity=9.81, z0=0, v0=0):
        self._x = np.array([z0, v0]).reshape(-1, 1)
        self._gravity = gravity

    def update(self, dt):
        A = np.array([[1, dt], [0, 1]])
        self._x = A @ self._x + np.array([[0], [self._gravity]])

    @staticmethod
    def run_sim(gravity=9.81, z0=0, v0=0, duration=100, dt=0.1, with_noise=False):
        ball = FallingBallModel(gravity, z0, v0)
        time = np.arange(0, duration, dt)
        traj = np.zeros((time.size, ball._x.size))
        traj[0] = ball._x.reshape(1, -1)
        
        for i, t in enumerate(time):
            ball.update(dt)
            traj[i] = ball._x.reshape(1, -1)

        return time, traj


if __name__ == "__main__":
    time, traj = FallingBallModel.run_sim()
    print(time)
    print(traj)


