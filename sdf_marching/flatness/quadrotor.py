import numpy as np
from sdf_marching.flatness.utils import normalise

class QuadrotorDynamics:
    def __init__(self, position_poly, yaw_poly=None):
        # list of BezierPolynomials 
        self.position_poly = position_poly
        # maybe move these to a setter? 
        self.velocity_poly = self.position_poly.derivative()
        self.acceleration_poly = self.velocity_poly.derivative()
        self.yaw_poly = yaw_poly

    # get the position and rotation matrix for the quadrotor at time t
    def query(self, time):
        pos = self.pos(time)
        body_x, body_y, body_z = self.body_x(time), self.body_y(time), self.body_z(time)
        return pos, np.stack([body_x, body_y, body_z], axis=-1)

    def pos(self, time):
        return self.position_poly.query(time)
    
    def vel(self, time):
        return self.velocity_poly.query(time)
    
    def acc(self, time):
        return self.acceleration_poly.query(time)

    def body_x(self, time):
        return normalise(
            np.cross(
                self.body_y(time),
                self.body_z(time)
            )
        )

    def body_y(self, time):
        body_z = self.body_z(time)
        heading = self.heading(time)
        body_y_unnorm = np.cross(body_z, heading)
        return normalise(body_y_unnorm)

    def body_z(self, time):
        thrust = self.thrust(time)
        return normalise(thrust)

    def thrust(self, time):
        acc = self.acc(time)
        gravity = np.zeros_like(acc)
        gravity[..., -1] = -9.8
        return acc + gravity

    def heading(self, time):
        if self.yaw_poly is not None:
            yaw = self.yaw_poly.query(time)
            return np.stack(
                [np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)],
                axis=-1
            )
        else:
            vel = self.vel(time)
            # normalize to get direction
            return normalise(vel)

    