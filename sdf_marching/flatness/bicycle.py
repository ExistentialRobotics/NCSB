import numpy as np
from sdf_marching.flatness.utils import normalise

class BicycleDynamics:
    def __init__(self, position_poly):
        self.position_poly = position_poly
        self.velocity_poly = self.position_poly.derivative()

    def query(self, time):
        return self.pos(time), 

    def pos(self, time):
        return self.position_poly.query(time) 

    def vel(self, time):
        return self.velocity_poly.query(time)

    def yaw(self, time):
        vel = self.vel(time)
        # normalize to get direction
        return np.atan2(vel[:, 1], vel[:, 0])

    def heading(self, time):
        vel = self.vel(time)
        # normalize to get direction
        return normalise(vel)