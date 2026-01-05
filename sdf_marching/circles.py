import numpy as np
from sdf_marching.geometry import norm

class Circle:
    def __init__(self, centre, radius):

        if radius < 0:
            raise ValueError("radius must be positive!")
        
        self.centre = np.array(centre)
        self.radius = radius

    # main functions
    def distance_to(self, other):
        return norm(self.centre - other.centre)

    ## Hausdorff distance
    ## this is given by <dist_between_centres> + <absolute diff. in radius> 
    ## https://hrcak.srce.hr/file/292566
    def hausdorff_distance_to(self, other):
        return self.distance_to(other) + np.abs(self.radius - other.radius)

    def single_sided_hausdorff_distance_to(self, other):
        return self.distance_to(other) + self.radius - other.radius

    def laguerre_distance_to(self, other):
        return np.square(self.distance_to(other)) - np.square(self.radius - other.radius)

    def distance_centre_to_point(self, point):
        return norm(self.centre - point, axis=-1)

    def distance_to_point(self, point):
        return norm(self.centre - point, axis=-1) - self.radius

    def distance_to_point_trunc(self, point):
        return np.abs(self.distance_to_point(point), 0.)

    def contains_point(self, point) -> bool:
        return self.distance_to_point(point) < 0.0

    def contains(self, other, overlap_factor=1.0) -> bool:
        return self.distance_to(other) <= self.radius - overlap_factor * other.radius

    def is_contained_by(self, other) -> bool:
        return other.contains(self)

    def overlaps(self, other) -> bool:
        return self.distance_to(other) < self.radius + other.radius
    
    # syntactic sugar
    def __str__(self):
        return f"Circle(r={self.radius},loc={self.centre})"

    def __repr__(self):
        return str(self)

    # == is assertion on radius and centre
    def __eq__(self, other):
        return np.all(self.centre == other.centre) and self.radius == other.radius

    # <, <=, >, and >= are assertions on SIZE (i.e. )
    def __lt__(self, other):
        return self.radius < other.radius
    
    def __gt__(self, other):
        return self.radius > other.radius
    
    def __le__(self, other):
        return self.radius <= other.radius
    
    def __ge__(self, other):
        return self.radius >= other.radius
    
    # contains means containment
    def __contains__(self, other):
        return self.contains(other)