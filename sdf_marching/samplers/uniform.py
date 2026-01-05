import numpy as np
from sdf_marching.circles import Circle
from sdf_marching.hasse import find_containment_hierarchy_max
from sdf_marching.overlap import get_overlaps_graph


def get_uniform_random(
    dist_function,  # callable that returns distance function
    epsilon,
    minimum_radius,
    num_samples,
    mins,
    maxs,
    rng=None,
    overlap_factor=1.0,
    start_point=None,
):
    test_positions, dists = get_uniform_random_query(
        dist_function, num_samples, mins, maxs, rng=rng, start_point=start_point
    )
    # generate circles based on the distance values
    # if dist < epsilon (clearance distance), ignored.
    circles = [
        Circle(pos, dist - epsilon) for pos, dist in zip(test_positions, dists) if dist - epsilon > minimum_radius
    ]

    # filter by containment -- this doesn't seem to do much in the current environment
    # hierarchy is a dict mapping from smaller, contained circle to bigger, containing circles
    # hierarchy = find_containment_hierarchy_max(circles, overlap_factor=overlap_factor)

    # max_circle_idxs = set(range(len(circles))) - set(hierarchy.keys())
    # max_circles = [
    #     circles[max_circle_idx]
    #     for max_circle_idx in max_circle_idxs
    # ]

    # get the overlap graph
    overlaps_graph = get_overlaps_graph(circles)
    return overlaps_graph, circles


def get_uniform_random_query(
    dist_function, num_samples, mins, maxs, rng=None, start_point=None  # callable that returns distance function
):
    test_positions = get_uniform_random_points(num_samples, mins, maxs, rng)
    if start_point is not None:
        test_positions = np.vstack((test_positions, start_point))
    dists = dist_function(test_positions)

    return test_positions, dists


def get_uniform_random_points(num_samples, mins, maxs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return np.stack([rng.uniform(low=min, high=max, size=[num_samples]) for min, max in zip(mins, maxs)], axis=-1)


def get_grid(num_samples, mins, maxs):
    ndim = len(mins)  # assume this is the ndim
    # TODO
    pass
