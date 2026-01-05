import numpy as np
from sdf_marching.circles import Circle
from sdf_marching.overlap import get_overlaps_graph, position_to_max_circle_idx
from sdf_marching.geometry import norm

def trace_toward_circle(circle, dist_function, epsilon, min_radius, point):
    traced_circles = [Circle(point, dist_function(point) - epsilon)]

    while not traced_circles[-1].overlaps(circle):
        diff_vector = circle.centre - traced_circles[-1].centre
        distance = norm(diff_vector)
        new_centre = traced_circles[-1].centre + traced_circles[-1].radius * diff_vector / distance
        new_radius = dist_function(new_centre)[0] - epsilon

        # hit obstacle before getting to graph
        if new_radius < min_radius:
            return []
        traced_circles.append(Circle(new_centre.flatten(), new_radius))

    return traced_circles


def trace_toward_graph_min(overlaps_graph, dist_function, epsilon, min_radius, point):
    circles: list[Circle] = overlaps_graph.vs["circle"]
    # find the nearest circle
    dists = np.fromiter(map(lambda circle: circle.distance_to_point(point), circles), np.float32)
    near_idx = np.argmin(dists)
    near_circle = circles[near_idx]
    if near_circle.contains_point(point):  # already contained
        return overlaps_graph, near_idx
    # steer towards it
    new_circles = trace_toward_circle(near_circle, dist_function, epsilon, min_radius, point)
    if len(new_circles) == 0:  # hit obstacle before getting to graph
        return overlaps_graph, -1
    overlaps_graph_new = get_overlaps_graph(circles + new_circles)
    return overlaps_graph_new, position_to_max_circle_idx(overlaps_graph_new, point)


def trace_toward_graph_all(overlaps_graph, dist_function, epsilon, min_radius, point):
    circles = overlaps_graph.vs["circle"]
    new_circles = sum(
        (trace_toward_circle(circle, dist_function, epsilon, min_radius, point) for circle in circles), []
    )
    # now point will be contained, but potentially unconnected
    overlaps_graph_new = get_overlaps_graph(circles + new_circles)
    return overlaps_graph_new, position_to_max_circle_idx(overlaps_graph_new, point)


def trace_toward_graph_knn(overlaps_graph, dist_function, epsilon, min_radius, point, knn=4):
    circles: list[Circle] = overlaps_graph.vs["circle"]
    # find the nearest circle
    dists = np.fromiter(map(lambda circle: circle.distance_to_point(point), circles), np.float32)
    near_indices = np.argsort(dists)
    knn = min(knn, len(circles))
    if knn == 0:
        return overlaps_graph, -1

    near_indices = near_indices[:knn].tolist()
    all_new_circles = []
    for near_idx in near_indices:
        near_circle = circles[near_idx]
        if near_circle.contains_point(point):  # already contained
            return overlaps_graph, near_idx
        # steer towards it
        new_circles = trace_toward_circle(near_circle, dist_function, epsilon, min_radius, point)
        if len(new_circles) == 0:  # hit obstacle before getting to graph
            continue
        if len(all_new_circles) == 0:
            all_new_circles.append(new_circles[0])  # the first circle is placed at the point
        all_new_circles.extend(new_circles[1:])  # the rest are the trace
        if len(all_new_circles) > 1:
            break  # stop when the point is contained in the graph
    if len(all_new_circles) == 0:  # failed among knn near circles
        return overlaps_graph, -1
    overlaps_graph_new = get_overlaps_graph(circles + all_new_circles)
    return overlaps_graph_new, position_to_max_circle_idx(overlaps_graph_new, point)


def update_overlaps_graph_sdf_tracing(
    overlaps_graph,
    dist_function,
    epsilon,
    minimum_radius,
    start_point,
    end_point,
    knn=4,
):
    if knn > 0:
        overlaps_graph, start_idx = trace_toward_graph_knn(
            overlaps_graph,
            dist_function,
            epsilon,
            minimum_radius,
            np.array(start_point),
            knn,
        )
        overlaps_graph, end_idx = trace_toward_graph_knn(
            overlaps_graph,
            dist_function,
            epsilon,
            minimum_radius,
            np.array(end_point),
            knn,
        )
    else:
        overlaps_graph, start_idx = trace_toward_graph_all(
            overlaps_graph,
            dist_function,
            epsilon,
            minimum_radius,
            np.array(start_point),
        )
        overlaps_graph, end_idx = trace_toward_graph_all(
            overlaps_graph,
            dist_function,
            epsilon,
            minimum_radius,
            np.array(end_point),
        )
    return overlaps_graph, start_idx, end_idx
