import igraph as ig
import numpy as np
from .circles import Circle
import multiprocessing as mp

def get_overlaps_single(args):
    circles, start, end = args
    overlaps = []
    for k in range(start, end):
        i = int(np.sqrt(2 * k)) + 1
        j = (2 * k - i * (i - 1)) // 2
        if j < 0:
            i = i - 1
            j = (2 * k - i * (i - 1)) // 2
        circle_i = circles[i]
        circle_j = circles[j]
        if circle_i.overlaps(circle_j) and not circle_i == circle_j:
            overlaps.append([i, j])
    return overlaps


def get_overlaps(circles):
    total = len(circles) * (len(circles) - 1) // 2
    n = mp.cpu_count()
    chunk_size = total // n
    chunks = [(circles, i * chunk_size, (i + 1) * chunk_size) for i in range(n - 1)]
    chunks.append((circles, (n - 1) * chunk_size, total))
    with mp.Pool(n) as pool:
        return sum(pool.map(get_overlaps_single, chunks), [])

    # ans2 = [
    #     [idx_i, idx_j]
    #     for idx_i, circle_i in enumerate(circles)
    #     for idx_j, circle_j in enumerate(circles[:idx_i])
    #     if circle_i.overlaps(circle_j) and not circle_i == circle_j
    # ]
    # assert ans1 == ans2
    # return ans1


def get_overlaps_graph(circles):
    return ig.Graph(
        n=len(circles),
        edges=get_overlaps(circles),
        vertex_attrs={
            "circle": circles,
            "position": [circle.centre for circle in circles],
            "radius": [circle.radius for circle in circles],
        },
    )


def update_overlaps_graph_single_step(overlaps_graph, dist_function, epsilon, minimum_radius, start_point, end_point):
    start_idx = position_to_max_circle_idx(overlaps_graph, start_point)
    end_idx = position_to_max_circle_idx(overlaps_graph, end_point)
    circles = overlaps_graph.vs["circle"]
    if start_idx < 0:  # try to create a circle at the start point and merge it to the graph if possible
        print("try to create a circle at the start point and merge it to the graph if possible")
        radius = dist_function(np.array(start_point))[0] - epsilon
        if radius < minimum_radius:
            print("failed because radius < minimum_radius")
            return overlaps_graph, start_idx, end_idx
        overlap_distances = [-circle.distance_to_point(start_point) + radius for circle in circles]
        overlap_idx = np.argmax(overlap_distances)  # find the maximum overlap
        if overlap_distances[overlap_idx] < 0:  # failed
            return overlaps_graph, start_idx, end_idx
        new_circle = Circle(start_point, radius)
        circles.append(new_circle)
    if end_idx < 0:  # try to create a circle at the end point and merge it to the graph if possible
        print("try to create a circle at the end point and merge it to the graph if possible")
        radius = dist_function(np.array(end_point))[0] - epsilon
        if radius < minimum_radius:
            return overlaps_graph, start_idx, end_idx
        overlap_distances = [-circle.distance_to_point(end_point) + radius for circle in circles]
        overlap_idx = np.argmax(overlap_distances)
        if overlap_distances[overlap_idx] < 0:
            print("failed because radius < minimum_radius")
            return overlaps_graph, start_idx, end_idx
        new_circle = Circle(end_point, radius)
        circles.append(new_circle)

    # update the graph with the new circles
    if start_idx < 0 or end_idx < 0:
        overlaps_graph = get_overlaps_graph(circles)
    if start_idx < 0:
        start_idx = position_to_max_circle_idx(overlaps_graph, start_point)
    if end_idx < 0:
        end_idx = position_to_max_circle_idx(overlaps_graph, end_point)
    return overlaps_graph, start_idx, end_idx


def point_in_circle(point, center, radius):
    return np.linalg.norm(point - center) <= radius


def position_to_circle_idx(overlaps_graph, position):
    """Return indices of all circles containing the position"""
    return [vertex.index for vertex in overlaps_graph.vs 
            if vertex["circle"].contains_point(position)]


def position_to_max_circle_idx(overlaps_graph, position):
    """Return index of largest circle containing the position"""
    candidate_idxs = position_to_circle_idx(overlaps_graph, position)
    if not candidate_idxs:
        return -1
    
    # Find circle with maximum radius among candidates
    radii = overlaps_graph.vs["radius"]
    return candidate_idxs[np.argmax([radii[idx] for idx in candidate_idxs])]


def extract_connected_component_circles(circles, start_point):
    graph = get_overlaps_graph(circles)
    start_idx = position_to_max_circle_idx(graph, start_point)
    connected_graph = extract_connected_component_graph(graph, start_idx)
    connected_circles = [v["circle"] for v in connected_graph.vs()]
    return connected_circles, connected_graph


def extract_connected_component_graph(graph, start_idx):
    components = graph.connected_components()
    component_idx = components.membership[start_idx]
    return components.subgraphs()[component_idx]


def get_overlaps_graph_numpy(centers, radii):
    n_circles = len(centers)
    # Create empty graph
    graph = ig.Graph(n_circles, directed=False)
    
    # Store centers and radii as vertex attributes
    graph.vs["center"] = centers
    graph.vs["radius"] = radii
    # Also store Circle objects for compatibility
    graph.vs["circle"] = [Circle(center, radius) for center, radius in zip(centers, radii)]
    
    # Compute all pairwise distances between centers
    dists = np.zeros((n_circles, n_circles))
    for i in range(n_circles):
        dists[i] = np.linalg.norm(centers - centers[i], axis=1)
    
    # Find overlapping circles
    sum_radii = radii[:, None] + radii[None, :]
    overlaps = dists <= sum_radii
    
    # Add edges where circles overlap (excluding self-loops)
    edges = np.where(np.triu(overlaps, k=1))
    graph.add_edges(list(zip(edges[0], edges[1])))
    
    return graph



