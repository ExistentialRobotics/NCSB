from scipy.spatial import ConvexHull
from sdf_marching.power_diagram.delaunay import lift
import igraph as ig
from itertools import combinations
import numpy as np
from sdf_marching.circles import Circle

def hull_to_edge_list(hull):
    return set(
        tuple(sorted(edge))
        for tri, eq in zip(hull.simplices, hull.equations)
        for edge in combinations(tri, 2)
        if eq[-1] <= 2
    )

def compute_laguerre_norm(centres, radii):
    return np.sum(np.square(centres), axis=-1) - np.square(radii)

def is_overlapping_edge(edge):
    A_ij = edge.source_vertex["position"] - edge.target_vertex["position"]
    b_ij = 0.5 * ( compute_laguerre_norm(
        edge.source_vertex["position"],
        edge.source_vertex["radius"]
    ) - compute_laguerre_norm(
        edge.target_vertex["position"],
        edge.target_vertex["radius"]
    ) )
    distance_to_plane = (A_ij.T @ edge.source_vertex["position"] - b_ij) / (np.linalg.norm(A_ij))
    return distance_to_plane <= edge.source_vertex["radius"]

class PowerDiagram:
    def __init__(self, centres, radii):
        points_lifted = lift(centres, radii)
        #TODO: escape cases where c-hull cannot be found
        self.hull = ConvexHull(
            points_lifted,
            incremental=True
        )
        self._graph = ig.Graph(
            n=self.hull.vertices.size, # hull.vertices indexes into hull.points
            edges= hull_to_edge_list(self.hull),
            vertex_attrs = {
                "position": centres,
                "radius": radii
            }
        )

    def add_ball(self):
        pass

    def get_closest_ball_point_linear(self, points):
        centres = np.array(self._graph.vs["position"])
        radii = np.array(self._graph.vs["radius"])
        half_l_norm = 0.5 * compute_laguerre_norm(centres, radii)

        plane_value = centres @ np.atleast_2d(points).T - half_l_norm[..., np.newaxis] # the c_i^Tx - d_i for each plane

        # compare plane value between edges
        graph_version = np.array( [
            np.all(
                plane_value[vertex, :] > plane_value[self._graph.neighbors(vertex), :], axis=0
            )
            for vertex in range(len(self._graph.vs))
        ])
        belongs_to_vertex = np.argmax(
            plane_value,
            axis=0
        )

        # belongs_to_source = [
        #     plane_value[:, edge.source] < plane_value[:, edge.target]
        #     for edge in self._graph.es
        # ]

        return belongs_to_vertex, plane_value, graph_version

    def get_overlaps_graph(self):
        overlaps_graph = self._graph.copy() # take a shallow copy

        edge_to_delete = [
            edge.index
            for edge in overlaps_graph.es
            if not is_overlapping_edge(edge)
        ]

        overlaps_graph.delete_edges(edge_to_delete)

        for v in overlaps_graph.vs:
            v["circle"] = Circle(
                v["position"],
                v["radius"]
            )

        return overlaps_graph
