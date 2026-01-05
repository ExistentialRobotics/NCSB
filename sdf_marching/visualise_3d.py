import itertools
import polyscope as ps
import numpy as np

ps.set_up_dir("z_up")
ps.set_front_dir("y_front")
# importing this file initialises polyscope
ps.init()


def visualize_bubbles(overlaps_graph, name="bubbles"):
    return visualize_balls(
        np.array(overlaps_graph.vs["position"]), np.array(overlaps_graph.vs["radius"]).flatten(), name=name
    )


def visualize_balls(centers, radii, name="balls"):
    bubbles_cloud = ps.register_point_cloud(name, centers)

    bubbles_cloud.add_scalar_quantity("radius", radii)

    bubbles_cloud.set_point_radius_quantity("radius", autoscale=False)
    return bubbles_cloud


def visualize_trajectory(traj_np, name="trajectory"):
    trajectory_network = ps.register_curve_network(name, traj_np, "line")
    return trajectory_network


def visualize_triangulation(tri_list, nodes, name="graph"):
    edge_np = np.array([edge for tri in tri_list for edge in itertools.combinations(tri, 2)])

    triangulation_network = ps.register_curve_network(
        name,
        nodes,
        edge_np,
    )

    return triangulation_network


def visualize_quadrotor(traj_np, rot_np):
    pass
