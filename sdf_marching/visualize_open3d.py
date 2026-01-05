import open3d as o3d


def get_geometry_from_bubbles(overlaps_graph):
    meshes = [
        o3d.geometry.TriangleMesh().create_sphere(radius=bubble["radius"]).translate(bubble["position"])
        for bubble in overlaps_graph.vs
    ]
    return meshes


def get_geometry_from_trajectory(positions):
    line_idxs = [[i, i + 1] for i in range(positions.shape[0] - 1)]
    traj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(positions), lines=o3d.utility.Vector2iVector(line_idxs)
    )
    traj.paint_uniform_color([1, 0, 0])
    return [traj]


def get_geometry_from_poses(positions, rotations):
    pass


def get_geometry_from_start_and_goal(start, goal):
    start_mesh = o3d.geometry.TriangleMesh().create_sphere(radius=0.1).translate(start)
    goal_mesh = o3d.geometry.TriangleMesh().create_sphere(radius=0.1).translate(goal)
    start_mesh.paint_uniform_color([0, 1, 0])
    goal_mesh.paint_uniform_color([1, 0, 0])
    return [start_mesh, goal_mesh]
