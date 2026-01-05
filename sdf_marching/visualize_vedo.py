import vedo


def get_geometry_from_bubbles(overlaps_graph):
    meshes = [
        vedo.Sphere(r=bubble["radius"], pos=bubble["position"]).alpha(0.1).color("skyblue")
        for bubble in overlaps_graph.vs
    ]
    return meshes


def get_geometry_from_trajectory(positions):
    return [vedo.Line(positions, c="red", lw=3)]


def get_geometry_from_poses(positions, rotations):
    pass


def get_geometry_from_start_and_goal(start, goal):
    return [
        vedo.Sphere(r=0.1, pos=start).color("green"),
        vedo.Sphere(r=0.1, pos=goal).color("red"),
    ]
