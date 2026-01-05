import cvxpy
import numpy as np

def get_big_m_problem(
    time_horizon, # N_t, integer
    circles # circles
):
    belongs_to_circles = cvxpy.Variable(
        (time_horizon, len(circles)),
        boolean=True
    )
    traj = cvxpy.Variable(
        (circles[0].centre.shape[0], time_horizon) 
    )

    # robot must be in at least one circle at all times
    constr = [
        cvxpy.sum(belongs_to_circles, axis=1) >= 1
    ]

    # this encodes if belongs_to_circle[t, c] == 1, then robot is in circle c at time t.
    constr += [
        micp_circle_constraint(
            traj[:, time],
            circle,
            belongs_to_circles[time, circle_idx]
        )
        for circle_idx, circle in enumerate(circles)
        for time in range(time_horizon)
    ]

    # constr += [
    #     micp_circle_constraint(
    #         traj[:, time-1],
    #         circle,
    #         belongs_to_circles[time, circle_idx]
    #     )
    #     for circle_idx, circle in enumerate(circles)
    #     for time in range(1, time_horizon)
    # ]

    return traj, belongs_to_circles, constr


def micp_circle_constraint(
    position,
    circle,
    belongs_to_circle,
    big_m = 1e10
):
    return cvxpy.sum_squares(position - circle.centre) <= circle.radius * circle.radius + big_m * (1 - belongs_to_circle)


def graph_shortest_path(overlaps_graph, start_v_idx, end_v_idx, weights):
    edge_variables, constr = graph_path_constraints(overlaps_graph, start_v_idx, end_v_idx)

    cost = np.array(weights) @ edge_variables

    return edge_variables, constr, cost

def graph_path_constraints(overlaps_graph, start_v_idx, end_v_idx):
    # create variables for whether edge is in path
    # these end up being integer anyway without specifying such a constraint.
    path_edge_variables = cvxpy.Variable( len(overlaps_graph.es), boolean=True) 

    constr = [] # list of constraints

    # integer constraints for each vertex
    for vertex in overlaps_graph.vs:
        in_edges = [edge.index for edge in vertex.in_edges()]
        out_edges = [edge.index for edge in vertex.out_edges()]

        in_edge_sum = cvxpy.sum(path_edge_variables[in_edges])
        out_edge_sum = cvxpy.sum(path_edge_variables[out_edges])

        # NOTE: start/end encoding differ from GCS paper because we likely have cycles
        # see https://lidicky.name/oldteaching/18.566/l13%20-%20Shortest%20Path%20and%20Linear%20Programming.pdf
        # at least one edge must come out of start
        if vertex.index == start_v_idx:
            constr += [
                out_edge_sum - in_edge_sum == 1
            ]
        # at least one edge must come into end
        elif vertex.index == end_v_idx:
            constr += [
                in_edge_sum - out_edge_sum == 1
            ]
        # for other edges, going in means coming out. only visit once
        else:
            constr += [
                in_edge_sum == out_edge_sum,
                out_edge_sum <= 1
            ]

    return path_edge_variables, constr


