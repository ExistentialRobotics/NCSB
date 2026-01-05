import cvxpy
import numpy as np
from sdf_marching.bezier import BezierPolynomial


def l2_cost(traj):
    return cvxpy.sum(cvxpy.norm(traj[:-1, :] - traj[1:, :], axis=1))


def circle_constraint(position, circle):
    # TODO: use parameters for circles!
    return cvxpy.sum_squares(position - circle.centre) <= circle.radius**2


def bezier_cost_all(bps, weights=None):
    if weights is None:
        weights = [0.1, 0.1, 0.1]
    return sum(bezier_cost_individual(bp, weights=weights) for bp in bps)


def bezier_cost_individual(bp, weights=None):
    if weights is None:
        weights = [0.1, 0.1, 0.1]
    cost = 0.0
    derivative = bp
    for weight in weights:
        # call derivative recursively
        derivative = derivative.derivative()
        cost += weight * derivative.norm_square_integral()

    return cost


def edgeseq_to_traj_constraint_bezier(edge_seq, start_position, end_position, bezier_order=5):  # igraph.EdgeSeq object
    ndim = start_position.size

    bps = [BezierPolynomial(cvxpy.Variable((bezier_order, ndim), f"traj{idx}")) for idx in range(len(edge_seq) + 1)]

    # enforce start and end
    constr_list = [
        bps[0].start == start_position,
        bps[-1].end == end_position,
        bps[0].derivative().start == np.zeros_like(start_position),
        bps[-1].derivative().end == np.zeros_like(end_position),
    ]

    # enforce containment in circles
    constr_list += [
        circle_constraint(point, edge.source_vertex["circle"])
        for idx, edge in enumerate(edge_seq)
        for point in bps[idx].points
    ]

    # also add the last circle
    if len(edge_seq) > 0:
        constr_list += [circle_constraint(point, edge_seq[-1].target_vertex["circle"]) for point in bps[-1].points]

    for idx, edge in enumerate(edge_seq):
        constr_list += [
            bps[idx].end == bps[idx + 1].start,
            circle_constraint(bps[idx].end, edge.target_vertex["circle"]),
        ]  # the last constraint ensures that edge_position is in the intersection.
        # trivial becaues bps[idx].end is constrained to be in source_vertex

        # maybe do this at many levels?
        constr_list += [
            bps[idx].derivative().end == bps[idx + 1].derivative().start,
        ]

    return bps, constr_list  # TODO: should i return edge intermediates as well? maybe not


def edgeseq_to_traj_constraint_straight_line(
    edge_seq,  # igraph.EdgeSeq object
    start_position,
    end_position,
):
    ndim = start_position.shape[0]
    len_path = len(edge_seq)
    traj = cvxpy.Variable((2 * len_path + 2, ndim))  # every even position is for edge, every odd is for vertex

    constr_list = [traj[0, :] == start_position, traj[-1, :] == end_position]

    # source circle added to 2*idx-th position
    constr_list += [
        circle_constraint(traj[2 * idx, :], edge.source_vertex["circle"]) for idx, edge in enumerate(edge_seq)
    ]

    # source + target circle added to (2*idx+1)-th position
    constr_list += [
        circle_constraint(traj[2 * idx + 1, :], edge.source_vertex["circle"]) for idx, edge in enumerate(edge_seq)
    ]

    constr_list += [
        circle_constraint(traj[2 * idx + 1, :], edge.target_vertex["circle"]) for idx, edge in enumerate(edge_seq)
    ]

    # target circle added to (2*idx+2)-th position
    constr_list += [
        circle_constraint(traj[2 * idx + 2, :], edge.target_vertex["circle"]) for idx, edge in enumerate(edge_seq)
    ]

    return traj, constr_list


def circle_path_constraint_intersect_only(circle_path, ndim=2):
    len_path = len(circle_path)  # get the length of the circles path
    traj = cvxpy.Variable((ndim, len_path + 2))

    constr_list = [circle_constraint(traj[idx, :], circle_path[idx - 1]) for idx in range(1, len_path)]

    constr_list += [circle_constraint(traj[idx, :], circle_path[idx]) for idx in range(1, len_path)]

    return traj, constr_list


def bezier_path_constraint_intersect_only(circle_path, ndim=2, order=4):
    len_path = len(circle_path)
    num_bezier = len_path - 1
    bps = [BezierPolynomial(cvxpy.Variable(ndim, order + 1))]
    constr_list = [bezier_in_circle(bp, circle) for bp, circle in zip(bps, circle_path[:-1])]


def bezier_in_circle(bp, circle):
    pass


def bezier_cost(bp):
    bp.derivative().derivative().derivative().derivative().integral_norm_sq()
