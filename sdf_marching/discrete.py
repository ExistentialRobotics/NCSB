def compute_edge_cost(
    cost_function,
    overlaps_graph,
):
    return [cost_function(edge.source_vertex["circle"], edge.target_vertex["circle"]) for edge in overlaps_graph.es]


def get_shortest_path(cost_function, overlaps_graph, start_idx, end_idx, cost_name=None, return_epath=True):
    cost = compute_edge_cost(cost_function, overlaps_graph)

    if cost_name is not None:
        overlaps_graph.es[cost_name] = cost

    return overlaps_graph.get_shortest_paths(
        start_idx, to=end_idx, weights=cost, output="epath" if return_epath else "vpath"
    )


def epath_to_vpath(overlaps_graph, epath):
    vpath = []
    for edge_idx in epath:
        # assume that epath is correct
        vpath.append(overlaps_graph.es[edge_idx].source_vertex.index)

    vpath.append(overlaps_graph.es[epath[-1]].target_vertex.index)

    return vpath
