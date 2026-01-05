from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import itertools

def triangulation_to_lines(tri_list, S):
	# Plot the power triangulation
	edge_set = frozenset(
		tuple(sorted(edge)) 
        for tri in tri_list 
        for edge in itertools.combinations(tri, 2)
	)
	line_list = LineCollection(
		[(S[i], S[j]) for i, j in edge_set], 
		lw = 1., 
		colors = '.9'
	)
	return line_list

def voronoi_map_to_lines(voronoi_cell_map):
    edge_map = { }
    for segment_list in voronoi_cell_map.values():
        for edge, (A, U, tmin, tmax) in segment_list:
            edge = tuple(sorted(edge))
            if edge not in edge_map:
                if tmax is None:
                    tmax = 10
                if tmin is None:
                    tmin = -10

                edge_map[edge] = (A + tmin * U, A + tmax * U)

    line_list = LineCollection(edge_map.values(), lw = 1., colors = 'k')
    return line_list
	