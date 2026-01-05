from collections import defaultdict
import itertools
import numpy as np
from .utils import norm2, normalized
from scipy.spatial import ConvexHull


# --- Delaunay triangulation --------------------------------------------------

def get_triangle_normal(A, B, C):
	return normalized(
		np.cross(A, B) 
		+ np.cross(B, C) 
		+ np.cross(C, A)
	)

def get_power_circumcenter(A, B, C):
	N = get_triangle_normal(A, B, C)
	return (-.5 / N[-1]) * N[:-1]

def is_ccw_triangle(A, B, C):
    M = np.concatenate([np.stack([A, B, C]), np.ones((3, 1))], axis = 1)
    is_ccw = np.linalg.det(M) > 0
    if not is_ccw:
        print("was not ccw")
    else:
        print("was ccw")
    return is_ccw

def is_oriented_simplex(points):
	M = np.concatenate(
		[points, np.ones((points.shape[0], 1))],
		axis = 1
    )
	return np.linalg.det(M) > 0

def permute(points):
    return np.concatenate(
		[points[:1], np.flip(points[1:])]
	)

def enforce_oriented_simplex(simplex_points):
	return permute(simplex_points) if not is_oriented_simplex(simplex_points) else simplex_points

def lift(points, radii):
	laguerre_norm = np.sum(np.square(points), axis=-1) - radii ** 2
	return np.concatenate(
		[points, laguerre_norm[:, np.newaxis]],
		axis = -1
    )

def get_power_circumcenter_nd(S, R):
    R_sqr = R ** 2
    
    Sp = S[1:] - S[0]
    Sp_norm_sqr = np.sum(Sp ** 2, axis = 1)
    
    U = ((Sp_norm_sqr + R_sqr[0] - R_sqr[1:]) / (2 * Sp_norm_sqr))[:, None] * Sp + S[0]

    return np.linalg.solve(Sp, np.sum(U * Sp, axis = 1))

def get_power_triangulation(S, R):
	# Compute the lifted weighted points
	S_lifted = lift(S, R)

	# # Special case for 3 points
	# if S.shape[0] == 3:
	# 	if is_ccw_triangle(S[0, ...], S[1, ...], S[2, ...]):
	# 		return [[0, 1, 2]], np.array([get_power_circumcenter(*S_lifted)])
	# 	else:
	# 		return [[0, 2, 1]], np.array([get_power_circumcenter(*S_lifted)])

	# Compute the convex hull of the lifted weighted points
	hull = ConvexHull(S_lifted)
	
	# Extract the Delaunay triangulation from the lower hull
	tri_list = tuple(
		tri
		if is_oriented_simplex(S[ tri ]) 
		else permute(tri)  
		for tri, eq in zip(hull.simplices, hull.equations) 
		if eq[2] <= 0
	)
	
	# Job done
	return tri_list, hull

def get_edge_map(tri_list):
	edge_map = defaultdict(lambda: [])
	for i, tri in enumerate(tri_list):
		for edge in itertools.combinations(tri, 2):
			edge = tuple(sorted(edge))
			edge_map[edge].append(i)
	return edge_map

# --- Compute Voronoi cells ---------------------------------------------------

'''
Compute the segments and half-lines that delimits each Voronoi cell
  * The segments are oriented so that they are in CCW order
  * Each cell is a list of (i, j), (A, U, tmin, tmax) where
     * i, j are the indices of two ends of the segment. Segments end points are
       the circumcenters. If i or j is set to None, then it's an infinite end
     * A is the origin of the segment
     * U is the direction of the segment, as a unit vector
     * tmin is the parameter for the left end of the segment. Can be -1, for minus infinity
     * tmax is the parameter for the right end of the segment. Can be -1, for infinity
     * Therefore, the endpoints are [A + tmin * U, A + tmax * U]
'''
def get_voronoi_cells(S, V, tri_list):
	# Keep track of which edge separate which tetrahedra
	edge_map = get_edge_map(tri_list)

	# For each triangle
	voronoi_cell_map = defaultdict(lambda: [])
	
	for i, (a, b, c) in enumerate(tri_list):
		# For each edge of the triangle
		for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
		# Finite Voronoi edge
			edge = tuple(sorted((u, v)))
			if len(edge_map[edge]) == 2:
				j, k = edge_map[edge]
				if k == i:
					j, k = k, j
				
				# Compute the segment parameters
				U = V[k] - V[j]
				U_norm = norm2(U)				

				# Add the segment
				voronoi_cell_map[u].append(((j, k), (V[j], U / U_norm, 0, U_norm)))
			else: 
			# Infinite Voronoi edge
				# Compute the segment parameters
				A, B, C, D = S[u], S[v], S[w], V[i]
				U = normalized(B - A)
				I = A + np.dot(D - A, U) * U
				W = normalized(I - D)
				if np.dot(W, I - C) < 0:
					W = -W	
			
				# Add the segment
				voronoi_cell_map[u].append(((edge_map[edge][0], -1), (D,  W, 0, None)))	
				voronoi_cell_map[v].append(((-1, edge_map[edge][0]), (D, -W, None, 0)))				

	# Order the segments
	def order_segment_list(segment_list):
		# Pick the first element
		first = min((seg[0][0], i) for i, seg in enumerate(segment_list))[1]

		# In-place ordering
		segment_list[0], segment_list[first] = segment_list[first], segment_list[0]
		for i in range(len(segment_list) - 1):
			for j in range(i + 1, len(segment_list)):
				if segment_list[i][0][1] == segment_list[j][0][0]:
					segment_list[i+1], segment_list[j] = segment_list[j], segment_list[i+1]
					break

		# Job done
		return segment_list

	# Job done
	return { i : order_segment_list(segment_list) for i, segment_list in voronoi_cell_map.items() }
