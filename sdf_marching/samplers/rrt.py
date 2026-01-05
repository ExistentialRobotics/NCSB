import numpy as np
from scipy.spatial import cKDTree
from sdf_marching.samplers.uniform import get_uniform_random_points
import cProfile
import pstats
import time
import logging
from functools import wraps
from sdf_marching.overlap import get_overlaps_graph_numpy 
from sdf_marching.circles import Circle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def timing_decorator(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         return result
#     return wrapper

def inflate_min_and_max(mins, maxs, factor):
    diff = maxs - mins
    centre = (maxs + mins) / 2
    return centre - factor * diff / 2, centre + factor * diff / 2

def point_in_circle(point, center, radius):
    return np.linalg.norm(point - center) <= radius

def get_valid_circles_batch(centers, radii, dist_function, new_positions, near_indices, epsilon, minimum_radius):
    # Handle empty inputs
    if len(new_positions) == 0:
        return [], []
        
    # Ensure inputs are numpy arrays and at least 2D
    new_positions = np.asarray(new_positions)
    if new_positions.ndim == 1:
        new_positions = new_positions.reshape(1, -1)
    
    near_indices = np.asarray(near_indices)
    if near_indices.ndim == 0:
        near_indices = near_indices.reshape(1)
        
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    
    # Handle batch of points together
    diff_vectors = new_positions - centers[near_indices]
    norms = np.linalg.norm(diff_vectors, axis=1)
    
    # Filter out zero-norm vectors
    valid_mask = norms > 0
    if not np.any(valid_mask):
        return [], []
    
    # Compute new centers for all valid points
    scale_factors = (radii[near_indices[valid_mask]] / norms[valid_mask])[:, np.newaxis]
    new_centers = centers[near_indices[valid_mask]] + scale_factors * diff_vectors[valid_mask]
    
    # Batch distance computation
    new_radii = dist_function(new_centers) - epsilon
    
    # Filter by minimum radius
    valid_radii_mask = new_radii > minimum_radius
    if not np.any(valid_radii_mask):
        return [], []
    
    return new_centers[valid_radii_mask], new_radii[valid_radii_mask]

def pick_circle(sampler, centers, radii, max_retry=500, tree=None):
    if tree is None:
        tree = cKDTree(centers)
    
    for retry in range(max_retry):
        random_position = sampler()
        # Find nearest neighbor using KD-tree
        _, near_idx = tree.query(random_position.reshape(1, -1), k=1)
        near_idx = near_idx[0]
        
        # Check if point is outside the circle
        if not point_in_circle(random_position, centers[near_idx], radii[near_idx]):
            return near_idx, random_position, tree
            
    return pick_circle(sampler, centers, radii, max_retry, tree)

def get_rapidly_exploring(
    dist_function,
    epsilon,
    minimum_radius,
    num_samples,
    mins,
    maxs,
    start_point,
    batch_size=100,
    max_num_iterations=np.inf,
    inflate_factor=1.0,
    prc=0.1,
    end_point=None,
    rng=None,
    profile=True,
    sample_fn=None,
    early_termination=False, 
    all_goals_reached_check = True
):
    if rng is None:
        rng = np.random.default_rng()

    if profile:
        pr = cProfile.Profile()
        pr.enable()
        start_time = time.time()


    # Pre-allocate arrays
    max_circles = num_samples if np.isinf(max_num_iterations) else min(num_samples, int(max_num_iterations))
    centers = np.zeros((max_circles, len(mins)))
    radii = np.zeros(max_circles)
    
    # Initialize with start point
    centers[0] = start_point
    radii[0] = dist_function(start_point.reshape(1, -1)) - epsilon
    n_circles = 1
    
    # Initialize KD-tree with first circle
    tree = cKDTree(centers[:1])

    # Setup sampling space
    inflated_mins, inflated_maxs = inflate_min_and_max(mins, maxs, inflate_factor)
    
    num_iterations = 0
    while n_circles < num_samples and num_iterations < max_num_iterations:
        # Check if we've reached any goal at the start of each iteration
        if end_point is not None:
            all_goals_reached = True
            for goal in end_point:
                goal_reshaped = np.array(goal).reshape(1, -1)
                _, near_idx = tree.query(goal_reshaped, k=1)
                if not point_in_circle(goal, centers[near_idx[0]], radii[near_idx[0]]):
                    all_goals_reached = False
                    break

            

            # terminate if all goals are reached (may result in suboptimal planned path length)
            if all_goals_reached and all_goals_reached_check:
                print("All goals reached! Terminating search.")
                overlaps_graph = get_overlaps_graph_numpy(centers[:n_circles], radii[:n_circles])
                return overlaps_graph, (centers[:n_circles], radii[:n_circles]), num_iterations

        # Use custom sampler if provided, otherwise use default sampling
        if sample_fn is not None:
            random_positions = sample_fn(batch_size)
            # Ensure random_positions is 2D array
            if isinstance(random_positions, np.ndarray) and random_positions.ndim == 1:
                random_positions = random_positions.reshape(1, -1)
        else:
            # Original sampling logic
            goal_biased_count = int(batch_size * prc) if end_point is not None else 0
            regular_count = batch_size - goal_biased_count
            
            # Ensure we get at least one sample
            regular_count = max(1, regular_count)
            
            random_positions = get_uniform_random_points(regular_count, inflated_mins, inflated_maxs, rng=rng)
            # Ensure random_positions is 2D array
            if random_positions.ndim == 1:
                random_positions = random_positions.reshape(1, -1)
                
            if goal_biased_count > 0:
                if isinstance(end_point, list):
                    # Handle multiple goals
                    goal_positions = np.array([end_point[rng.integers(0, len(end_point))] 
                                            for _ in range(goal_biased_count)])
                else:
                    # Single goal
                    goal_positions = np.tile(end_point, (goal_biased_count, 1))
                random_positions = np.vstack([random_positions, goal_positions])

        # Find nearest neighbors for all points in batch
        distances, near_indices = tree.query(random_positions, k=1)
        
        # Filter points that are outside their nearest circles
        outside_mask = np.array([
            not point_in_circle(pos, centers[idx], radii[idx])
            for pos, idx in zip(random_positions, near_indices)
        ])
        
        if not np.any(outside_mask):
            num_iterations += 1
            continue
            
        # Process valid points in batch
        new_centers, new_radii = get_valid_circles_batch(
            centers[:n_circles],
            radii[:n_circles],
            dist_function,
            random_positions[outside_mask],
            near_indices[outside_mask],
            epsilon,
            minimum_radius
        )
        
        if len(new_centers) > 0:
            # Add valid new circles
            num_new = len(new_centers)
            end_idx = min(n_circles + num_new, max_circles)
            centers[n_circles:end_idx] = new_centers[:end_idx-n_circles]
            radii[n_circles:end_idx] = new_radii[:end_idx-n_circles]
            n_circles = end_idx
            
            # Update tree with new points
            tree = cKDTree(centers[:n_circles])

            # Check if we've reached any goal
            if early_termination and end_point is not None:
                for goal in end_point:
                    goal_reshaped = np.array(goal).reshape(1, -1)
                    _, near_idx = tree.query(goal_reshaped, k=1)
                    if point_in_circle(goal, centers[near_idx[0]], radii[near_idx[0]]):
                        print("Goal reached! Early termination.")
                        # Create overlaps graph before returning
                        overlaps_graph = get_overlaps_graph_numpy(centers[:n_circles], radii[:n_circles])
                        return overlaps_graph, (centers[:n_circles], radii[:n_circles]), num_iterations

            # Print progress
            # print(f"Circles: {n_circles} at iteration {num_iterations}")

        num_iterations += 1

    # Create overlaps graph before returning
    overlaps_graph = get_overlaps_graph_numpy(centers[:n_circles], radii[:n_circles])
    #overlaps_graph = get_overlaps_graph(circles)

    if profile:
        end_time = time.time()

        logger.info(f"\nPerformance Statistics:")
        logger.info(f"Total time: {end_time - start_time:.2f}s")
        logger.info(f"Circles created: {n_circles}")
        logger.info(f"Iterations: {num_iterations}")
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

    return overlaps_graph, (centers[:n_circles], radii[:n_circles]), num_iterations





def get_rapidly_exploring_connect(
    dist_function,
    epsilon,
    minimum_radius,
    num_samples,
    mins,
    maxs,
    start_point,
    end_point=None,
    batch_size=100,
    max_num_iterations=np.inf,
    inflate_factor=1.0,
    prc=0.1,
    rng=None,
    profile=True,
    early_termination=False,
    stagnation_threshold=20
):
    # Initialize num_iterations early for early returns
    num_iterations = 0
    
    # Modify end_point handling
    if end_point is not None:
        end_point = np.array(end_point)
        if len(end_point.shape) == 1:
            end_point = end_point.reshape(1, -1)  # Single point to 2D array
    else:
        logger.warning("No goal point provided. Falling back to single-tree RRT.")
        return None, None, num_iterations  # Return None to indicate failure

    if rng is None:
        rng = np.random.default_rng()

    if profile:
        pr = cProfile.Profile()
        pr.enable()
        start_time = time.time()

    # Initialize goal trees for each end point
    num_goals = len(end_point)
    num_trees = 1 + num_goals  # start tree + goal trees
    max_circles_per_tree = num_samples // num_trees  # Integer division
    actual_total_samples = max_circles_per_tree * num_trees  # This will be less than num_samples if not divisible
    
    print(f"Samples per tree: {max_circles_per_tree} (Total requested: {num_samples}, " +
          f"Actual total: {actual_total_samples})")

    # Arrays for all goal trees
    goal_centers = [np.zeros((max_circles_per_tree, len(mins))) for i in range(num_goals)]
    goal_radii = [np.zeros(max_circles_per_tree) for i in range(num_goals)]
    goal_trees = []
    n_goal_circles = [1] * num_goals
    
    # Initialize each goal tree
    for i in range(num_goals):
        goal_centers[i][0] = end_point[i]
        goal_radii[i][0] = dist_function(end_point[i].reshape(1, -1)) - epsilon
        goal_trees.append(cKDTree(goal_centers[i][:1]))

    # Track which goals have been connected
    goals_connected = [False] * num_goals
    
    # Pre-allocate arrays for both trees
    start_centers = np.zeros((max_circles_per_tree, len(mins)))
    start_radii = np.zeros(max_circles_per_tree)
    
    # Initialize both trees with their respective start points
    start_centers[0] = start_point
    start_radii[0] = dist_function(start_point.reshape(1, -1)) - epsilon
    n_start_circles = 1
    
    # Initialize KD-trees
    start_tree = cKDTree(start_centers[:1])

    # Setup sampling space
    inflated_mins, inflated_maxs = inflate_min_and_max(mins, maxs, inflate_factor)
    
    trees_connected = False
    
    # Add stagnation tracking
    last_progress = {
        'start': {'circles': 0, 'iterations': 0},
        'goals': [{
            'circles': 0,
            'iterations': 0
        } for _ in range(num_goals)]
    }

    def create_final_graph(start_centers, start_radii, goal_centers, goal_radii, goals_connected):
        """Create a graph containing only the start tree and connected goal trees"""
        # Start with the start tree
        all_centers = start_centers[:n_start_circles]
        all_radii = start_radii[:n_start_circles]
        
        # Only add connected goal trees
        for i in range(num_goals):
            if goals_connected[i]:  # Only include connected trees
                all_centers = np.vstack([all_centers, goal_centers[i][:n_goal_circles[i]]])
                all_radii = np.concatenate([all_radii, goal_radii[i][:n_goal_circles[i]]])
        
        if len(all_centers) == n_start_circles:  # No goals connected
            print("Warning: No goals connected to start tree!")
            return None, (None, None)
            
        # Create overlap graph only for connected components
        overlaps_graph = get_overlaps_graph_numpy(all_centers, all_radii)
        print(f"Created graph with {len(all_centers)} vertices from {sum(goals_connected)} connected trees")
        return overlaps_graph, (all_centers, all_radii)

    while num_iterations < max_num_iterations:
        # Add check at the start of each iteration
        if all(goals_connected):
            print("All goals connected! Terminating search.")
            overlaps_graph, (all_centers, all_radii) = create_final_graph(
                start_centers[:n_start_circles],
                start_radii[:n_start_circles],
                goal_centers,
                goal_radii,
                goals_connected
            )
            return overlaps_graph, (all_centers, all_radii), num_iterations

        # Determine which tree we're growing
        if num_iterations % (num_goals + 1) == 0:
            # Check if start tree is full
            if n_start_circles >= max_circles_per_tree:
                num_iterations += 1
                continue
        else:
            # Check if current goal tree is full
            current_goal_idx = (num_iterations % (num_goals + 1)) - 1
            if n_goal_circles[current_goal_idx] >= max_circles_per_tree:
                num_iterations += 1
                continue
            
        total_samples = n_start_circles + sum(n_goal_circles)
        if total_samples >= actual_total_samples:  # Check against actual total
            print(f"Max samples reached: {total_samples}/{actual_total_samples}")
            break
            
        # Calculate number of goal-biased samples
        goal_biased_count = int(batch_size * prc)
        regular_count = batch_size - goal_biased_count
        
        # Determine which tree we're growing
        if num_iterations % (num_goals + 1) == 0:
            # Grow start tree
            growing_centers = start_centers[:n_start_circles]
            growing_radii = start_radii[:n_start_circles]
            growing_tree = start_tree
            is_start_tree = True
            current_goal_idx = None
            
            # Bias towards closest unconnected goal
            unconnected_goals = [i for i, connected in enumerate(goals_connected) if not connected]
            if unconnected_goals:
                # Find closest unconnected goal
                distances = [np.min([np.linalg.norm(c - end_point[g]) for c in growing_centers]) 
                           for g in unconnected_goals]
                closest_goal_idx = unconnected_goals[np.argmin(distances)]
                target_point = end_point[closest_goal_idx]
            else:
                target_point = None
        else:
            # Grow one of the goal trees
            current_goal_idx = (num_iterations % (num_goals + 1)) - 1
            if goals_connected[current_goal_idx]:
                num_iterations += 1
                continue
                
            growing_centers = goal_centers[current_goal_idx][:n_goal_circles[current_goal_idx]]
            growing_radii = goal_radii[current_goal_idx][:n_goal_circles[current_goal_idx]]
            growing_tree = goal_trees[current_goal_idx]
            is_start_tree = False
            target_point = start_point  # Goal trees always bias toward start point

        # Generate samples
        random_positions = get_uniform_random_points(regular_count, 
                                                   inflated_mins, inflated_maxs, rng=rng)
        
        # Add biased samples
        if target_point is not None:
            # Generate random directions in the configuration space dimension
            dim = len(inflated_mins)
            random_directions = rng.uniform(-1, 1, size=(goal_biased_count, dim))
            norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
            random_directions /= norms
            distances = rng.random(goal_biased_count)[:, None]
            biased_positions = target_point + random_directions * distances
            random_positions = np.vstack([random_positions, biased_positions])

        # Check if new points are covered by ANY existing bubbles from ANY tree
        covered_mask = np.zeros(len(random_positions), dtype=bool)
        
        # Check against start tree
        for center, radius in zip(start_centers[:n_start_circles], start_radii[:n_start_circles]):
            covered_mask |= np.linalg.norm(random_positions - center, axis=1) <= radius
            
        # Check against all goal trees
        for goal_idx in range(num_goals):
            for center, radius in zip(goal_centers[goal_idx][:n_goal_circles[goal_idx]], 
                                    goal_radii[goal_idx][:n_goal_circles[goal_idx]]):
                covered_mask |= np.linalg.norm(random_positions - center, axis=1) <= radius

        random_positions = random_positions[~covered_mask]
        if len(random_positions) == 0:
            num_iterations += 1
            continue

        # Find nearest neighbors for remaining points in the CURRENT growing tree
        distances, near_indices = growing_tree.query(random_positions, k=1)
        
        # Filter points that are outside their nearest circles
        outside_mask = np.array([
            not point_in_circle(pos, growing_centers[idx], growing_radii[idx])
            for pos, idx in zip(random_positions, near_indices)
        ])
        
        if not np.any(outside_mask):
            num_iterations += 1
            continue
            
        # Process valid points in batch
        new_centers, new_radii = get_valid_circles_batch(
            growing_centers,
            growing_radii,
            dist_function,
            random_positions[outside_mask],
            near_indices[outside_mask],
            epsilon,
            minimum_radius
        )
        
        if len(new_centers) > 0:
            # Add valid new circles to appropriate tree
            num_new = len(new_centers)
            if is_start_tree:
                end_idx = min(n_start_circles + num_new, max_circles_per_tree)
                start_centers[n_start_circles:end_idx] = new_centers[:end_idx-n_start_circles]
                start_radii[n_start_circles:end_idx] = new_radii[:end_idx-n_start_circles]
                n_start_circles = end_idx
                start_tree = cKDTree(start_centers[:n_start_circles])
            else:
                end_idx = min(n_goal_circles[current_goal_idx] + num_new, max_circles_per_tree)
                goal_centers[current_goal_idx][n_goal_circles[current_goal_idx]:end_idx] = new_centers[:end_idx-n_goal_circles[current_goal_idx]]
                goal_radii[current_goal_idx][n_goal_circles[current_goal_idx]:end_idx] = new_radii[:end_idx-n_goal_circles[current_goal_idx]]
                n_goal_circles[current_goal_idx] = end_idx
                goal_trees[current_goal_idx] = cKDTree(goal_centers[current_goal_idx][:n_goal_circles[current_goal_idx]])

            # Check for connections
            for new_center, new_radius in zip(new_centers, new_radii):
                if is_start_tree:
                    # Check connection with goal trees
                    for goal_idx, connected in enumerate(goals_connected):
                        if connected:
                            continue
                        _, nearest_idx = goal_trees[goal_idx].query(new_center.reshape(1, -1), k=1)
                        nearest_idx = nearest_idx[0]
                        nearest_center = goal_centers[goal_idx][nearest_idx]
                        nearest_radius = goal_radii[goal_idx][nearest_idx]
                        
                        if point_in_circle(new_center, nearest_center, nearest_radius + new_radius):
                            goals_connected[goal_idx] = True
                            print(f"Connected to goal {goal_idx}!")
                            if early_termination:
                                overlaps_graph, (all_centers, all_radii) = create_final_graph(
                                    start_centers[:n_start_circles],
                                    start_radii[:n_start_circles],
                                    goal_centers,
                                    goal_radii,
                                    goals_connected
                                )
                                return overlaps_graph, (all_centers, all_radii), num_iterations
                else:
                    # Check connection with start tree
                    _, nearest_idx = start_tree.query(new_center.reshape(1, -1), k=1)
                    nearest_idx = nearest_idx[0]
                    if point_in_circle(new_center, start_centers[nearest_idx], 
                                     start_radii[nearest_idx] + new_radius):
                        goals_connected[current_goal_idx] = True
                        print(f"Connected to goal {current_goal_idx}!")
                        if early_termination:
                            overlaps_graph, (all_centers, all_radii) = create_final_graph(
                                start_centers[:n_start_circles],
                                start_radii[:n_start_circles],
                                goal_centers,
                                goal_radii,
                                goals_connected
                            )
                            return overlaps_graph, (all_centers, all_radii), num_iterations

            # print(f"Start circles: {n_start_circles}, " + 
            #       ", ".join([f"Goal {i} circles: {n}" for i, n in enumerate(n_goal_circles)]) +
            #       f", Iteration: {num_iterations}")

        # Check for stagnation before processing new iteration
        current_total_circles = n_start_circles + sum(n_goal_circles)
        
        # Update progress trackers
        if num_iterations % (num_goals + 1) == 0:
            # Start tree
            if n_start_circles > last_progress['start']['circles']:
                last_progress['start']['circles'] = n_start_circles
                last_progress['start']['iterations'] = num_iterations
        else:
            # Goal tree
            current_goal_idx = (num_iterations % (num_goals + 1)) - 1
            if n_goal_circles[current_goal_idx] > last_progress['goals'][current_goal_idx]['circles']:
                last_progress['goals'][current_goal_idx]['circles'] = n_goal_circles[current_goal_idx]
                last_progress['goals'][current_goal_idx]['iterations'] = num_iterations

        # Check if we're stuck (no progress in any tree for too long)
        iterations_since_progress = num_iterations - max(
            last_progress['start']['iterations'],
            max(goal['iterations'] for goal in last_progress['goals'])
        )
        
        if iterations_since_progress > stagnation_threshold:
            print(f"\nStagnation detected - No progress for {iterations_since_progress} iterations")
            print(f"Start tree: {n_start_circles} circles")
            for i, n in enumerate(n_goal_circles):
                print(f"Goal {i} tree: {n} circles (Connected: {goals_connected[i]})")
                
            if not any(goals_connected):
                print("No goals connected - returning None")
                return None, (None, None), num_iterations
            else:
                print(f"Some goals connected ({sum(goals_connected)}/{len(goals_connected)}) - creating partial graph")
                overlaps_graph, (all_centers, all_radii) = create_final_graph(
                    start_centers[:n_start_circles],
                    start_radii[:n_start_circles],
                    goal_centers,
                    goal_radii,
                    goals_connected
                )
                return overlaps_graph, (all_centers, all_radii), num_iterations

        num_iterations += 1

    # After loop ends, check why we terminated
    if not any(goals_connected):
        print(f"No goals connected:")
        print(f"Samples used: {n_start_circles + sum(n_goal_circles)}/{num_samples}")
        return None, (None, None), num_iterations
    
    # Create final graph with only connected components
    overlaps_graph, (all_centers, all_radii) = create_final_graph(
        start_centers[:n_start_circles],
        start_radii[:n_start_circles],
        goal_centers,
        goal_radii,
        goals_connected
    )

    if profile:
        end_time = time.time()
        logger.info(f"\nPerformance Statistics:")
        logger.info(f"Total time: {end_time - start_time:.2f}s")
        logger.info(f"Start tree circles: {n_start_circles}")
        logger.info(f"Goal trees circles: {[n for n in n_goal_circles]}")
        logger.info(f"Total circles: {n_start_circles + sum(n_goal_circles)}")
        logger.info(f"Iterations: {num_iterations}")
        logger.info(f"Goals connected: {sum(goals_connected)}/{len(goals_connected)}")
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

    return overlaps_graph, (all_centers, all_radii), num_iterations
