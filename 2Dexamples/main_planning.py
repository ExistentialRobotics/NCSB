import numpy as np
from typing import List
from utils_env import create_obstacles
from utils_visualization import visualize_results, visualize_cdf_bubble_planning, visualize_ompl_rrt_planning
import torch
from robot_cdf import RobotCDF
from robot_sdf import RobotSDF
from utils_new import inverse_kinematics_analytical

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from planner.bubble_planner import BubblePlanner, PlanningMetrics
from planner.rrt_ompl import OMPLRRTPlanner

import cProfile
import pstats
from pstats import SortKey

src_dir = os.path.dirname(os.path.abspath(__file__))


def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
    obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
    np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)


def plan_and_visualize(robot_cdf, robot_sdf, obstacles, initial_config, goal_configs, 
                      max_bubble_samples=500, seed=42, early_termination=False, 
                      planner_type='bubble', visualize=True, safety_margin=0.1,
                      use_profile=False):
    """
    Plan and visualize a path using different planners
    
    Args:
        planner_type: str, one of ['bubble', 'bubble_connect', 'cdf_rrt', 'sdf_rrt']
        safety_margin: float, safety margin for planning (default: 0.1)
        use_profile: bool, whether to use cProfile for performance analysis (default: False)
    """
    print(f"\nStarting planning process with planner: {planner_type}...")
    print(f"Using safety margin: {safety_margin}")
    
    # Create profiler only if use_profile is True
    pr = None
    if use_profile:
        pr = cProfile.Profile()
        pr.enable()
    
    try:
        # Set random seed for reproducibility
        joint_limits = (
            np.full_like(initial_config, -np.pi),  # lower bounds
            np.full_like(initial_config, np.pi)    # upper bounds
        )
        
        # Get obstacle points
        obstacle_points = torch.tensor(concatenate_obstacle_list(obstacles), 
                                     device=robot_cdf.device)
        print(f"Obstacle points shape: {obstacle_points.shape}")
        
        if planner_type in ['bubble', 'bubble_connect']:
            # Use bubble planner
            planner = BubblePlanner(
                robot_cdf, joint_limits, max_samples=max_bubble_samples, batch_size=5,
                device=robot_cdf.device, seed=seed, planner_type=planner_type, 
                early_termination=early_termination,
                safety_margin=safety_margin
            )
            result = planner.plan(initial_config, goal_configs, obstacle_points)
            
        elif planner_type in ['cdf_rrt', 'sdf_rrt', 'lazy_rrt', 'rrt_connect', 'informed_rrt', 'bit_star', 'rrt_star', 'bit']:
            # Use OMPL RRT planner
            planner = OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                planner_type=planner_type,
                device=robot_cdf.device,
                seed=seed,
                safety_margin=safety_margin
            )
            
            result = planner.plan(
                start_config=initial_config,
                goal_configs=goal_configs,
                obstacle_points=obstacle_points,
                max_time=10.0,
                early_termination=early_termination
            )
        
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        
        if result is None:
            print("Planning failed!")
            return None
        
        # Extract trajectory and metrics
        trajectory = result['waypoints']
        metrics = result['metrics']
        
        print(f"\nPlanning metrics:")
        print(f"Success: {metrics.success}")
        print(f"Planning time: {metrics.planning_time:.2f} seconds")
        print(f"Number of samples: {metrics.num_samples}")
        print(f"Path length: {metrics.path_length:.2f}")
        print(f"Number of collision checks: {metrics.num_collision_checks}")
        if hasattr(metrics, 'reached_goal_index'):
            print(f"Reached goal index: {metrics.reached_goal_index}")
        
        # Create visualization
        if visualize:
            if planner_type in ['bubble', 'bubble_connect']:
                visualize_results(obstacles, initial_config, goal_configs, 
                             trajectory, src_dir=src_dir)
            
                visualize_cdf_bubble_planning(robot_cdf, initial_config, goal_configs, 
                                        trajectory, result['bubbles'], 
                                        obstacle_points, src_dir, planner_type=planner_type)
            else:  # cdf_rrt or sdf_rrt
                visualize_results(obstacles, initial_config, goal_configs, 
                             trajectory, src_dir=src_dir)
            
                visualize_ompl_rrt_planning(robot_cdf, robot_sdf, initial_config, goal_configs, 
                                      trajectory, obstacle_points, src_dir, 
                                      planner_type=planner_type)
        
        return result
    finally:
        # Print profiling results only if profiler was used
        if use_profile and pr is not None:
            pr.disable()
            print(f"\nProfiling Results for {planner_type} planner:")
            stats = pstats.Stats(pr)
            stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)
            stats.dump_stats(f"profile_{planner_type}.stats")

if __name__ == "__main__":
    # Set all random seeds
    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create obstacles
    obstacles = create_obstacles(rng=rng)

    # Initialize RobotCDF and RobotSDF classes
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    # Make sure the figures directory exists
    os.makedirs(os.path.join(src_dir, 'figures'), exist_ok=True)

    # Set initial configuration and goal position
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_pos = np.array([-2.5, 2.5], dtype=np.float32)     # Single goal position for end-effector

    # find multiple goal configurations by inverse kinematics
    goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])

    # Test different planners with profiling
    planners_to_test = ['bubble']  # Add any planners you want to compare
    
    for planner in planners_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {planner} planner")
        print(f"{'='*50}")
        
        result = plan_and_visualize(
            robot_cdf, robot_sdf, obstacles, initial_config, goal_configs, 
            max_bubble_samples=150, seed=seed, early_termination=True, 
            planner_type=planner, visualize=True, use_profile=False
        )





