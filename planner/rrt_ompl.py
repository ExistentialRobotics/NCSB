try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    print("Warning: OMPL could not be imported. Some functionality may be limited.")
    

import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class PlanningMetrics:
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float
    reached_goal_index: int

def to_tensor(s, dof):
    return torch.tensor([s[i] for i in range(dof)], dtype=torch.float32)

class ValidityCheckerWrapper:
    def __init__(self, robot_sdf, robot_cdf, points_robot, dof=6, safety_margin=0.05, device='cuda'):
        self.robot_sdf = robot_sdf
        self.robot_cdf = robot_cdf
        self.points_robot = points_robot  # Should be shape [N, 3]
        self.device = device
        self.safety_margin = safety_margin
        self.dof = dof
        self.counter = 0
    
    def __call__(self, state) -> bool:
        """Check if configuration is collision-free"""
        self.counter += 1
        
        # Convert state to tensor [1, 6] (batch size 1)
        config = torch.tensor([state[i] for i in range(self.dof)], 
                            device=self.device, 
                            dtype=torch.float32).unsqueeze(0)
        
        # Reshape points to [1, N, 3] (batch size 1)
        points = self.points_robot.unsqueeze(0)  # Add batch dimension
        
        # Query SDF valuesm if use SDF field (generally slower due to kinematics chains)
        # distance_values = self.robot_sdf.query_sdf(
        #     points=points,  # Shape: [1, N, 3]
        #     joint_angles=config  # Shape: [1, num_links]
        # )
        
        # if planner type is cdf_rrt, use cdf values
        distance_values = self.robot_cdf.query_cdf(
            points=points,  # Shape: [1, N, 3]
            joint_angles=config  # Shape: [1, num_links]
        )
        
        is_valid = distance_values.min().item() > self.safety_margin
        #print(f"Checking config {config.cpu().numpy()[0]}, valid: {is_valid}")
        return is_valid
    
    def reset_count(self):
        self.counter = 0

class OMPLRRTPlanner:
    def __init__(self, 
                 robot_sdf,
                 robot_cdf,
                 robot_fk,
                 joint_limits: tuple,
                 planner_type: str = 'sdf_rrt',
                 check_resolution: float = 0.01,
                 device: str = 'cuda',
                 seed: int = None,
                 safety_margin: float = 0.05):
        """
        Initialize OMPL planner
        
        Args:
            robot_sdf: Robot SDF model
            robot_fk: Robot forward kinematics
            joint_limits: Tuple of (lower_limits, upper_limits)
            planner_type: 'cdf_rrt', 'sdf_rrt', 'lazy_rrt', 'informed_rrt', 
                         'bit_star', 'rrt_star', 
                         'rrt_connect'
            device: Computing device
            seed: Random seed for reproducibility
        """
        self.robot_sdf = robot_sdf
        self.robot_cdf = robot_cdf
        self.robot_fk = robot_fk
        self.device = device
        self.dof = len(joint_limits[0])

        self.safety_margin = safety_margin
        # Set random seed if provided
        if seed is not None:
            print(f"Setting RRT random seed: {seed}")
            ou.RNG.setSeed(seed)  # Set OMPL's random seed
        
        # Setup state space
        self.space = ob.RealVectorStateSpace(self.dof)
        bounds = ob.RealVectorBounds(self.dof)
        for i, (low, high) in enumerate(zip(joint_limits[0], joint_limits[1])):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.space.setBounds(bounds)
        
        # Setup space information
        self.si = ob.SpaceInformation(self.space)
        
        # Will be set during planning
        self.validity_checker = None
        self.planner_type = planner_type
        self.check_resolution = check_resolution
    def plan(self, 
             start_config: np.ndarray,
             goal_configs: List[np.ndarray],
             obstacle_points: torch.Tensor,
             max_time: float = 30.0,
             return_metrics: bool = True,
             early_termination: bool = True,
             optimization_time: float = None) -> Dict[str, Any]:
        """
        Plan a path using OMPL
        
        Args:
            early_termination: If True, return as soon as any valid path is found
                             If False, continue optimizing until max_time is reached
            optimization_time: For optimizing planners, how long to optimize after 
                             finding first solution. If None, uses max_time.
        """
        print("\nPlanning Debug Info:")
        print(f"Start config: {start_config}")
        print(f"Number of goals: {len(goal_configs)}")
        print(f"Early termination: {early_termination}")
        
        # Setup validity checker
        self.validity_checker = ValidityCheckerWrapper(self.robot_sdf, self.robot_cdf, obstacle_points, 
                                                       self.dof, self.safety_margin, self.device)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker))
          
        # 0.01 for 2 joints, 0.002 for 6 joints
        self.si.setStateValidityCheckingResolution(self.check_resolution)
        self.si.setup()
        
        all_results = []
        remaining_goals = list(range(len(goal_configs)))
        start_time = time.time()
        
        while remaining_goals and (time.time() - start_time) < max_time:
            # Create new problem definition
            pdef = ob.ProblemDefinition(self.si)
            
            # Set start state
            start = ob.State(self.space)
            for i in range(self.dof):
                start[i] = float(start_config[i])
            pdef.addStartState(start)
            
            # Create multi-goal for remaining goals
            mg = ob.GoalStates(self.si)
            for goal_idx in remaining_goals:
                goal = ob.State(self.space)
                for j in range(self.dof):
                    goal[j] = float(goal_configs[goal_idx][j])
                mg.addState(goal)
            
            mg.setThreshold(0.1)  # Allow solutions within this distance of goal
            
            pdef.setGoal(mg)
            
            # Create and setup planner based on type
            if self.planner_type in ['sdf_rrt', 'cdf_rrt']:
                planner = og.RRT(self.si)
                planner.setRange(0.1)
                planner.setGoalBias(0.1)
            elif self.planner_type == 'rrt_star':
                planner = og.RRTstar(self.si)
                planner.setRange(0.1)
                planner.setGoalBias(0.1)
            elif self.planner_type == 'lazy_rrt':
                planner = og.LazyRRT(self.si)
                planner.setRange(0.1)
                planner.setGoalBias(0.1)
            elif self.planner_type == 'informed_rrt':
                planner = og.InformedRRTstar(self.si)
                planner.setRange(0.1)
                planner.setGoalBias(0.1)
            elif self.planner_type == 'bit_star':
                planner = og.BITstar(self.si)
            elif self.planner_type == 'rrt_connect':
                planner = og.RRTConnect(self.si)
                planner.setRange(0.1)  # Maximum length of motion to be added to a tree
            else:
                raise ValueError(f"Unknown planner type: {self.planner_type}")

            planner.setProblemDefinition(pdef)
            planner.setup()
            
            # For optimizing planners, we might want to adjust the termination condition
            if not early_termination and self.planner_type in ['rrt_star', 'informed_rrt', 'bit_star']:
                # First find any solution
                first_solution = planner.solve(ob.timedPlannerTerminationCondition(max_time))
                
                if first_solution:
                    # If found, spend additional time optimizing
                    opt_time = optimization_time if optimization_time is not None else max_time
                    planner.solve(ob.timedPlannerTerminationCondition(opt_time))
                solved = first_solution
            else:
                # Stop as soon as any valid solution is found
                solved = planner.solve(ob.timedPlannerTerminationCondition(max_time))
            
            if solved:
                path = pdef.getSolutionPath()
                # Optimize path
                simplifier = og.PathSimplifier(self.si)
                simplifier.shortcutPath(path)
                simplifier.smoothBSpline(path)
                
                # Extract waypoints
                waypoints = []
                for state in path.getStates():
                    waypoint = [state[i] for i in range(self.dof)]
                    waypoints.append(waypoint)
                waypoints = np.array(waypoints)
                
                # Calculate path length
                path_length = sum(np.linalg.norm(waypoints[i+1] - waypoints[i]) 
                                for i in range(len(waypoints)-1))
                
                # Find which remaining goal was reached
                final_waypoint = waypoints[-1]
                reached_idx = None
                min_dist = float('inf')
                for goal_idx in remaining_goals:
                    dist = np.linalg.norm(final_waypoint - goal_configs[goal_idx])
                    if dist < min_dist:
                        min_dist = dist
                        reached_idx = goal_idx
                
                # Store result
                result = {
                    'waypoints': waypoints,
                    'reached_goal_index': reached_idx,
                    'path_length': path_length,
                    'metrics': PlanningMetrics(
                        success=True,
                        num_collision_checks=self.validity_checker.counter,
                        path_length=path_length,
                        num_samples=len(waypoints),
                        planning_time=time.time() - start_time,
                        reached_goal_index=reached_idx
                    )
                }
                all_results.append(result)
                
                # Remove reached goal from remaining goals
                if reached_idx is not None:
                    remaining_goals.remove(reached_idx)
                    
                # Early termination check
                if early_termination:
                    # Return the shortest path if we found multiple
                    if len(all_results) > 0:
                        best_result = min(all_results, key=lambda x: x['path_length'])
                        return {
                            'waypoints': best_result['waypoints'],
                            'metrics': best_result['metrics']
                        }
        
        # If we get here, either:
        # 1. We found all paths (remaining_goals is empty)
        # 2. We ran out of time
        # 3. We failed to find any paths
        
        if len(all_results) > 0:
            print(f"\nFound {len(all_results)} paths to goals")
            # Return all paths if early_termination=False, or best path if True
            if early_termination:
                best_result = min(all_results, key=lambda x: x['path_length'])
                #print(f"Early termination: Returning shortest path to goal {best_result['reached_goal_index']}")
                return {
                    'waypoints': best_result['waypoints'],
                    'metrics': best_result['metrics']
                }
            else:
                # Find the best path among all found paths
                best_result = min(all_results, key=lambda x: x['path_length'])
                #print(f"Found paths to goals: {[r['reached_goal_index'] for r in all_results]}")
                #print(f"Best path is to goal {best_result['reached_goal_index']} with length {best_result['path_length']:.3f}")
                return {
                    'waypoints': best_result['waypoints'],
                    'all_paths': all_results,
                    'found_all_goals': len(remaining_goals) == 0,
                    'metrics': PlanningMetrics(
                        success=True,
                        num_collision_checks=self.validity_checker.counter,
                        path_length=best_result['path_length'],  # Use shortest path length
                        num_samples=sum(len(r['waypoints']) for r in all_results),  # Total samples across all paths
                        planning_time=time.time() - start_time,
                        reached_goal_index=best_result['reached_goal_index']  # Index of shortest path goal
                    )
                }
        else:
            print("\nNo paths found to any goals")
            return {
                'waypoints': None,
                'all_paths': [],
                'found_all_goals': False,
                'metrics': PlanningMetrics(
                    success=False,
                    num_collision_checks=self.validity_checker.counter,
                    path_length=float('inf'),
                    num_samples=0,
                    planning_time=time.time() - start_time,
                    reached_goal_index=-1
                )
            }
