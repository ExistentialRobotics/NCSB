import pybullet as p
import numpy as np
import torch
import cvxpy
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) 


from sdf_marching.samplers import get_rapidly_exploring, get_uniform_random, get_rapidly_exploring_connect
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.samplers.tracing import trace_toward_graph_all
from sdf_marching.discrete import get_shortest_path
from sdf_marching.cvx import edgeseq_to_traj_constraint_bezier, bezier_cost_all
import time

@dataclass
class Bubble:
    """Represents a configuration space bubble"""
    center: np.ndarray  # Configuration at bubble center
    radius: float       # Bubble radius in configuration space

@dataclass
class PlanningMetrics:
    """Represents planning metrics"""
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float
    reached_goal_index: int = 0  # Add default value of 0

def create_multigoal_sampler(mins, maxs, goal_configs, p0=0.2, rng=None):
    """
    Creates a sampler that:
    - With probability p0, chooses one of the G goals (each with p0/G probability)
    - With probability 1-p0, returns uniform random sample
    
    Args:
        mins: Lower bounds for sampling
        maxs: Upper bounds for sampling
        goal_configs: List of goal configurations
        p0: Probability of sampling from goals (default: 0.2)
        rng: Random number generator (optional)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Ensure all configs have same shape as mins/maxs
    mins = np.asarray(mins)
    maxs = np.asarray(maxs)
    goal_configs = [np.asarray(g).reshape(mins.shape) for g in goal_configs]
    
    def sample_fn(batch_size=100):
        # Pre-allocate array with correct shape
        samples = np.zeros((batch_size, mins.shape[0]), dtype=np.float32)
        
        for i in range(batch_size):
            if rng.random() < p0:
                # Choose one of the G goals randomly
                goal_idx = rng.integers(0, len(goal_configs))
                samples[i] = goal_configs[goal_idx]
            else:
                # Uniform random sampling
                samples[i] = rng.uniform(mins, maxs)
        
        return samples[0] if batch_size == 1 else samples
    
    return sample_fn

class BubblePlanner:
    def __init__(self, robot_cdf, joint_limits, self_collision_cdf=None, max_samples=5E4, batch_size=50, 
                 device='cuda', seed=42, planner_type='bubble', early_termination=True, safety_margin=0.1):
        """
        Initialize the bubble planner using CDF for collision checking
        
        Args:
            robot_cdf: Workspace CDF for robot-environment collisions
            self_collision_cdf: Optional self-collision CDF (default: None)
            joint_limits: Joint limits for the robot
            max_samples: Maximum number of samples to generate (default: 5E4)
            batch_size: Batch size for sampling (default: 50)
            device: Device to use for computation (default: 'cuda')
            seed: Random seed for reproducibility
            planner_type: Planner type ('bubble' or 'bubble_connect')
            early_termination: Whether to terminate early if a goal is reached
            safety_margin: Safety margin for collision checking
        """
        self.random_seed = seed
        np.random.seed(seed)

        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create local RNG
        self.rng = np.random.default_rng(seed)
        self.safety_margin = safety_margin
        
        # Store robot info
        self.robot_cdf = robot_cdf
        self.self_collision_cdf = self_collision_cdf  # Can be None
        self.joint_limits = (
            np.asarray(joint_limits[0], dtype=np.float32),
            np.asarray(joint_limits[1], dtype=np.float32)
        )
        self.device = device

        if joint_limits[0].shape[0] == 2:
            self.epsilon = 1E-1
            self.min_radius = 5E-2
        else:
            self.epsilon = 1E-1
            self.min_radius = 1E-1
        
        # Planning parameters
        self.num_samples = max_samples               
        self.max_iterations = 5E4
        self.goal_bias = 0.1
        self.cdf_query_count = 0
        self.batch_size = batch_size
        self.early_termination = early_termination
        
        # Planner type ('rrt' or 'rrt_connect')
        assert planner_type in ['bubble', 'bubble_connect'], "planner_type must be 'bubble' or 'bubble_connect'"
        self.planner_type = planner_type

    def query_cdf(self, config: np.ndarray, obstacle_points: torch.Tensor) -> float:
        """
        Query CDF value - workspace CDF and self-collision CDF (if available)
        """
        self.cdf_query_count += 1

        # Query workspace CDF
        config_tensor = torch.tensor(config, device=self.device, dtype=torch.float32)
        if config_tensor.dim() == 1:
            config_tensor = config_tensor.unsqueeze(0)
        
        if obstacle_points.dim() == 2:
            obstacle_points = obstacle_points.unsqueeze(0)
        
        workspace_cdf = self.robot_cdf.query_cdf(
            points=obstacle_points,
            joint_angles=config_tensor,
            return_gradients=False
        ).min().detach().cpu().numpy()

        # Query self-collision CDF if available
        if self.self_collision_cdf is not None:
            self_collision_cdf = self.self_collision_cdf.query_cdf(
                config_tensor,
                return_gradients=False
            ).min().detach().cpu().numpy()
            
            # Take minimum of both CDFs
            min_cdf = min(workspace_cdf - self.safety_margin, self_collision_cdf)
        else:
            # Only use workspace CDF if self-collision CDF is not available
            min_cdf = workspace_cdf - self.safety_margin

        # Ensure minimum radius for numerical stability
        min_cdf = max(min_cdf, 0.05)
        
        return min_cdf

    def get_bubble(self, config: np.ndarray, points: torch.Tensor) -> Bubble:
        """Generate a bubble at the given configuration using CDF with radius capping"""
        # Query CDF to get distance to obstacles
        distance = self.query_cdf(config, points)
        
        # Cap the radius to ensure we only use reliable CDF values
        radius = max(distance - self.epsilon, self.min_radius)
        
        return Bubble(center=config, radius=radius)

    def bubbles_connect(self, b1: Bubble, b2: Bubble) -> bool:
        """Check if two bubbles can be connected"""
        # Check if bubbles overlap
        center_dist = np.linalg.norm(b1.center - b2.center)
        return center_dist <= (b1.radius + b2.radius + self.connection_threshold)

    def interpolate_path(self, b1: Bubble, b2: Bubble, num_points: int = 10) -> np.ndarray:
        """Generate interpolated path between two connected bubbles"""
        return np.linspace(b1.center, b2.center, num_points)

    def generate_bubbles(self, start_config: np.ndarray, goal_configs: List[np.ndarray], obstacle_points: torch.Tensor):
        """Generate bubbles using either RRT or RRT-Connect with multiple goals"""
        print("Starting bubble generation...")
        try:

            
            # Convert configs to float32 and ensure consistent shapes
            start_config = np.asarray(start_config, dtype=np.float32)
            goal_configs = [np.asarray(goal, dtype=np.float32) for goal in goal_configs]
            
            
            # Ensure all configs have same dimensionality
            assert all(goal.shape == start_config.shape for goal in goal_configs), \
                f"All configurations must have the same shape. Start: {start_config.shape}, Goals: {[g.shape for g in goal_configs]}"
            
            # Wrap CDF query for compatibility
            def cdf_wrapper(x):
                if isinstance(x, np.ndarray) and len(x.shape) > 1:
                    return np.array([self.query_cdf(xi, obstacle_points) for xi in x])
                return self.query_cdf(x, obstacle_points)
            
            
            if self.planner_type == 'bubble_connect':
                # Use RRT-Connect with multiple goals
                overlaps_graph, max_circles, _ = get_rapidly_exploring_connect(
                    cdf_wrapper,
                    self.epsilon,
                    self.min_radius,
                    int(self.num_samples),
                    self.joint_limits[0],
                    self.joint_limits[1],
                    start_point=start_config,
                    batch_size=self.batch_size,
                    max_num_iterations=int(self.max_iterations),
                    prc=self.goal_bias,
                    end_point=goal_configs,
                    rng=self.rng,
                    profile=False,
                    early_termination=self.early_termination
                )
            else:
                print("Using Bubble-RRT planner...")
                # Create multi-goal sampler for RRT
                sampler = create_multigoal_sampler(
                    self.joint_limits[0],  # mins
                    self.joint_limits[1],  # maxs
                    goal_configs,
                    p0=self.goal_bias,
                    rng=self.rng
                )
                
                # Use standard RRT with custom sampler
                overlaps_graph, max_circles, _ = get_rapidly_exploring(
                    cdf_wrapper,
                    self.epsilon,
                    self.min_radius,
                    int(self.num_samples),
                    self.joint_limits[0],
                    self.joint_limits[1],
                    start_point=start_config,
                    end_point=goal_configs,
                    batch_size=self.batch_size,                       # for 2D, batch_size small (e.g.: 2); for xArm, batch_size large (e.g.: 100)
                    max_num_iterations=int(self.max_iterations),
                    sample_fn=sampler,  # Use our custom sampler
                    rng=self.rng,
                    profile=False,
                    early_termination=self.early_termination,
                    all_goals_reached_check=True
                )
            
            print(f"Bubble generation complete. Number of bubbles: {len(overlaps_graph.vs)}")
            
            # Add hausdorff_distance attribute to edges if not present
            for edge in overlaps_graph.es:
                if "hausdorff_distance" not in edge.attributes():
                    from_circle = overlaps_graph.vs[edge.source]["circle"]
                    to_circle = overlaps_graph.vs[edge.target]["circle"]
                    edge["hausdorff_distance"] = from_circle.hausdorff_distance_to(to_circle)
            
            return overlaps_graph, max_circles
            
        except Exception as e:
            print(f"Bubble generation failed with error: {str(e)}")
            raise e

    def find_path(self, bubbles: List[Bubble]) -> np.ndarray:
        """Find path through connected bubbles using simple waypoint interpolation"""
        path = []
        for i in range(len(bubbles) - 1):
            segment = self.interpolate_path(bubbles[i], bubbles[i+1])
            path.extend(segment)
        return np.array(path)

    def plan(self, start_config: np.ndarray, goal_configs: List[np.ndarray], obstacle_points: torch.Tensor):
        """Plan a path from start to multiple goal configurations"""
        print("\nStarting bubble-based planning...")
        # print(f"Input shapes - Start: {start_config.shape}, Goals: {[g.shape for g in goal_configs]}, Obstacles: {obstacle_points.shape}")
        
        total_start_time = time.time()  # Move timer to start of everything
        self.cdf_query_count = 0
        
        try:
            # goal_configs is a list containing a single list of configurations
            # We need to extract the inner list
            if isinstance(goal_configs, list) and len(goal_configs) == 1 and isinstance(goal_configs[0], list):
                goal_configs = goal_configs[0]
            
            # Generate bubbles with obstacle points
            overlaps_graph, max_circles = self.generate_bubbles(start_config, goal_configs, obstacle_points)

            # print('cdf_query_count_in generating bubbles', self.cdf_query_count)
            
            # Find start index and try to connect if needed
            start_idx = position_to_max_circle_idx(overlaps_graph, start_config)
            print(f"Start index: {start_idx}")
            if start_idx < 0:
                print("Repairing graph for start")
                overlaps_graph, start_idx = trace_toward_graph_all(
                    overlaps_graph, 
                    lambda x: self.query_cdf(x, obstacle_points),
                    self.epsilon, 
                    self.min_radius, 
                    start_config
                )

            # Try to connect each goal and store successful connections
            goal_connections = []
            for goal_idx, goal_config in enumerate(goal_configs):
                #print(f"Processing goal {goal_idx}: {goal_config}")
                end_idx = position_to_max_circle_idx(overlaps_graph, goal_config)
                # print(f"Goal {goal_idx} index: {end_idx}")
                if end_idx < 0:
                    # print(f"Attempting to repair graph for goal {goal_idx}")
                    try:
                        overlaps_graph, end_idx = trace_toward_graph_all(
                            overlaps_graph, 
                            lambda x: self.query_cdf(x, obstacle_points),
                            self.epsilon, 
                            self.min_radius, 
                            goal_config
                        )
                        goal_connections.append((end_idx, goal_config))
                        # print(f"Successfully connected goal {goal_idx}")
                    except Exception as e:
                        print(f"Failed to connect goal {goal_idx}: {e}")
                else:
                    goal_connections.append((end_idx, goal_config))
                    print(f"Goal {goal_idx} already connected")

            if not goal_connections:
                raise Exception("No goals could be connected to the graph")


            # Find best path among all connected goals
            overlaps_graph.to_directed()
            best_path = None
            best_cost = float('inf')
            best_goal_config = None
            best_goal_index = 0  # Add this to track which goal was reached

            for goal_idx, (end_idx, goal_config) in enumerate(goal_connections):
                try:
                    # print(f"Trying path to goal {goal_idx} at index {end_idx}")
                    path_result = get_shortest_path(
                        lambda from_circle, to_circle: from_circle.single_sided_hausdorff_distance_to(to_circle),
                        overlaps_graph,
                        start_idx,
                        end_idx,
                        cost_name="hausdorff_distance",
                        return_epath=True,
                    )
                    
                    if path_result is not None:
                        # print(f"Found path: {path_result}")
                        path_cost = sum(overlaps_graph.es[e]["hausdorff_distance"] for e in path_result[0])
                        print(f"Path cost: {path_cost}")
                        if path_cost < best_cost:
                            best_cost = path_cost
                            best_path = path_result
                            best_goal_config = goal_config
                            best_goal_index = goal_idx  # Store the index of the best goal
                except Exception as e:
                    print(f"Error finding path to goal {goal_idx}: {e}")
                    continue
            
            if best_path is None:
                raise Exception("No valid path found to any goal")
            
            print(f"Found best path: {best_path}")
            print(f"Best cost: {best_cost}")
            
            try:
                # Optimize trajectory for best path
                bps, constr_bps = edgeseq_to_traj_constraint_bezier(
                    overlaps_graph.es[best_path[0]], 
                    start_config, 
                    best_goal_config
                )
                
                cost = bezier_cost_all(bps, weights=[0.1])     # weights=[0.1] for minimizing path length, [1.0, 0.1] for smoother
                prob = cvxpy.Problem(cvxpy.Minimize(cost), constr_bps)
                
                # Print problem details
                # print(f"Number of variables: {sum(v.size for v in prob.variables())}")
                # print(f"Number of constraints: {len(prob.constraints)}")
                
                solve_start = time.time()
                prob.solve()
                # print(f"CVXPY post optimization Solve time: {time.time() - solve_start:.3f}s")
                
                # Generate final trajectory
                times = np.linspace(0, 1.0, 50)
                trajectory = np.vstack([bp.query(times).value for bp in bps])
                
                #print(f"Planning complete! Generated trajectory with {len(trajectory)} waypoints")
                
                # Include all operations in total planning time
                total_planning_time = time.time() - total_start_time
                
                metrics = PlanningMetrics(
                    success=True,
                    num_collision_checks=self.cdf_query_count,
                    path_length=np.sum([np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                                      for i in range(len(trajectory)-1)]),
                    num_samples=len(overlaps_graph.vs),
                    planning_time=total_planning_time,  # Use total time including CDF queries and optimization
                    reached_goal_index=best_goal_index
                )
                
                return {
                    'waypoints': trajectory,
                    'bezier_curves': bps,
                    'times': times,
                    'bubbles': overlaps_graph,
                    'metrics': metrics,
                    'selected_goal': best_goal_config
                }
            except Exception as e:
                print(f"Error in trajectory generation: {e}")
                raise e
            
        except Exception as e:
            print(f"Planning failed: {str(e)}")
            return None

def main():
    """Test function"""
    print("Bubble planner module. Import and use with XArm visualizer.")

if __name__ == "__main__":
    main() 