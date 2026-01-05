import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from scipy.spatial import cKDTree
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlanningMetrics:
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float

class RRTBasePlanner:
    def __init__(self, 
                 robot_sdf,
                 robot_fk,
                 joint_limits: Tuple[np.ndarray, np.ndarray],
                 step_size: float = 0.1,
                 max_nodes: int = 1e6,
                 batch_size: int = 1000,
                 goal_bias: float = 0.1,
                 safety_margin: float = 0.02,
                 device: str = 'cuda'):
        self.robot_sdf = robot_sdf
        self.robot_fk = robot_fk
        self.joint_lower_limits = joint_limits[0]
        self.joint_upper_limits = joint_limits[1]
        self.step_size = step_size
        self.max_nodes = max_nodes
        self.batch_size = batch_size
        self.goal_bias = goal_bias
        self.safety_margin = safety_margin
        self.device = device
        self.collision_check_count = 0

    def _get_uniform_random_configs(self, n_samples: int, rng=None) -> np.ndarray:
        """Generate batch of random configurations"""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.joint_lower_limits, self.joint_upper_limits, size=(n_samples, len(self.joint_lower_limits)))

    def _check_collision_batch(self, configs: np.ndarray, obstacle_points: Optional[torch.Tensor] = None) -> np.ndarray:
        """Check collision for batch of configurations using RobotSDF"""
        self.collision_check_count += len(configs)  # Count each config in batch
        configs_tensor = torch.tensor(configs, device=self.device)
        
        # Get SDF values for all configurations
        if obstacle_points is not None:
            # Ensure obstacle_points is the right shape [N, 3]
            if obstacle_points.dim() == 4:
                obstacle_points = obstacle_points.squeeze(0)  # Remove extra dimensions
            
            # Query SDF values for each configuration
            sdf_values = self.robot_sdf.query_sdf(
                points=obstacle_points.expand(len(configs), -1, -1),  # [B, N, 3]
                joint_angles=configs_tensor,  # [B, 6]
                return_gradients=False
            )
            print(f"SDF values shape: {sdf_values.shape}")
            print(f"SDF min value: {sdf_values.min().item():.6f}")
            print(f"SDF max value: {sdf_values.max().item():.6f}")
            print(f"SDF mean value: {sdf_values.mean().item():.6f}")
            # Configuration is valid if minimum SDF value is above safety margin
            min_sdf_values = sdf_values.min(dim=1)[0]  # Get minimum across all points
            return min_sdf_values > self.safety_margin
        return torch.ones(len(configs), dtype=torch.bool, device=self.device)

    def _extend_towards(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Extend with adaptive step size based on distance to goal"""
        diff = to_config - from_config
        dist = np.linalg.norm(diff)
        
        # Make step size proportional to distance, but never larger than base step_size
        adaptive_step = min(self.step_size * (dist / 2.0), self.step_size)
        
        # Ensure minimum step size
        adaptive_step = max(adaptive_step, self.step_size * 0.1)
        
        if dist > adaptive_step:
            diff = diff / dist * adaptive_step
        return from_config + diff

    def _get_ee_position(self, config: np.ndarray) -> np.ndarray:
        """Get end-effector position for configuration"""
        config_tensor = torch.tensor(config, device=self.device, dtype=torch.float32)
        if len(config_tensor.shape) == 1:
            config_tensor = config_tensor.unsqueeze(0)
        ee_pos = self.robot_fk.fkine(config_tensor)
        return ee_pos.squeeze().cpu().numpy()

class RRTPlanner(RRTBasePlanner):
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray,
            obstacle_points: Optional[torch.Tensor] = None, 
            return_metrics: bool = False,
            rng: Optional[np.random.RandomState] = None) -> Union[List[np.ndarray], PlanningMetrics]:
        """Plan path using standard RRT
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            obstacle_points: Point cloud of obstacles
            return_metrics: Whether to return metrics instead of path
            rng: Random number generator for reproducibility
        """
        if rng is None:
            rng = np.random.default_rng()

        start_time = time.time()
        self.collision_check_count = 0  # Reset counter
        print("\nStarting RRT planning...")
        
        # Initialize tree with start configuration
        tree = [(start_config, -1)]
        tree_kdtree = cKDTree(np.array([start_config]))
        
        # Statistics tracking
        closest_dist_to_goal = float('inf')
        attempts = 0
        
        while len(tree) < self.max_nodes:
            attempts += 1
            n_nodes_before = len(tree)
            
            # Generate batch of samples
            goal_biased_count = int(self.batch_size * self.goal_bias)
            regular_count = self.batch_size - goal_biased_count
            
            # Generate random samples and add goal bias
            random_configs = self._get_uniform_random_configs(regular_count, rng)
            if goal_biased_count > 0:
                goal_biased_configs = np.tile(goal_config, (goal_biased_count, 1))
                random_configs = np.vstack([random_configs, goal_biased_configs])
            
            # Process batch
            distances, nearest_indices = tree_kdtree.query(random_configs, k=1)
            new_configs = np.array([
                self._extend_towards(tree[idx][0], sample)
                for sample, idx in zip(random_configs, nearest_indices)
            ])
            
            # Check collisions for all new configurations
            valid_mask = self._check_collision_batch(new_configs, obstacle_points)
            
            # Add valid configurations to tree
            for config, nearest_idx, is_valid in zip(new_configs, nearest_indices, valid_mask):
                if is_valid and len(tree) < self.max_nodes:
                    tree.append((config, nearest_idx))
                    
                    # Check if reached goal
                    dist_to_goal = np.linalg.norm(config - goal_config)
                    closest_dist_to_goal = min(closest_dist_to_goal, dist_to_goal)
                    
                    if dist_to_goal < self.step_size:  # Close enough to goal
                        path = self._extract_path(tree, len(tree)-1)
                        elapsed_time = time.time() - start_time
                        print(f"\nPath found!")
                        print(f"Attempts: {attempts}")
                        print(f"Time: {elapsed_time:.2f}s")
                        print(f"Tree size: {len(tree)} nodes")
                        print(f"Path length: {len(path)} waypoints")
                        if path:  # If planning succeeded
                            metrics = PlanningMetrics(
                                success=True,
                                num_collision_checks=self.collision_check_count,
                                path_length=np.sum([np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]),
                                num_samples=attempts * self.batch_size,
                                planning_time=time.time() - start_time
                            )
                            return metrics if return_metrics else path
                        return path
            
            # Update KD-tree
            tree_kdtree = cKDTree(np.array([node[0] for node in tree]))
            
            # Print progress
            if attempts % 2 == 0:
                n_nodes = len(tree)
                nodes_added = n_nodes - n_nodes_before
                print(f"Attempts: {attempts} | Nodes: {n_nodes} | Added: {nodes_added} | Best dist: {closest_dist_to_goal:.4f}")
        
        print(f"\nFailed to find path after reaching max nodes ({self.max_nodes})")
        print(f"Best distance to goal: {closest_dist_to_goal:.4f}")
        return []

    def _extract_path(self, tree: List[Tuple], goal_idx: int) -> List[np.ndarray]:
        """Extract path from tree from start to goal"""
        path = []
        current_idx = goal_idx
        while current_idx != -1:
            path.append(tree[current_idx][0])
            current_idx = tree[current_idx][1]
        return list(reversed(path)) 
    

class RRTConnectPlanner(RRTBasePlanner):
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray,
            obstacle_points: Optional[torch.Tensor] = None, rng=None) -> List[np.ndarray]:
        """Plan path using RRT-Connect algorithm"""
        if rng is None:
            rng = np.random.default_rng()

        start_time = time.time()
        print("\nStarting RRT-Connect planning...")
        
        # Initialize two trees (start_tree, goal_tree)
        start_tree = [(start_config, -1)]
        goal_tree = [(goal_config, -1)]
        start_kdtree = cKDTree(np.array([start_config]))
        goal_kdtree = cKDTree(np.array([goal_config]))
        
        attempts = 0
        
        while len(start_tree) + len(goal_tree) < self.max_nodes:
            attempts += 1
            
            # Alternate between trees
            if len(start_tree) <= len(goal_tree):
                source_tree, target_tree = start_tree, goal_tree
                source_kdtree, target_kdtree = start_kdtree, goal_kdtree
                growing_from_start = True
            else:
                source_tree, target_tree = goal_tree, start_tree
                source_kdtree, target_kdtree = goal_kdtree, start_kdtree
                growing_from_start = False
            
            # Generate random samples
            random_configs = self._get_uniform_random_configs(self.batch_size, rng)
            
            # Find nearest neighbors in source tree
            distances, nearest_indices = source_kdtree.query(random_configs, k=1)
            
            # Extend source tree
            new_configs = np.array([
                self._extend_towards(source_tree[idx][0], sample)
                for sample, idx in zip(random_configs, nearest_indices)
            ])
            
            # Check collisions
            valid_mask = self._check_collision_batch(new_configs, obstacle_points)
            
            # Try to connect trees
            for new_config, nearest_idx, is_valid in zip(new_configs, nearest_indices, valid_mask):
                if not is_valid:
                    continue
                    
                # Add to source tree
                source_tree.append((new_config, nearest_idx))
                new_source_idx = len(source_tree) - 1
                
                # Try to connect to target tree
                success, bridge_path = self._connect_trees(
                    new_config, 
                    target_tree,
                    target_kdtree,
                    obstacle_points
                )
                
                if success:
                    # Extract and combine paths
                    if growing_from_start:
                        start_path = self._extract_path(source_tree, new_source_idx)
                        goal_path = self._extract_path(target_tree, bridge_path[-1][1])
                        full_path = start_path + [node[0] for node in bridge_path[1:]]
                    else:
                        start_path = self._extract_path(target_tree, bridge_path[-1][1])
                        goal_path = self._extract_path(source_tree, new_source_idx)
                        full_path = start_path + [node[0] for node in reversed(bridge_path[:-1])]
                    
                    elapsed_time = time.time() - start_time
                    print(f"\nPath found!")
                    print(f"Attempts: {attempts}")
                    print(f"Time: {elapsed_time:.2f}s")
                    print(f"Tree sizes: {len(start_tree)}, {len(goal_tree)} nodes")
                    print(f"Path length: {len(full_path)} waypoints")
                    return full_path
            
            # Update KD-trees
            start_kdtree = cKDTree(np.array([node[0] for node in start_tree]))
            goal_kdtree = cKDTree(np.array([node[0] for node in goal_tree]))
            
            # Print progress
            if attempts % 2 == 0:
                print(f"Attempts: {attempts} | Nodes: {len(start_tree) + len(goal_tree)}")
        
        print(f"\nFailed to find path after reaching max nodes ({self.max_nodes})")
        return []

    def _connect_trees(self, config: np.ndarray, target_tree: List[Tuple], 
                      target_kdtree: cKDTree, obstacle_points: Optional[torch.Tensor]) -> Tuple[bool, List[Tuple]]:
        """Try to connect to target tree using greedy extension"""
        bridge_path = []
        current_config = config
        
        while True:
            # Find nearest node in target tree
            distances, nearest_idx = target_kdtree.query([current_config], k=1)
            target_config = target_tree[nearest_idx[0]][0]
            
            # Extend towards target
            new_config = self._extend_towards(current_config, target_config)
            
            # Check if extension is valid
            if not self._check_collision_batch(new_config.reshape(1, -1), obstacle_points)[0]:
                return False, bridge_path
            
            # Add to bridge path
            bridge_path.append((new_config, nearest_idx[0]))
            
            # Check if reached target tree
            if np.linalg.norm(new_config - target_config) < self.step_size:
                return True, bridge_path
                
            current_config = new_config

    def _extract_path(self, tree: List[Tuple], node_idx: int) -> List[np.ndarray]:
        """Extract path from tree by following parent indices back to root"""
        path = []
        current_idx = node_idx
        
        while current_idx != -1:  # -1 is the parent index of root node
            path.append(tree[current_idx][0])  # Add configuration
            current_idx = tree[current_idx][1]  # Move to parent
            
        return list(reversed(path))  # Reverse to get start-to-goal order


