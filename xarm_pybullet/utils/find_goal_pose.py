import torch
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.xarm_model import XArmFK
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF

def find_goal_configuration(
    goal_pos: torch.Tensor,
    points_robot: torch.Tensor,
    robot_fk: Optional[XArmFK] = None,
    robot_sdf: Optional[RobotSDF] = None,
    robot_cdf: Optional[RobotCDF] = None,
    n_samples: int = 100000,
    threshold: float = 0.1,
    sdf_margin: float = 0.02,
    cdf_margin: float = 0.10,
    device: str = 'cuda',
    max_solutions: int = 10,
    seed: int = None
) -> List[Tuple[np.ndarray, float, float, float]]:
    """Find valid goal configurations for a given goal position using random sampling.
    
    Args:
        goal_pos: Target position tensor [3]
        points_robot: Point cloud of robot environment [N, 3]
        robot_fk: XArm forward kinematics model (will create new one if None)
        robot_sdf: Robot SDF model (will create new one if None)
        robot_cdf: Robot CDF model (will create new one if None)
        n_samples: Number of random samples to try
        threshold: Maximum acceptable distance to goal position
        sdf_margin: Minimum acceptable SDF value for safety
        cdf_margin: Minimum acceptable CDF value for safety
        device: Device to use for computations ('cuda' or 'cpu')
        max_solutions: Maximum number of valid configurations to return
        seed: Random seed for reproducibility
    
    Returns:
        List of tuples (config, distance, min_sdf, min_cdf) for all valid configurations found, sorted by distance
    """
    # Set random seed if provided
    if seed is not None:
        print(f"Setting goal finding random seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Initialize models if not provided
    if robot_fk is None:
        robot_fk = XArmFK(device=device)
    if robot_sdf is None:
        robot_sdf = RobotSDF(device=device)
    if robot_cdf is None:
        robot_cdf = RobotCDF(device=device)

    print(f"\nSearching for goal configurations (max {max_solutions}):")
    print(f"Target position: {goal_pos.cpu().numpy()}")
    
    # Get joint limits and ensure float32
    lower_limits = robot_fk.joint_limits[:, 0].cpu().numpy().astype(np.float32)
    upper_limits = robot_fk.joint_limits[:, 1].cpu().numpy().astype(np.float32)
    
    # Sample configurations in batches for efficiency
    batch_size = 1000
    n_batches = int(n_samples // batch_size)
    valid_solutions = []
    
    # Ensure point cloud is properly formatted
    points_robot = points_robot.to(dtype=torch.float32, device=device)
    while points_robot.dim() > 2:
        points_robot = points_robot.squeeze(0)
    
    for batch in range(n_batches):
        # Exit if we've found enough solutions
        if len(valid_solutions) >= max_solutions:
            break
            
        # Sample random configurations
        configs = np.random.uniform(
            low=lower_limits,
            high=upper_limits,
            size=(batch_size, len(lower_limits))
        ).astype(np.float32)
        
        configs_tensor = torch.tensor(configs, device=device, dtype=torch.float32)
        
        try:
            # Get end-effector positions
            ee_positions = robot_fk.fkine(configs_tensor)[:, -1]  # [batch_size, 3]
            
            # Compute distances to goal
            distances = torch.norm(ee_positions - goal_pos.unsqueeze(0), dim=1)
            
            # Find valid configurations
            valid_indices = torch.where(distances < threshold)[0]
            
            if len(valid_indices) > 0:
                # Check each valid configuration for collisions
                for idx in valid_indices:
                    config = configs[idx]
                    config_tensor = torch.tensor(config, device=device, dtype=torch.float32).unsqueeze(0)
                    
                    # Check SDF values
                    sdf_values = robot_sdf.query_sdf(
                        points=points_robot.unsqueeze(0),
                        joint_angles=config_tensor,
                        return_gradients=False
                    )
                    
                    # Check CDF values
                    cdf_values = robot_cdf.query_cdf(
                        points=points_robot.unsqueeze(0),
                        joint_angles=config_tensor,
                        return_gradients=False
                    )
                    
                    # Configuration is valid if both SDF and CDF margins are satisfied
                    if sdf_values.min() > sdf_margin and cdf_values.min() > cdf_margin:
                        current_dist = distances[idx].item()
                        min_sdf = sdf_values.min().item()
                        min_cdf = cdf_values.min().item()
                        
                        # Store this valid solution with safety margins
                        valid_solutions.append((config, current_dist, min_sdf, min_cdf))
                        print(f"Found valid configuration {len(valid_solutions)}/{max_solutions}:")
                        print(f"Distance to goal: {current_dist:.4f} meters")
                        print(f"Min SDF: {min_sdf:.4f}")
                        print(f"Min CDF: {min_cdf:.4f}")
                        
                        # Exit if we've found enough solutions
                        if len(valid_solutions) >= max_solutions:
                            break
                
        except Exception as e:
            print(f"Error in batch {batch}: {str(e)}")
            continue
            
        if (batch + 1) % 10 == 0:
            print(f"Processed {(batch + 1) * batch_size}/{n_samples} samples...")
            print(f"Found {len(valid_solutions)}/{max_solutions} valid configurations")
    
    if valid_solutions:
        # Sort solutions by distance
        valid_solutions.sort(key=lambda x: x[1])
        print(f"\nFound {len(valid_solutions)} valid configurations")
        print(f"Best distance: {valid_solutions[0][1]:.4f} meters")
        return valid_solutions
    else:
        print("Failed to find any valid goal configurations")
        return []

