import numpy as np
import os
import sys
from pathlib import Path
from typing import List
import torch

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_dir = os.path.dirname(os.path.abspath(__file__))


from control.pd_cbf_qp import ClfCbfQpController
from control.pd_dro_cbf import ClfCbfDrccpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
from control.reference_governor_bezier import BezierReferenceGovernor
from main_planning import plan_and_visualize, concatenate_obstacle_list
from utils_env import create_obstacles, create_animation, create_dynamic_obstacles

from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from utils_new import inverse_kinematics_analytical
from utils_visualization import save_control_snapshots

def generate_random_goal(rng: np.random.Generator, robot_sdf: RobotSDF, obstacles: List[np.ndarray],
                        min_distance: float = 3.0, radius: float = 4.0, 
                        safety_threshold: float = 0.2) -> np.ndarray:
    """
    Generate a random goal position within a circle of given radius,
    maintaining minimum distance from start position and ensuring it's collision-free.
    """
    start_pos = np.array([0.0, 0.0])
    device = robot_sdf.device
    
    # Convert obstacles to tensor once
    obstacle_points = torch.tensor(np.concatenate(obstacles), 
                                 dtype=torch.float32, device=device)
    
    while True:
        # Generate random angle and radius
        theta = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, radius)
        
        # Convert to Cartesian coordinates
        goal_pos = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
        
        # Check distance from start
        if np.linalg.norm(goal_pos - start_pos) < min_distance:
            continue
            
        # Get IK solutions
        goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])
        
        # Convert goal_configs to tensor and reshape properly
        config_tensor = torch.tensor(goal_configs, dtype=torch.float32, device=device)
        if len(config_tensor.shape) == 1:
            config_tensor = config_tensor.unsqueeze(0)  # Add batch dimension if needed
            
        # Reshape obstacle points for batch processing
        obstacle_points_batch = obstacle_points.unsqueeze(0)  # [1, N, 2]
        
        # Check each IK solution
        for config in config_tensor:
            config_batch = config.unsqueeze(0)  # [1, 2]
            sdf_values = robot_sdf.query_sdf(obstacle_points_batch, config_batch)
            
            if torch.min(sdf_values) > safety_threshold:
                return goal_pos
            
    return None  # In case no valid goal is found (shouldn't happen in practice)



def track_planned_path(obstacles, trajectory_data, initial_config, dt=0.02, duration=20.0, 
                      control_type='clf_dro_cbf', use_bezier=True, dynamic_obstacles_exist=True):
    """
    Track the planned path using selected controller and reference governor.
    Dynamic obstacles are added during execution for control but not planning.
    
    Args:
        obstacles: List of obstacle arrays
        trajectory_data: Dictionary containing trajectory information
            - waypoints: Discrete waypoints
            - bezier_curves: Bezier curve segments (optional)
            - times: Time parameterization
        initial_config: Initial joint configuration
        dt: Time step
        duration: Maximum duration
        control_type: Controller type ('pd', 'clf_cbf', or 'clf_dro_cbf')
        use_bezier: Whether to use Bezier curves (True) or discrete waypoints (False)
        dynamic_obstacles_exist: Whether dynamic obstacles exist (True) or not (False)
    """
    # Convert static obstacles to tensor
    static_obstacle_points = torch.tensor(concatenate_obstacle_list(obstacles), dtype=torch.float32)
    
    # Initialize RobotCDF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cdf = RobotCDF(device)
    static_obstacle_points = static_obstacle_points.to(device)
    
    # Initialize RobotSDF for collision checking
    robot_sdf = RobotSDF(device)
    
    # Initialize appropriate controller based on type
    if control_type == 'pd':
        controller = PDController(kp=1.0, kd=0.2, control_limits=2.0)
    elif control_type == 'clf_cbf':

        pd_controller = PDController(kp=1.0, kd=0.2, control_limits=2.0)
        controller = ClfCbfQpController(
            p1=1.0,    
            p2=1e1,    # Increased penalty on CLF slack
            clf_rate=1.0,  # Increased CLF convergence rate
            cbf_rate=1.0,
            safety_margin=0.2,
            state_dim=2,
            control_limits=2.0
        )
    elif control_type == 'clf_dro_cbf':
        pd_controller = PDController(kp=0.8, kd=0.2, control_limits=2.0)
        controller = ClfCbfDrccpController(
            p1=1.0, 
            p2=1e1, 
            clf_rate=1.0, 
            cbf_rate=1.0, 
            wasserstein_r=0.016, 
            epsilon=0.1, 
            num_samples=10,
            state_dim=2,
            control_limits=2.0,
        )
    
    # Initialize appropriate governor based on mode
    if use_bezier and 'bezier_curves' in trajectory_data:
        print("Using Bezier-based reference governor")
        governor = BezierReferenceGovernor(
            initial_state=initial_config,
            trajectory_data=trajectory_data,
            dt=dt,
            k=0.2,  # Slower progression
            zeta=8   # Smoother progression
        )
    else:
        print("Using discrete waypoint-based reference governor")
        waypoints = (trajectory_data['waypoints'] if isinstance(trajectory_data, dict) 
                    else trajectory_data)
        governor = ReferenceGovernor(
            initial_state=initial_config,
            path_configs=waypoints,
            dt=dt
        )
    
    # Initialize simulation
    current_config = initial_config
    current_vel = np.zeros_like(initial_config)
    tracked_configs = [current_config]
    tracked_vels = [current_vel]
    reference_configs = [current_config]
    reference_vels = [np.zeros_like(initial_config)]
    time = 0
    
    # For collision checking
    is_safe = True
    safety_threshold = 0.01
    
    while time < duration:
        # Get current dynamic obstacles and their velocities
        dynamic_obstacles, dynamic_velocities = create_dynamic_obstacles(time)
        dynamic_points = torch.tensor(concatenate_obstacle_list(dynamic_obstacles), 
                                dtype=torch.float32, device=device)
        if dynamic_obstacles_exist:
            dynamic_vels = torch.tensor(np.concatenate(dynamic_velocities), 
                                  dtype=torch.float32, device=device)
        
            # Combine static and dynamic obstacle points
            all_obstacle_points = torch.cat([static_obstacle_points, dynamic_points], dim=0)
        else:
            all_obstacle_points = static_obstacle_points
            dynamic_vels = torch.zeros_like(dynamic_points)
        
        # Get reference from governor
        if use_bezier and 'bezier_curves' in trajectory_data:
            reference_config, s, reference_vel = governor.update(current_config)
        else:
            reference_config, s = governor.update(current_config)
            reference_vel = np.zeros_like(current_config)
            
        # Store reference
        reference_configs.append(reference_config)
        reference_vels.append(reference_vel)
        
        # Generate control based on controller type
        if control_type == 'pd':
            u = controller.compute_control(
                current_config,
                reference_config,
                current_vel
            )
        else:  # clf_cbf or clf_dro_cbf
            # Convert current config to appropriate shape for RobotCDF
            config_tensor = torch.tensor(current_config, dtype=torch.float32, device=device).unsqueeze(0)
            config_tensor.requires_grad_(True)
            
            # Get CDF values for all obstacles (static + dynamic)
            cdf_values = robot_cdf.query_cdf(
                points=all_obstacle_points.unsqueeze(0),
                joint_angles=config_tensor,
                return_gradients=False
            )

            u_nominal = pd_controller.compute_control(
                current_config,
                reference_config,
                current_vel
            )

            # print(f"u_nominal: {u_nominal}")
            
            if control_type == 'clf_cbf':
                # Get minimum CDF value and its gradient
                h = float(torch.min(cdf_values).item())
                min_idx = torch.argmin(cdf_values)
                
                # Compute gradient for minimum value
                h_val = cdf_values[0, min_idx]
                h_val.backward()
                dh_dtheta = config_tensor.grad.clone().cpu().numpy()[0]
                config_tensor.grad.zero_()
                
                # Compute dh/dt if the minimum point is from dynamic obstacles
                dh_dt = 0.0
                if min_idx >= len(static_obstacle_points):
                    # Get the point gradient
                    points = all_obstacle_points.unsqueeze(0)
                    points.requires_grad_(True)
                    cdf_val = robot_cdf.query_cdf(points, config_tensor, return_gradients=False)[0, min_idx]
                    cdf_val.backward()
                    dh_dp = points.grad[0, min_idx]  # gradient w.r.t point
                    
                    # Get velocity for this point
                    dyn_idx = min_idx - len(static_obstacle_points)
                    dp_dt = dynamic_vels[dyn_idx]
                    dh_dt = torch.dot(dh_dp, dp_dt).item()
                
                u = controller.generate_controller(
                    current_config,
                    reference_config,
                    h,
                    dh_dtheta,
                    dh_dt,
                    u_nominal
                )
            else:  # clf_dro_cbf
                # Get k smallest CDF values
                k = controller.num_samples
                h_values, indices = torch.topk(cdf_values[0], k, largest=False)
                
                # Compute gradients and dh/dt for k smallest values
                h_grads = []
                dh_dt_values = []
                
                for idx in indices:
                    # Compute theta gradient
                    h_val = cdf_values[0, idx]
                    h_val.backward(retain_graph=True)
                    h_grads.append(config_tensor.grad.clone().cpu().numpy()[0])
                    config_tensor.grad.zero_()
                    
                    # Compute dh/dt if point is from dynamic obstacles
                    if idx >= len(static_obstacle_points):
                        # Get the point gradient
                        points = all_obstacle_points.unsqueeze(0)
                        points.requires_grad_(True)
                        cdf_val = robot_cdf.query_cdf(points, config_tensor, return_gradients=False)[0, idx]
                        cdf_val.backward()
                        dh_dp = points.grad[0, idx]  # gradient w.r.t point
                        
                        # Get velocity for this point
                        dyn_idx = idx - len(static_obstacle_points)
                        dp_dt = dynamic_vels[dyn_idx]
                        dh_dt_values.append(torch.dot(dh_dp, dp_dt).item())
                    else:
                        dh_dt_values.append(0.0)
                
                u = controller.generate_controller(
                    current_config,
                    reference_config,
                    h_values.detach().cpu().numpy(),
                    np.stack(h_grads),
                    np.array(dh_dt_values),
                    u_nominal
                )
        
        # Proper dynamics integration
        current_vel = u
        current_config = current_config + current_vel * dt
        
        # Store result
        tracked_configs.append(current_config)
        tracked_vels.append(current_vel)
        
        # Update time
        time += dt
        
        # Print progress occasionally
        if int(time/dt) % 50 == 0:
            tracking_error = np.linalg.norm(current_config - reference_config)
            print(f"Time: {time:.1f}s, s: {s:.3f}, Tracking error: {tracking_error:.3f}")
        
        # Check collision at current configuration
        config_tensor = torch.tensor(current_config, dtype=torch.float32, 
                                   device=device).unsqueeze(0)  # [1, 2]
        sdf_values = robot_sdf.query_sdf(all_obstacle_points.unsqueeze(0),  # [1, N, 2]
                                       config_tensor)  # [1, N]
        
        if torch.min(sdf_values) <= safety_threshold:
            is_safe = False
            print(f"Collision detected at time {time:.2f}s")
            break
        
        # Break if we're close to the end AND tracking error relative to goal is small
        goal_config = trajectory_data['waypoints'][-1]  # Final configuration from planned path
        tracking_error = np.linalg.norm(current_config - reference_config)
        goal_error = np.linalg.norm(current_config - goal_config)
        if s > 0.995 and goal_error < 0.05:
            print(f"\nReached goal at time {time:.1f}s")
            print(f"Final tracking error: {tracking_error:.3f}")
            print(f"Final goal error: {goal_error:.3f}")
            break
    
    if time >= duration:
        print(f"\nReached duration limit ({duration}s)")
        print(f"Final s: {s:.3f}")
        print(f"Final tracking error: {tracking_error:.3f}")
    
    return np.array(tracked_configs), np.array(reference_configs), np.array(tracked_vels), np.array(reference_vels), is_safe

def animate_path(obstacles: List[np.ndarray], tracked_configs, reference_configs, 
                dt: float = 0.02, dynamic_obstacles=True, goal_pos=None):
    """Create an animation of the robot arm tracking the planned path."""
    return create_animation(obstacles, tracked_configs, reference_configs, dt, src_dir, 
                          dynamic_obstacles=dynamic_obstacles, goal_pos=goal_pos)

if __name__ == "__main__":
    # Setup for planning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 5
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
    #goal_pos = np.array([-2.5, 2.5], dtype=np.float32)     # Single goal position for end-effector
    goal_pos = generate_random_goal(rng, robot_sdf, obstacles)
    # find multiple goal configurations by inverse kinematics
    goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])

    # Test different planners: bubble, bubble_connect, cdf_rrt, sdf_rrt
    planner = 'bubble'
    result = plan_and_visualize(
            robot_cdf, robot_sdf, obstacles, initial_config, goal_configs, 
            max_bubble_samples=100, seed=seed, early_termination=False, 
            planner_type=planner, safety_margin=0.25
        )
    
    if result is None:
        print("Planning failed! Exiting...")
        sys.exit(1)

    # simulate with dynamic obstacles
    use_dynamic_obstacles = True
    
    # Execute control
    tracked_configs, reference_configs, tracked_vels, reference_vels, is_safe = track_planned_path(
        obstacles, 
        result, 
        initial_config,
        dt=0.02, 
        duration=40.0,
        control_type='clf_dro_cbf',                # clf_cbf, clf_dro_cbf, pd
        use_bezier=True,
        dynamic_obstacles_exist=use_dynamic_obstacles
    )
    # snapshot_dir = os.path.join(src_dir, 'figures', 'control_snapshots')
    # save_control_snapshots(obstacles, tracked_configs, reference_configs, dt=0.02, 
    #                       output_dir=snapshot_dir)
    
    print(f"Is safe: {is_safe}")
    
    # Create animation
    animate_path(obstacles, tracked_configs, reference_configs, dt=0.02, 
                dynamic_obstacles=use_dynamic_obstacles, goal_pos=goal_pos)