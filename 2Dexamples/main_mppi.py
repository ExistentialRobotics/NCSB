import numpy as np
import matplotlib
import imageio.v2 as imageio
import torch
import time
import json
import sys
import os

import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List

# Set matplotlib backend before importing pyplot to avoid Wayland warnings
matplotlib.use('Agg')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_dir = os.path.dirname(os.path.abspath(__file__))

from utils_new import inverse_kinematics_analytical, forward_kinematics_analytical, forward_kinematics_ee_from_array
from utils_env import create_obstacles, plot_environment, create_dynamic_obstacles
from mppi_functional_2d import setup_mppi_controller_2d
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF

def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
        obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
        np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)

def run_mppi_control(
    obstacles: List[np.ndarray],
    initial_config: np.ndarray,
    goal_pos: np.ndarray,
    robot_cdf: RobotCDF,
    dt: float = 0.02,
    duration: float = 40.0,
    goal_threshold: float = 0.05,
    samples: int = 200,
    horizon: int = 20,
    safety_margin: float = 0.1,
    control_bound: float = 2.0,
    bootstrap_iters: int = 5,
    seed: int = 0,
    test_single_iteration: bool = False,
    dynamic_obstacles_flag: bool = False,
):
    """
    Run MPPI control from initial config to goal position
    
    Args:
        obstacles: List of obstacle arrays
        initial_config: Initial joint configuration [2]
        goal_pos: Goal end-effector position [2]
        robot_cdf: RobotCDF instance
        dt: Time step
        duration: Maximum duration
        goal_threshold: Distance threshold for goal reaching
        samples: Number of MPPI samples
        horizon: Planning horizon
        safety_margin: Safety margin for CDF
        control_bound: Maximum control input
    
    Returns:
        tracked_configs: Array of tracked configurations
        rollout_trajectories_list: List of rollout trajectories for each step
        is_safe: Whether execution was safe
    """
    device = robot_cdf.device
    
    # Convert obstacles to tensor
    static_obstacle_points_np = concatenate_obstacle_list(obstacles)
    static_obstacle_points = torch.tensor(
        static_obstacle_points_np, 
        dtype=torch.float32, 
        device=device
    )
    
    # Convert goal to tensor
    goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.float32, device=device)
    
    # Initialize MPPI controller
    mppi = setup_mppi_controller_2d(
        robot_cdf=robot_cdf,
        robot_n=2,
        input_size=2,
        initial_horizon=horizon,
        samples=samples,
        control_bound=control_bound,
        dt=dt,
        use_GPU=(device == 'cuda'),
        costs_lambda=0.03,
        cost_goal_coeff=30.0,
        cost_safety_coeff=10.0,  
        cost_perturbation_coeff=0.02,
        action_smoothing=0.5
    )
    
    # Initialize RobotSDF for collision checking
    robot_sdf = RobotSDF(device=device)
    
    # Initialize state
    current_config = torch.tensor(initial_config, dtype=torch.float32, device=device)
    tracked_configs = [current_config.cpu().numpy()]
    rollout_trajectories_list = []
    elapsed_time = 0.0
    is_safe = True
    safety_threshold = 0.01
    
    # Metrics
    goal_distances = []
    sdf_distances = []
    cdf_distances = []
    path_length = 0.0
    prev_config = current_config.clone()
    num_collision_checks_env = 0

    # ---- Wrap CDF ----
    def wrap_robot_cdf(cdf_module):
        original = cdf_module.query_cdf
        def wrapped(points, joint_angles, *args, **kwargs):
            nonlocal num_collision_checks_env
            B = joint_angles.shape[0] if isinstance(joint_angles, torch.Tensor) else len(joint_angles)
            num_collision_checks_env += B
            return original(points, joint_angles, *args, **kwargs)
        cdf_module.query_cdf = wrapped
    wrap_robot_cdf(robot_cdf)

    # Initialize control sequence
    U = torch.zeros((horizon, 2), device=device)
    
    # Print interval (in seconds)
    print_interval = 0.1  # Print every 0.1 seconds
    last_print_time = -print_interval  # Initialize to allow first print
    
    print(f"\nStarting MPPI control...")
    print(f"Initial config: {initial_config}")
    print(f"Goal position: {goal_pos}")
    print(f"Samples: {samples}, Horizon: {horizon}")
    print(f"Bootstrap iters: {bootstrap_iters}")
    
    # ---- BOOTSTRAP MPPI ----
    bootstrap_time = 0.0
    if bootstrap_iters > 0:
        print(f"\nBootstrapping MPPI with {bootstrap_iters} iterations...")
        t_boot_start = time.time()
        for k in range(bootstrap_iters):
            # We only refine U here; we don't use action/states/rollouts
            _, _, U, _ = mppi( U=U, init_state=current_config, goal_pos=goal_pos_tensor, obstaclesX=static_obstacle_points, safety_margin=safety_margin, batch_size=100)
        t_boot_end = time.time()
        bootstrap_time = t_boot_end - t_boot_start
        print(f"Bootstrap completed in {bootstrap_time:.2f} seconds.")
    else:
        print("\nSkipping bootstrap (bootstrap_iters <= 0).")

    iteration = 0
    t_wall_start = time.time()
    
    # Timing statistics for MPPI iterations
    mppi_times = []
    step_times = []
    
    # Test mode: run only 1 iteration for timing test
    if test_single_iteration:
        print("\n" + "="*60)
        print("TEST MODE: Running only 1 MPPI iteration for timing")
        print("="*60)
    
    while elapsed_time < duration:
        step_start_time = time.time()
        
        if dynamic_obstacles_flag:
            # ---- Build dynamic obstacles at current time ----
            dynamic_obstacles, dynamic_vels = create_dynamic_obstacles(elapsed_time, num_points=50)

            if len(dynamic_obstacles) > 0:
                dynamic_points_np = concatenate_obstacle_list(dynamic_obstacles)
                dynamic_points = torch.tensor(dynamic_points_np, dtype=torch.float32, device=device)

                all_obstacle_points = torch.cat([static_obstacle_points, dynamic_points], dim=0)
            else:
                all_obstacle_points = static_obstacle_points
        else:
            all_obstacle_points = static_obstacle_points

        # Run MPPI step
        mppi_start_time = time.time()
        states_final, action, U, rollout_trajectories = mppi(
            U=U,
            init_state=current_config,
            goal_pos=goal_pos_tensor,
            obstaclesX=all_obstacle_points,
            safety_margin=safety_margin,
            batch_size=100
        )
        mppi_time = time.time() - mppi_start_time
        mppi_times.append(mppi_time)
        
        # Store rollout trajectories for visualization
        rollout_trajectories_list.append(rollout_trajectories.numpy())
        
        # Apply first action
        current_config = current_config + action.squeeze() * dt

        # ---- Path length (Σ |Δq|) ---- 
        path_length += torch.abs(current_config - prev_config).sum().item()
        prev_config = current_config.clone()
        tracked_configs.append(current_config.cpu().numpy())
        
        # Calculate step time (includes MPPI + other operations)
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # Calculate average frequency over last 10 steps
        if len(step_times) >= 10:
            avg_step_time = np.mean(step_times[-10:])
            avg_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0
            avg_mppi_time = np.mean(mppi_times[-10:])
        else:
            avg_step_time = step_time
            avg_freq = 1.0 / step_time if step_time > 0 else 0
            avg_mppi_time = mppi_time
        
        # Print timing every 10 iterations or at first iteration
        if iteration % 10 == 0 or iteration == 0:
            print(f"\nMPPI Iteration {iteration}:")
            print(f"  MPPI computation time: {mppi_time*1000:.2f} ms")
            print(f"  Total step time: {step_time*1000:.2f} ms")
            print(f"  Current frequency: {1.0/step_time:.2f} Hz (target: {1.0/dt:.1f} Hz)")
            if iteration > 0:
                print(f"  Avg frequency (last 10): {avg_freq:.2f} Hz")
                print(f"  Avg MPPI time (last 10): {avg_mppi_time*1000:.2f} ms")
        
        # If test mode, break after 1 iteration
        if test_single_iteration and iteration == 0:
            print(f"\n{'='*60}")
            print(f"Single MPPI Iteration Timing Report:")
            print(f"{'='*60}")
            print(f"MPPI computation time: {mppi_time*1000:.2f} ms")
            print(f"Total step time: {step_time*1000:.2f} ms")
            print(f"MPPI frequency: {1.0/mppi_time:.2f} Hz")
            print(f"Step frequency: {1.0/step_time:.2f} Hz")
            print(f"Target frequency: {1.0/dt:.1f} Hz (dt={dt:.3f}s)")
            if mppi_time > dt:
                print(f"⚠️  WARNING: MPPI is slower than target ({mppi_time*1000:.2f} ms > {dt*1000:.1f} ms)")
            else:
                print(f"✓ MPPI is faster than target ({mppi_time*1000:.2f} ms < {dt*1000:.1f} ms)")
            print(f"{'='*60}\n")
            break
        
        # Update time
        elapsed_time += dt
        iteration += 1
        
        # Check current end-effector position
        current_ee_pos = forward_kinematics_ee_from_array(current_config.cpu().numpy())
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        
        # Check collision
        config_tensor = current_config.unsqueeze(0)  # [1, 2]
        obstacle_points_batch = all_obstacle_points.unsqueeze(0)  # [1, N, 2]
        sdf_values = robot_sdf.query_sdf(obstacle_points_batch, config_tensor)
        min_sdf = torch.min(sdf_values).item()

        # ---- CDF-based distance for metrics ----
        cdf_values = robot_cdf.query_cdf(points=obstacle_points_batch, joint_angles=config_tensor, return_gradients=False)
        min_cdf = torch.min(cdf_values).item()

        # Log metrics per step
        goal_distances.append(float(distance_to_goal))
        sdf_distances.append(float(min_sdf))
        cdf_distances.append(float(min_cdf))

        if torch.min(sdf_values) <= safety_threshold:
            is_safe = False
            print(f"Collision detected at time {elapsed_time:.2f}s")
            break
        
        # Print progress at regular time intervals (simulation time, not wall clock)
        if elapsed_time - last_print_time >= print_interval:
            min_sdf = torch.min(sdf_values).item()
            print(f"Step: {iteration}, Distance to goal: {distance_to_goal:.4f}, Min SDF: {min_sdf:.4f}, Config: {current_config.cpu().numpy()}")
            last_print_time = elapsed_time
        
        # Check if goal reached
        if distance_to_goal < goal_threshold:
            print(f"\nGoal reached at time {elapsed_time:.1f}s!")
            print(f"Final distance to goal: {distance_to_goal:.4f}")
            break
    
    t_wall_end = time.time()
    control_time = t_wall_end - t_wall_start
    
    # Print final timing statistics
    if len(mppi_times) > 0:
        print(f"\n{'='*60}")
        print(f"MPPI Timing Statistics:")
        print(f"  Total iterations: {len(mppi_times)}")
        print(f"  Total control time: {control_time:.2f} s")
        print(f"  Target frequency: {1.0/dt:.1f} Hz (dt={dt:.3f}s)")
        
        avg_mppi_time = np.mean(mppi_times)
        std_mppi_time = np.std(mppi_times)
        min_mppi_time = np.min(mppi_times)
        max_mppi_time = np.max(mppi_times)
        
        avg_step_time = np.mean(step_times)
        avg_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0
        target_freq = 1.0 / dt
        
        print(f"  Actual average frequency: {avg_freq:.2f} Hz")
        print(f"  MPPI iteration time:")
        print(f"    Mean: {avg_mppi_time*1000:.2f} ms")
        print(f"    Std:  {std_mppi_time*1000:.2f} ms")
        print(f"    Min:  {min_mppi_time*1000:.2f} ms")
        print(f"    Max:  {max_mppi_time*1000:.2f} ms")
        print(f"  Step time (mean): {avg_step_time*1000:.2f} ms")
        if avg_freq < target_freq * 0.9:
            print(f"  ⚠️  WARNING: Running slower than target ({avg_freq:.2f} Hz < {target_freq:.1f} Hz)")
        print(f"{'='*60}\n")

    if elapsed_time >= duration:
        print(f"\nReached duration limit ({duration}s)")
        print(f"Final distance to goal: {distance_to_goal:.4f}")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = project_root / "2Dexamples" / "results" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        final_ee_pos = forward_kinematics_ee_from_array(tracked_configs[-1])
        success = bool(distance_to_goal < goal_threshold and is_safe)
        
        metrics = {
            "timestamp": timestamp,
            "planner": "mppi_2d",
            "dynamic_obstacles": dynamic_obstacles_flag,
            "initial_config": initial_config.tolist(),
            "goal_pos": goal_pos.tolist(),
            "final_config": tracked_configs[-1].tolist(),
            "final_ee_pos": final_ee_pos.tolist(),
            "final_distance_to_goal": float(distance_to_goal),
            "steps": int(len(tracked_configs) - 1),
            "dt": float(dt),
            "duration_limit": float(duration),
            "samples": int(samples),
            "horizon": int(horizon),
            "safety_margin": float(safety_margin),
            "control_bound": float(control_bound),
            "is_safe": bool(is_safe),
            "success": success,
            "seed": int(seed),
            "collision_checks_env": int(num_collision_checks_env),
            "path_length": float(path_length),
            "control_time": float(control_time),
            "bootstrap_time": float(bootstrap_time),
            "mppi_iterations": len(mppi_times),
            "avg_mppi_time_ms": float(np.mean(mppi_times) * 1000) if len(mppi_times) > 0 else 0.0,
            "std_mppi_time_ms": float(np.std(mppi_times) * 1000) if len(mppi_times) > 0 else 0.0,
            "min_mppi_time_ms": float(np.min(mppi_times) * 1000) if len(mppi_times) > 0 else 0.0,
            "max_mppi_time_ms": float(np.max(mppi_times) * 1000) if len(mppi_times) > 0 else 0.0,
            "avg_frequency_hz": float(1.0 / np.mean(step_times)) if len(step_times) > 0 and np.mean(step_times) > 0 else 0.0,
            "target_frequency_hz": float(1.0 / dt),
            "bootstrap_iters": int(bootstrap_iters),
            "goal_distances": goal_distances,
            "sdf_distances": sdf_distances,
            "cdf_distances": cdf_distances,
            "trajectory": [config.tolist() for config in tracked_configs] if tracked_configs is not None else None
        }
        
        metrics_path = metrics_dir / f"metrics_mppi_2d_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nSaved 2D MPPI metrics to: {metrics_path}")
    except Exception as e:
        print(f"\nWarning: failed to save 2D metrics: {e}")

    return np.array(tracked_configs), rollout_trajectories_list, is_safe

def create_mppi_animation( obstacles: List[np.ndarray], dynamic_obstacles: bool, tracked_configs: np.ndarray, rollout_trajectories_list: List[np.ndarray], goal_pos: np.ndarray, dt: float = 0.02, src_dir=None):
    """
    Create animation of MPPI control with rollouts
    
    Args:
        obstacles: List of obstacle arrays
        tracked_configs: Array of tracked configurations [T, 2]
        rollout_trajectories_list: List of rollout trajectories for each step
            Each element is [num_vis_samples, horizon+1, 2]
        goal_pos: Goal end-effector position [2]
        dt: Time step
        src_dir: Source directory for saving
    """
    # Slow down animation by reducing fps (show more frames per second of simulation)
    animation_fps = 10  # Slower animation - 10 fps instead of 50
    n_configs = len(tracked_configs)
    
    print(f"\nCreating MPPI animation for {n_configs} configurations (dt={dt}s, animation_fps={animation_fps})")
    
    frames = []
    # Use standard figure size for GIF (no need for specific dimensions)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(n_configs):
        ax.clear()
        current_config = tracked_configs[i]
        t = i * dt
        
        # Plot static environment and current configuration
        plot_environment(obstacles, current_config, ax=ax, robot_color='blue', label='Robot')
        
        if dynamic_obstacles:
            dynamic_obstacles, dynamic_velocities = create_dynamic_obstacles(t, num_points=50)
            for obs, vel in zip(dynamic_obstacles, dynamic_velocities):
                ax.fill(obs[:, 0], obs[:, 1], color='purple', alpha=0.5, label='Dynamic Obstacle')
                ax.scatter(obs[:, 0], obs[:, 1], color='purple', s=1, alpha=0.8)

        # Plot goal position
        ax.plot(goal_pos[0], goal_pos[1], '*', color='red', markersize=15, label='Goal')
        
        # Plot rollouts if available
        if i < len(rollout_trajectories_list):
            rollout_trajs = rollout_trajectories_list[i]  # [num_vis_samples, horizon+1, 2]
            
            # Plot each rollout trajectory
            for traj_idx in range(len(rollout_trajs)):
                traj = rollout_trajs[traj_idx]  # [horizon+1, 2]
                
                # Skip the first state (index 0) - it's the current state, not future
                # Only plot future states (indices 1 to horizon+1)
                future_traj = traj[1:]  # [horizon, 2] - future joint configurations
                
                # Convert future joint angles to end-effector positions for visualization
                ee_traj = np.array([forward_kinematics_ee_from_array(joint_angles) for joint_angles in future_traj])  # [horizon, 2]
                
                # Plot future end-effector trajectory (light gray, semi-transparent)
                ax.plot(ee_traj[:, 0], ee_traj[:, 1], '--', color='gray', alpha=0.3, linewidth=0.5)
        
        ax.legend(fontsize=24)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        
        # Convert plot to RGB array (in-memory, no file saving)
        fig.canvas.draw()
        # Get the buffer directly without saving to file
        # Use BytesIO to save to memory instead of disk
        from io import BytesIO
        buf = BytesIO()
        # Use fixed bbox to ensure consistent frame sizes
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        image = imageio.imread(buf)
        # Ensure all images have the same shape by resizing if needed
        if len(frames) > 0:
            target_shape = frames[0].shape
            if image.shape != target_shape:
                from PIL import Image
                image_pil = Image.fromarray(image)
                image_pil = image_pil.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
                image = np.array(image_pil)
        frames.append(image)
        buf.close()

    plt.close(fig)
    
    if src_dir:
        print("Saving MPPI animation...")
        os.makedirs(os.path.join(src_dir, 'figures'), exist_ok=True)
        output_path = os.path.join(src_dir, 'figures', 'mppi_animation.gif')
        imageio.mimsave(output_path, frames, fps=animation_fps, format='GIF', loop=0)
        print(f"Animation saved as '{output_path}'")
        
        # Clean up: Delete any leftover PNG frame files
        figures_dir = os.path.join(src_dir, 'figures')
        if os.path.exists(figures_dir):
            for filename in os.listdir(figures_dir):
                if filename.startswith('mppi_frame_') and filename.endswith('.png'):
                    png_path = os.path.join(figures_dir, filename)
                    try:
                        os.remove(png_path)
                    except Exception as e:
                        print(f"Warning: Could not delete {png_path}: {e}")
    
    return frames

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try to use apple mps if available
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rng = np.random.default_rng(seed)
    
    # Create obstacles
    obstacles = create_obstacles(rng=rng)
    
    # Initialize RobotCDF
    robot_cdf = RobotCDF(device=device)
    
    # Make sure the figures directory exists
    os.makedirs(os.path.join(src_dir, 'figures'), exist_ok=True)
    
    # Set initial configuration and goal position
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_pos = np.array([-2.5, 2.5], dtype=np.float32)  # End-effector goal position
    
    print(f"\n{'='*50}")
    print("MPPI Control for 2D Robot")
    print(f"{'='*50}")
    print(f"Initial config: {initial_config}")
    print(f"Goal position: {goal_pos}")
    
    # Run MPPI control
    # Set test_single_iteration=True to test just 1 iteration for timing
    tracked_configs, rollout_trajectories_list, is_safe = run_mppi_control(
        obstacles=obstacles,
        initial_config=initial_config,
        goal_pos=goal_pos,
        robot_cdf=robot_cdf,
        dt=0.02,
        duration=40.0,
        goal_threshold=0.05,
        samples=100,                
        horizon=20,
        safety_margin=0.4,
        control_bound=2.0,
        bootstrap_iters=50,
        test_single_iteration=False, #Set to True to test just 1 iteration
        dynamic_obstacles_flag=True,
    )
    final_ee_pos = forward_kinematics_ee_from_array(tracked_configs[-1])
    
    print(f"\nIs safe: {is_safe}")
    print(f"Final config: {tracked_configs[-1]}")
    print(f"Final end-effector position: {final_ee_pos}")
    print(f"Distance to goal: {np.linalg.norm(final_ee_pos - goal_pos):.4f}")
    
    # Create and save animation
    create_mppi_animation(
        obstacles=obstacles,
        dynamic_obstacles=True,
        tracked_configs=tracked_configs,
        rollout_trajectories_list=rollout_trajectories_list,
        goal_pos=goal_pos,
        dt=0.02,
        src_dir=src_dir
    )
    
    print("\nMPPI control complete!")

