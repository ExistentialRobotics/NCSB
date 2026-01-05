import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from typing import List
import torch
from pathlib import Path

from utils_env import plot_environment, create_dynamic_obstacles, forward_kinematics

def visualize_control_snapshot(obstacles, tracked_configs, reference_configs, t_idx, dt,
                             save_path=None, figsize=(10, 10)):
    """
    Create a publication-quality snapshot of the robot arm at a specific time step.
    
    Args:
        obstacles: List of static obstacle arrays
        tracked_configs: Array of tracked configurations
        reference_configs: Array of reference configurations
        t_idx: Time index for the snapshot
        dt: Time step between frames
        save_path: Path to save the figure
        figsize: Figure size (width, height) in inches
    """
    t = t_idx * dt
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot static obstacles
    for obstacle in obstacles:
        ax.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.5)
        ax.scatter(obstacle[:, 0], obstacle[:, 1], color='red', s=1, alpha=0.8)
    
    # Plot end-effector trajectory up to current time
    ee_positions = []
    for config in tracked_configs[:t_idx+1]:
        x, y = forward_kinematics(config)
        ee_positions.append([x[-1], y[-1]])  # Take the last point (end-effector)
    ee_positions = np.array(ee_positions)
    
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], '--', color='red', alpha=0.7, linewidth=3, label='End-effector')
    
    # Plot dynamic obstacles with velocity arrows
    dynamic_obs, dynamic_vels = create_dynamic_obstacles(t, num_points=50)
    for obs_idx, (obs, vel) in enumerate(zip(dynamic_obs, dynamic_vels)):
        # Plot obstacle
        ax.fill(obs[:, 0], obs[:, 1], color='purple', alpha=0.5)
        ax.scatter(obs[:, 0], obs[:, 1], color='purple', s=1, alpha=0.8)
        
        # Plot velocity arrow
        center = np.mean(obs, axis=0)
        vel_mean = vel[0]  # All points have same velocity
        speed = np.linalg.norm(vel_mean)
        # Scale arrow length based on speed
        arrow_scale = 1.0
        ax.arrow(center[0], center[1], 
                vel_mean[0] * arrow_scale, vel_mean[1] * arrow_scale,
                head_width=0.3, head_length=0.4, fc='purple', ec='purple', alpha=1.0)
        # Add speed label, shifted to the right
        # ax.text(center[0] + 0.3, center[1] + 0.3, f'v = {speed:.1f} m/s', 
        #         horizontalalignment='left', fontsize=24)
    
    # Plot current configuration
    plot_environment(obstacles, tracked_configs[t_idx], ax=ax, robot_color='blue', 
                    plot_obstacles=False, label='Current', highlight_joints=True)
    
    # Plot reference configuration
    plot_environment(obstacles, reference_configs[t_idx], ax=ax, robot_color='green', 
                    plot_obstacles=False, label='Reference', robot_alpha=0.5)
    
    # Clean up the plot
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)
    ax.legend(fontsize=26, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Snapshot saved as {save_path}")
        plt.close()
    else:
        plt.show()

def save_control_snapshots(obstacles, tracked_configs, reference_configs, dt, output_dir):
    """
    Save snapshots at specific time points during the control execution.
    
    Args:
        obstacles: List of static obstacle arrays
        tracked_configs: Array of tracked configurations
        reference_configs: Array of reference configurations
        dt: Time step between frames
        output_dir: Directory to save snapshots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define time points for snapshots (in seconds)
    snapshot_times = [0.0, 8.0, 10.5, 20.1]  # Adjust as needed
    
    for t in snapshot_times:
        t_idx = int(t / dt)
        if t_idx < len(tracked_configs):
            save_path = os.path.join(output_dir, f'control_snapshot_t{t:.1f}.png')
            visualize_control_snapshot(obstacles, tracked_configs, reference_configs, 
                                    t_idx, dt, save_path=save_path)


def visualize_results(obstacles, initial_config, goal_configs, trajectory, src_dir):
    """Create visualizations of the planning results"""
    # Create workspace visualization with adjusted size
    fig_ws, ax_ws = plt.subplots(figsize=(8, 7))  # Reduced from (12, 12) to (8, 7)
    
    # Convert goal_configs to list if it's a numpy array
    if isinstance(goal_configs, np.ndarray):
        if goal_configs.ndim == 1:
            goal_configs = [goal_configs]
        else:
            goal_configs = [g for g in goal_configs]
    
    if trajectory is not None:
        # Plot trajectory in configuration space
        fig_cs, ax_cs = plt.subplots(figsize=(10, 10))
        trajectory = np.array(trajectory)
        
        # Plot trajectory in C-space
        ax_cs.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Path', alpha=0.7)
        ax_cs.plot(initial_config[0], initial_config[1], 'go', label='Start')
        
        # Plot goal configurations
        markers = ['^', 's']  # triangle and square markers
        for i, goal in enumerate(goal_configs):
            marker = markers[i] if i < len(markers) else 'o'
            ax_cs.plot(goal[0], goal[1], f'r{marker}', label=f'Goal {i+1}')
        
        ax_cs.set_xlabel('θ₁', fontsize=20)
        ax_cs.set_ylabel('θ₂', fontsize=20)
        ax_cs.set_title('Path in Configuration Space')
        ax_cs.legend(fontsize=20)
        
        # Save configuration space plot
        plt.savefig(os.path.join(src_dir, 'figures/config_space_path.png'))
        plt.close(fig_cs)
        
        # Plot initial configuration in workspace
        plot_environment(obstacles, initial_config, ax=ax_ws, 
                        robot_color='blue', 
                        robot_alpha=0.9,
                        plot_obstacles=True,
                        label='Start',
                        highlight_joints=True)
        
        # Plot goal configurations with different styles
        linestyles = ['--', ':']  # dashed for goal 1, dotted for goal 2
        for i, goal in enumerate(goal_configs):
            plot_environment(obstacles, goal, ax=ax_ws, 
                           robot_color='red', 
                           robot_alpha=0.7,
                           plot_obstacles=False,
                           label=f'Goal {i+1}',
                           linestyle=linestyles[i],
                           highlight_joints=True)
        
        # Plot waypoints with specific fractions of total trajectory length
        N = len(trajectory)
        fractions = [8, 4, 2, 1.5, 1.2, 1.1, 1.05, 1.02]  # Denominators for N
        indices = [int(N/f) for f in fractions]
        indices = np.unique(indices)  # Remove any duplicates
        
        for idx in indices:
            config = trajectory[idx]
            plot_environment(obstacles, config, ax=ax_ws, 
                           robot_color='green', 
                           robot_alpha=0.4,
                           plot_obstacles=False,
                           label='Waypoints' if idx == indices[0] else None,  # Only label first waypoint
                           add_to_legend=idx == indices[0])  # Only add first waypoint to legend
    
    # Increase tick sizes
    ax_ws.tick_params(axis='both', which='major', labelsize=22)
    
    # Increase axis label sizes
    ax_ws.set_xlabel('X', fontsize=22)
    ax_ws.set_ylabel('Y', fontsize=22)
    
    # Adjust legend
    ax_ws.legend(fontsize=22, loc='lower left')
    
    # Make layout tight
    plt.tight_layout()
    
    # Save workspace visualization
    plt.savefig(os.path.join(src_dir, 'figures/workspace_visualization.png'), dpi=300)
    plt.close(fig_ws)
    
    print("Saved workspace visualization")

def plot_path_comparison(planned_configs, tracked_configs, src_dir):
    """
    Create a visualization comparing planned and executed paths.
    
    Args:
        planned_configs: Array of planned configurations (N, 2)
        tracked_configs: Array of tracked/executed configurations (M, 2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot in configuration space (left plot)
    ax1.plot(planned_configs[:, 0], planned_configs[:, 1], 
             'b-', linewidth=2, label='Planned Path')
    ax1.plot(tracked_configs[:, 0], tracked_configs[:, 1], 
             'r--', linewidth=2, label='Executed Path')
    
    # Add start and end points
    ax1.plot(planned_configs[0, 0], planned_configs[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(planned_configs[-1, 0], planned_configs[-1, 1], 'ro', markersize=10, label='Goal')
    
    ax1.set_xlabel('θ₁', fontsize=14)
    ax1.set_ylabel('θ₂', fontsize=14)
    ax1.set_title('Configuration Space Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    
    # Plot tracking error over time (right plot)
    # Interpolate planned path to match length of tracked path
    from scipy.interpolate import interp1d
    planned_times = np.linspace(0, 1, len(planned_configs))
    tracked_times = np.linspace(0, 1, len(tracked_configs))
    
    # Create interpolation function for each joint
    interp_funcs = [interp1d(planned_times, planned_configs[:, i]) 
                   for i in range(planned_configs.shape[1])]
    
    # Get interpolated planned configurations at tracked times
    interpolated_planned = np.column_stack([f(tracked_times) for f in interp_funcs])
    
    # Compute tracking error
    tracking_error = np.linalg.norm(tracked_configs - interpolated_planned, axis=1)
    
    ax2.plot(tracked_times, tracking_error, 'k-', linewidth=2)
    ax2.set_xlabel('Normalized Time', fontsize=14)
    ax2.set_ylabel('Tracking Error', fontsize=14)
    ax2.set_title('Tracking Error Over Time', fontsize=16)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(src_dir, 'figures/path_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved path comparison plot")
    plt.close()


def visualize_cdf_bubble_planning(robot_cdf, initial_config, goal_configs, trajectory, bubbles, 
                          obstacle_points, src_dir, planner_type='bubble', resolution=50):
    """
    Visualize CDF planning in configuration space (theta1-theta2)
    """
    print("\nDebug - visualize_cdf_planning:")
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)
    
    # Reshape obstacle points to match expected format [B, N, 2]
    obstacle_points = torch.tensor(obstacle_points, device=robot_cdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    print(f"Obstacle points shape: {obstacle_points.shape}")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = resolution * resolution // batch_size
    cdf_values = np.zeros(resolution * resolution)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        # Create batch of configurations
        batch_configs = torch.tensor(
            [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
            device=robot_cdf.device,
            dtype=torch.float32
        ).unsqueeze(0)  # Shape: [1, batch_size, 2]
        
        # For each config in the batch, query CDF with all obstacle points
        for i in range(batch_configs.shape[1]):
            config = batch_configs[:, i:i+1, :]  # [1, 1, 2]
            cdf_val = robot_cdf.query_cdf(obstacle_points, config)
            cdf_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
    
    # Reshape back to grid
    cdf_values = cdf_values.reshape(resolution, resolution)
    
    # Create visualization with adjusted size and layout
    fig, ax = plt.subplots(figsize=(8, 7))  # Reduced figure size
    
    # Plot CDF field without transpose
    im = ax.imshow(cdf_values, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='viridis', aspect='equal')
    cbar = plt.colorbar(im, ax=ax)
    
    # Increase colorbar label size but keep it compact
    cbar.ax.tick_params(labelsize=20)
    # Remove colorbar label to save space
    # cbar.set_label('CDF Value', size=20)
    
    # Plot bubbles from igraph Graph object
    if bubbles is not None:
        for vertex in bubbles.vs:
            circle = vertex["circle"]
            if circle is not None:
                center = circle.centre
                radius = circle.radius
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = center[0] + radius * np.cos(theta)
                circle_y = center[1] + radius * np.sin(theta)
                circle_x = np.clip(circle_x, -np.pi, np.pi)
                circle_y = np.clip(circle_y, -np.pi, np.pi)
                ax.plot(circle_x, circle_y, color='cyan', alpha=0.8, linewidth=2)
    
    # Plot trajectory if available
    if trajectory is not None:
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Planned Path')
    
    # Plot start and goals with different shapes
    ax.scatter(initial_config[0], initial_config[1], color='yellow', marker='o', s=150, label='Start')
    markers = ['^', 's']  # triangle and square markers
    for i, goal in enumerate(goal_configs):
        marker = markers[i] if i < len(markers) else 'o'
        ax.scatter(goal[0], goal[1], color='r', marker=marker, s=150, label=f'Goal {i+1}')
    
    ax.set_xlabel('θ₁', fontsize=20)
    ax.set_ylabel('θ₂', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=20, loc='lower left')  # Adjusted legend position

    # Adjust layout to minimize whitespace
    plt.tight_layout()

    if planner_type == 'bubble':
        plt.savefig(os.path.join(src_dir, 'figures/cdf_bubble_planning_visualization.png'), 
                    bbox_inches='tight', dpi=300)
    elif planner_type == 'bubble_connect':
        plt.savefig(os.path.join(src_dir, 'figures/connect_bubble_planning_visualization.png'), 
                    bbox_inches='tight', dpi=300)
    plt.close()

def visualize_cdf_field(robot_cdf, obstacle_points, resolution=100):
    """
    Visualize CDF field in configuration space (theta1-theta2) with specific level sets
    """
    print("\nVisualizing CDF field...")
    
    # Create figures directory in the same folder as this script
    script_dir = Path(__file__).parent
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)
    
    # Reshape obstacle points to match expected format [B, N, 2]
    obstacle_points = torch.tensor(obstacle_points, device=robot_cdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    print(f"Obstacle points shape: {obstacle_points.shape}")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = resolution * resolution // batch_size
    cdf_values = np.zeros(resolution * resolution)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        # Create batch of configurations
        batch_configs = torch.tensor(
            [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
            device=robot_cdf.device,
            dtype=torch.float32
        ).unsqueeze(0)  # Shape: [1, batch_size, 2]
        
        # For each config in the batch, query CDF with all obstacle points
        for i in range(batch_configs.shape[1]):
            config = batch_configs[:, i:i+1, :]  # [1, 1, 2]
            cdf_val = robot_cdf.query_cdf(obstacle_points, config)
            cdf_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
    
    # Reshape back to grid
    cdf_values = cdf_values.reshape(resolution, resolution)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot CDF field
    im = ax.imshow(cdf_values, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='viridis', aspect='equal')
    
    # Add specific level sets
    levels = [0.2, 0.6, 1.0]
    X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, resolution),
                      np.linspace(-np.pi, np.pi, resolution))
    CS = ax.contour(X, Y, cdf_values, levels=levels, colors='white', alpha=0.7, linewidths=2)
    ax.clabel(CS, inline=True, fontsize=16)  # Increased font size for contour labels
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)  # Increased colorbar tick font size
    cbar.set_label('CDF Value', size=20)  # Increased colorbar label font size
    
    # Increase font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increased tick font size
    ax.set_xlabel('θ₁', fontsize=20)
    ax.set_ylabel('θ₂', fontsize=20)
    ax.set_title('CDF Field', fontsize=24)
    
    # Save with user ownership
    save_path = figures_dir / 'cdf_field.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Ensure file is owned by current user
    os.system(f"sudo chown $USER:$USER {save_path}")
    
    return cdf_values

def visualize_sdf_field(robot_sdf, obstacle_points, resolution=100):
    """
    Visualize SDF field in configuration space (theta1-theta2) with specific level sets
    """
    print("\nVisualizing SDF field...")
    
    # Create figures directory in the same folder as this script
    script_dir = Path(__file__).parent
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Create a grid of configurations
    theta1 = torch.linspace(-np.pi, np.pi, resolution)
    theta2 = torch.linspace(-np.pi, np.pi, resolution)
    T1, T2 = torch.meshgrid(theta1, theta2, indexing='ij')
    
    # Reshape obstacle points to match expected format [1, N, 2]
    obstacle_points = torch.tensor(obstacle_points, device=robot_sdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    
    # Process in batches to avoid memory issues
    batch_size = 500
    num_batches = resolution * resolution // batch_size
    sdf_values = torch.zeros(resolution * resolution, device=robot_sdf.device)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        # Create batch of configurations
        batch_configs = torch.stack([
            T1.flatten()[start_idx:end_idx],
            T2.flatten()[start_idx:end_idx]
        ], dim=1)  # Shape: [batch_size, 2]
        
        # For each config in batch, compute minimum SDF over all obstacle points
        batch_sdfs = []
        for i in range(batch_configs.shape[0]):
            config = batch_configs[i:i+1]  # Take one config [1, 2]
            sdf_val = robot_sdf.query_sdf(obstacle_points, config)  # [1, N]
            min_sdf = sdf_val.min()  # Minimum distance to any obstacle
            batch_sdfs.append(min_sdf)
        
        sdf_values[start_idx:end_idx] = torch.stack(batch_sdfs)
    
    # Reshape back to grid
    sdf_field = sdf_values.reshape(resolution, resolution).cpu()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot SDF field
    im = ax.imshow(sdf_field.T, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='RdBu', aspect='equal')
    
    # Add specific level sets
    levels = [0.2, 0.6, 1.0]
    X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, resolution),
                      np.linspace(-np.pi, np.pi, resolution))
    CS = ax.contour(X, Y, sdf_field.T, levels=levels, colors='black', alpha=0.7, linewidths=2)
    ax.clabel(CS, inline=True, fontsize=16)  # Increased font size for contour labels
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)  # Increased colorbar tick font size
    cbar.set_label('SDF Value', size=20)  # Increased colorbar label font size
    
    # Increase font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increased tick font size
    ax.set_xlabel('θ₁', fontsize=20)
    ax.set_ylabel('θ₂', fontsize=20)
    ax.set_title('SDF Field', fontsize=24)
    
    # Save with user ownership
    save_path = figures_dir / 'sdf_field.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Ensure file is owned by current user
    os.system(f"sudo chown $USER:$USER {save_path}")
    
    return sdf_field

def visualize_ompl_rrt_planning(robot_cdf, robot_sdf, initial_config, goal_configs, trajectory, 
                              obstacle_points, src_dir, planner_type='sdf_rrt', resolution=50):
    """
    Visualize OMPL RRT planning in configuration space (theta1-theta2)
    
    Args:
        robot_cdf: RobotCDF instance
        robot_sdf: RobotSDF instance
        initial_config: Initial configuration [2]
        goal_configs: List of goal configurations [N, 2]
        trajectory: Planned trajectory waypoints [M, 2]
        obstacle_points: Obstacle points tensor
        src_dir: Directory to save figures
        planner_type: 'cdf_rrt' or 'sdf_rrt'
        resolution: Grid resolution for field visualization
    """
    print(f"\nVisualizing {planner_type} planning...")
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)
    
    # Reshape obstacle points to [1, N, 2]
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    
    # Get field values based on planner type
    if planner_type == 'cdf_rrt':
        field_values = np.zeros(resolution * resolution)
        batch_size = 100
        num_batches = resolution * resolution // batch_size
        
        for b in range(num_batches + 1):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, resolution * resolution)
            if start_idx >= end_idx:
                break
                
            # Create batch of configurations [batch_size, 2]
            batch_configs = torch.tensor(
                [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
                device=robot_cdf.device,
                dtype=torch.float32
            )
            
            for i in range(batch_configs.shape[0]):
                # Reshape config to [1, 1, 2] for CDF query
                config = batch_configs[i:i+1].unsqueeze(0)  # [1, 1, 2]
                cdf_val = robot_cdf.query_cdf(obstacle_points, config)
                field_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
        
        field_values = field_values.reshape(resolution, resolution)
        cmap = 'viridis'
        field_label = 'CDF Value'
        
    else:  # sdf_rrt
        field_values = np.zeros(resolution * resolution)
        batch_size = 100
        num_batches = resolution * resolution // batch_size
        
        for b in range(num_batches + 1):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, resolution * resolution)
            if start_idx >= end_idx:
                break
                
            # Create batch of configurations [batch_size, 2]
            batch_configs = torch.tensor(
                [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
                device=robot_sdf.device,
                dtype=torch.float32
            )
            
            for i in range(batch_configs.shape[0]):
                # Reshape config to [1, 2] for SDF query
                config = batch_configs[i:i+1]  # [1, 2]
                sdf_val = robot_sdf.query_sdf(obstacle_points, config)
                field_values[start_idx + i] = sdf_val.min().detach().cpu().numpy()
        
        field_values = field_values.reshape(resolution, resolution)
        cmap = 'RdBu'
        field_label = 'SDF Value'
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot field
    im = ax.imshow(field_values, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap=cmap, aspect='equal')
    plt.colorbar(im, ax=ax, label=field_label)
    
    # Plot trajectory if available
    if trajectory is not None:
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Planned Path')
    
    # Plot start and goals with different shapes
    ax.scatter(initial_config[0], initial_config[1], color='g', marker='o', s=100, label='Start')
    for i, goal in enumerate(goal_configs):
        marker = '^' if i == 0 else 's'  # triangle for first goal, square for second
        ax.scatter(goal[0], goal[1], color='r', marker=marker, s=100, label=f'Goal {i+1}')
    
    # Make planner name more readable for the title
    planner_title = planner_type.replace('_', ' ').upper()
    ax.set_xlabel('θ₁', fontsize=18)
    ax.set_ylabel('θ₂', fontsize=18)
    ax.set_title(f'OMPL {planner_title} Planning Visualization', fontsize=18)
    ax.legend(fontsize=12)
    
    # Save figure
    planner_name = planner_type.replace('_', '-')  # Make filename more readable
    plt.savefig(os.path.join(src_dir, f'figures/ompl_{planner_name}_planning_visualization.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def visualize_workspace(obstacles, initial_config=None, save=True, resolution=100):
    """
    Visualize and save the workspace with obstacles and robot arm
    Args:
        obstacles: List of obstacle point clouds
        initial_config: Initial configuration of the robot arm (optional)
        save: Whether to save the plot
        resolution: Resolution for plotting
    """
    print("\nVisualizing workspace...")
    
    # Create figures directory in the same folder as this script
    script_dir = Path(__file__).parent
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot obstacles
    for obstacle in obstacles:
        ax.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.5)
        ax.scatter(obstacle[:, 0], obstacle[:, 1], color='red', s=1, alpha=0.8)
    
    # Plot robot arm if configuration is provided
    if initial_config is not None:
        plot_environment(obstacles, initial_config, ax=ax, 
                        plot_obstacles=False,  # Obstacles already plotted
                        robot_color='blue',
                        highlight_joints=True,
                        label='Robot')
    
    # Set axis properties
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    
    # Increase font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_title('Workspace', fontsize=24)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save:
        save_path = figures_dir / 'workspace.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        os.system(f"sudo chown $USER:$USER {save_path}")
        plt.close()
    else:
        plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # Example usage
    from robot_cdf import RobotCDF
    from utils_env import create_obstacles

    from robot_sdf import RobotSDF

    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rng = np.random.default_rng(seed)
    
    # Create robot CDF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    # Create obstacles
    obstacles = create_obstacles(rng=rng)
    obstacle_points = np.concatenate(obstacles, axis=0)
    
    # Create initial configuration
    initial_config = np.array([0.0, 0.0])  # Example configuration
    
    # Visualize workspace with robot arm
    visualize_workspace(obstacles, initial_config=initial_config, save=True, resolution=100)




