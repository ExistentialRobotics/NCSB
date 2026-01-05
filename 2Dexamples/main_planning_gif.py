import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

from utils_env import create_obstacles
from utils_new import inverse_kinematics_analytical
from robot_cdf import RobotCDF
from planner.bubble_planner import BubblePlanner

def create_cdf_field(robot_cdf, obstacle_points, resolution=100):
    """Create CDF field for visualization"""
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)
    
    obstacle_points = torch.tensor(obstacle_points, device=robot_cdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    
    batch_size = 100
    num_batches = resolution * resolution // batch_size
    cdf_values = np.zeros(resolution * resolution)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        batch_configs = torch.tensor(
            [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
            device=robot_cdf.device,
            dtype=torch.float32
        ).unsqueeze(0)
        
        for i in range(batch_configs.shape[1]):
            config = batch_configs[:, i:i+1, :]
            cdf_val = robot_cdf.query_cdf(obstacle_points, config)
            cdf_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
    
    return cdf_values.reshape(resolution, resolution)

def create_planning_animation(robot_cdf, initial_config, goal_configs, 
                            obstacle_points, bubbles, trajectory, output_path, resolution=50, batch_size=5):
    """Create animated GIF of the planning process"""
    # Create figure with same size as the video
    fig = plt.figure(figsize=(10, 10))
    
    # Create axis with C-space limits (-π to π)
    ax = fig.add_subplot(111)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect('equal')
    
    # Compute CDF field once
    cdf_field = create_cdf_field(robot_cdf, obstacle_points, resolution)
    
    # Calculate number of frames based on batch size
    num_bubbles = len(bubbles.vs)
    num_batches = (num_bubbles + batch_size - 1) // batch_size  # Ceiling division
    
    def update(frame):
        ax.clear()
        
        # Reset axis limits to C-space bounds
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect('equal')
        
        # Plot CDF field
        im = ax.imshow(cdf_field, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                      origin='lower', cmap='viridis', aspect='equal')
        
        # Plot bubbles up to current frame
        if frame < num_batches:  # Normal frames
            # Calculate bubble indices for this batch
            start_idx = 0  # Always include initial bubble
            end_idx = min(batch_size * (frame + 1), num_bubbles)
            
            for vertex in bubbles.vs[:end_idx]:
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
        else:  # Final frame: show all bubbles and trajectory
            # Plot all bubbles
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
            
            # Plot trajectory
            trajectory_array = np.array(trajectory)
            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
                   'r-', linewidth=2, label='Path')
        
        # Plot start and goals
        ax.scatter(initial_config[0], initial_config[1], 
                  color='yellow', marker='o', s=150, label='Start')
        markers = ['^', 's']
        for i, goal in enumerate(goal_configs):
            marker = markers[i] if i < len(markers) else 'o'
            ax.scatter(goal[0], goal[1], 
                      color='r', marker=marker, s=150, label=f'Goal {i+1}')
        
        # Match video font sizes
        ax.set_xlabel('θ₁', fontsize=28)
        ax.set_ylabel('θ₂', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.legend(fontsize=24, loc='lower left')
        
        return [im]
    
    # Create animation
    anim = FuncAnimation(fig, update, 
                        frames=num_batches + 5,
                        interval=200, blit=True)
    
    # Save as GIF
    plt.tight_layout()
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=300)
    plt.close()

def main():
    # Set random seeds
    seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create obstacles and initialize robots
    rng = np.random.default_rng(seed)
    obstacles = create_obstacles(rng=rng)
    robot_cdf = RobotCDF(device=device)
    
    # Set initial and goal configurations
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_pos = np.array([-2.5, 2.5], dtype=np.float32)
    goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize planner
    planner = BubblePlanner(
        robot_cdf=robot_cdf,
        joint_limits=(
            np.full_like(initial_config, -np.pi),
            np.full_like(initial_config, np.pi)
        ),
        max_samples=150,
        batch_size=5,
        device=device,
        seed=seed,
        planner_type='bubble',
        early_termination=False,
        safety_margin=0.1
    )
    
    # Get obstacle points
    obstacle_points = torch.tensor(np.concatenate(obstacles, axis=0), 
                                 device=device)
    
    # Plan path
    print("Planning path...")
    result = planner.plan(initial_config, goal_configs, obstacle_points)
    
    if result is not None and 'metrics' in result and result['metrics'].success:
        print("Planning successful! Creating animation...")
        output_gif = os.path.join(output_dir, 'bubble_planning_animation.gif')
        create_planning_animation(
            robot_cdf=robot_cdf,
            initial_config=initial_config,
            goal_configs=goal_configs,
            obstacle_points=obstacle_points,
            bubbles=result['bubbles'],
            trajectory=result['waypoints'],
            output_path=output_gif,
            batch_size=5  # Pass batch size to animation
        )
        print(f"Animation saved to: {output_gif}")
    else:
        print("Planning failed!")

if __name__ == "__main__":
    main() 