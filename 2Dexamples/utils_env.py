import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional
import random
import os
import imageio

from utils_new import forward_kinematics_analytical, forward_kinematics_ee_from_array, forward_kinematics_chain_from_array

def create_circle(center: Tuple[float, float], radius: float, num_points: int = 100) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y))

def create_polygon(vertices: List[Tuple[float, float]], num_points: int = 100) -> np.ndarray:
    points = []
    num_edges = len(vertices)
    points_per_edge = num_points // num_edges
    remaining_points = num_points % num_edges

    for i in range(num_edges):
        start = np.array(vertices[i])
        end = np.array(vertices[(i+1) % num_edges])
        edge_points = points_per_edge + (1 if i < remaining_points else 0)
        t = np.linspace(0, 1, edge_points)
        segment_points = start[None, :] + t[:, None] * (end - start)[None, :]
        points.append(segment_points)
    
    return np.vstack(points)

def create_ellipse(center: Tuple[float, float], a: float, b: float, angle: float = 0, num_points: int = 100) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_points = np.dot(rotation_matrix, np.vstack((x, y)))
    return np.column_stack((rotated_points[0] + center[0], rotated_points[1] + center[1]))

def create_obstacles(num_points: int = 50, rng=None) -> List[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(12345)

    obstacles = []
    
    def random_position(quadrant):
        x_range, y_range = {
            1: ((2, 3), (2, 3)),
            2: ((2, 3), (-2.5, -1.5)),
            3: ((-3, -2.5), (-3.5, -3.)),
            4: ((-3.5, -3.1), (3.1, 3.5))
        }[quadrant]
        return rng.uniform(*x_range), rng.uniform(*y_range)

    # Obstacle 1: Circle in first quadrant
    center = random_position(1)
    obstacles.append(create_circle(center, radius=rng.uniform(0.6, 0.8), num_points=num_points))

    # Obstacle 2: Ellipse in 4th quadrant
    center = random_position(2)
    obstacles.append(create_ellipse(center, a=rng.uniform(0.7, 0.8), b=rng.uniform(0.4, 0.6), 
                                    angle=rng.uniform(0, np.pi), num_points=num_points))
    #obstacles.append(create_circle(center, radius=rng.uniform(0.3, 0.6), num_points=num_points))

    # Obstacle 3: Triangle in third quadrant
    center = random_position(3)
    vertices = [
        (center[0] - random.uniform(0.7, 0.9), center[1] - random.uniform(0.5, 0.8)),
        (center[0] + random.uniform(0.7, 0.9), center[1] - random.uniform(0.5, 0.8)),
        (center[0] + random.uniform(-0.3, 0.3), center[1] + random.uniform(0.5, 0.8))
    ]
    obstacles.append(create_polygon(vertices, num_points=num_points))
    #obstacles.append(create_circle(center, radius=rng.uniform(0.3, 0.6), num_points=num_points))

    # Obstacle 4: Hexagon in 2nd quadrant
    center = random_position(4)
    hexagon_vertices = []
    for i in range(6):
        angle = i * (2 * np.pi / 6)
        x = center[0] + 0.6 * np.cos(angle)
        y = center[1] + 0.6 * np.sin(angle)
        hexagon_vertices.append((x, y))
    obstacles.append(create_polygon(hexagon_vertices, num_points=num_points))
    

    return obstacles

def plot_environment(obstacles: List[np.ndarray], arm_angles: np.ndarray, 
                    ax: Optional[plt.Axes] = None, save_path: Optional[str] = None,
                    goal_angles: Optional[np.ndarray] = None,
                    robot_color: str = 'blue', robot_alpha: float = 1.0,
                    plot_obstacles: bool = True, add_to_legend: bool = True,
                    label: Optional[str] = None, linestyle: str = '-',
                    highlight_joints: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

    # Plot obstacles
    if plot_obstacles:
        for obstacle in obstacles:
            ax.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.5)
            ax.scatter(obstacle[:, 0], obstacle[:, 1], color='red', s=1, alpha=0.8)

    # Plot current configuration
    arm_x, arm_y = forward_kinematics(arm_angles)
    ax.plot(arm_x, arm_y, color=robot_color, linestyle=linestyle, 
            linewidth=5, alpha=robot_alpha, 
            label=label if add_to_legend and label else None)
    
    # Highlight joints if requested
    if highlight_joints:
        # Base joint and middle joint are always black
        ax.plot(arm_x[0], arm_y[0], 'ko', markersize=15, alpha=robot_alpha)
        ax.plot(arm_x[1], arm_y[1], 'ko', markersize=15, alpha=robot_alpha)
        
        # End effector color depends on whether this is start or goal configuration
        if label == 'Start':
            ax.plot(arm_x[2], arm_y[2], 'go', markersize=15, alpha=robot_alpha)  # Green for start
        elif label and label.startswith('Goal'):
            ax.plot(arm_x[2], arm_y[2], 'ro', markersize=15, alpha=robot_alpha)  # Red for goal
        else:
            ax.plot(arm_x[2], arm_y[2], 'ko', markersize=15, alpha=robot_alpha)  # Black for intermediate

    num_links = arm_angles.shape[0]
    ax.set_xlim(-num_links * 2 -1 , num_links * 2 + 1)
    ax.set_ylim(-num_links * 2 -1 , num_links * 2 + 1)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)

    plt.tight_layout()
    
    if add_to_legend:
        ax.legend(fontsize=26)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    
    return fig, ax

def create_dynamic_obstacles(t: float, num_points: int = 50) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create dynamic obstacles at time t with bouncing motion between start and goal positions.
    
    Args:
        t: Current time in seconds
        num_points: Number of points to sample on each obstacle surface
        
    Returns:
        Tuple of (obstacle_points, obstacle_velocities)
        - obstacle_points: List of obstacle point clouds at time t
        - obstacle_velocities: List of velocity vectors for each point in the obstacles
    """
    obstacles = []
    velocities = []
    
    # First obstacle: moving horizontally
    start1 = (-4.5, 1.7)
    goal1 = (4.5, 1.7)
    speed1 = 0.4  # m/s
    distance1 = np.linalg.norm(np.array(start1) - np.array(goal1))  # Total distance between start and goal
    cycle_time1 = distance1 / speed1  # Time for one-way trip
    
    # Calculate current position and velocity
    t1 = t % (2 * cycle_time1)
    if t1 <= cycle_time1:
        # Moving towards goal
        velocity1 = np.array([speed1, 0.0])
        progress = t1 / cycle_time1
        x1 = start1[0] + (goal1[0] - start1[0]) * progress
    else:
        # Moving back to start
        velocity1 = np.array([-speed1, 0.0])
        progress = (t1 - cycle_time1) / cycle_time1
        x1 = goal1[0] + (start1[0] - goal1[0]) * progress
    
    center1 = (x1, start1[1])
    obstacle1 = create_circle(center1, radius=0.5, num_points=num_points)
    obstacles.append(obstacle1)
    velocities.append(np.tile(velocity1, (num_points, 1)))  # Same velocity for all points
    
    # Second obstacle: moving vertically
    start2 = (-2, 4.5)
    goal2 = (-2, -4.5)
    speed2 = 0.5  # m/s
    distance2 = np.linalg.norm(np.array(start2) - np.array(goal2))  # Total distance between start and goal
    cycle_time2 = distance2 / speed2  # Time for one-way trip
    
    # Calculate current position and velocity
    t2 = t % (2 * cycle_time2)
    if t2 <= cycle_time2:
        # Moving towards goal
        velocity2 = np.array([0.0, -speed2])
        progress = t2 / cycle_time2
        y2 = start2[1] + (goal2[1] - start2[1]) * progress
    else:
        # Moving back to start
        velocity2 = np.array([0.0, speed2])
        progress = (t2 - cycle_time2) / cycle_time2
        y2 = goal2[1] + (start2[1] - goal2[1]) * progress
    
    center2 = (start2[0], y2)
    obstacle2 = create_circle(center2, radius=0.5, num_points=num_points)
    obstacles.append(obstacle2)
    velocities.append(np.tile(velocity2, (num_points, 1)))  # Same velocity for all points
    
    return obstacles, velocities

def create_animation(obstacles: List[np.ndarray], tracked_configs, reference_configs, 
                    dt: float = 0.02, src_dir=None, dynamic_obstacles: bool = False, goal_pos=None):
    """
    Create an animation of the robot arm tracking the planned path.
    
    Args:
        obstacles: List of static obstacle arrays
        tracked_configs: Array of tracked configurations
        reference_configs: Array of reference configurations
        dt: Time step between frames (seconds)
        src_dir: Source directory for saving the animation
        dynamic_obstacles: Whether to include dynamic obstacles
    """
    fps = int(1/dt)
    
    n_configs = len(tracked_configs)
    print(f"\nCreating animation for {n_configs} configurations (dt={dt}s, fps={fps})")
    
    frames = []
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(n_configs):
        ax.clear()
        current_config = tracked_configs[i]
        t = i * dt  # Current time
        
        # Plot the static environment and current configuration
        plot_environment(obstacles, current_config, ax=ax, robot_color='blue', label='Current')
        
        # Add dynamic obstacles if requested
        if dynamic_obstacles:
            dynamic_obs, dynamic_vels = create_dynamic_obstacles(t, num_points=50)
            for obs, vel in zip(dynamic_obs, dynamic_vels):
                ax.fill(obs[:, 0], obs[:, 1], color='purple', alpha=0.5)
                ax.scatter(obs[:, 0], obs[:, 1], color='purple', s=1, alpha=0.8)
        
        # Plot reference configuration
        plot_environment(obstacles, reference_configs[i], ax=ax, robot_color='green', plot_obstacles=False, label='Reference', robot_alpha=0.5)
        
        # Plot goal position if provided
        if goal_pos is not None:
            ax.plot(goal_pos[0], goal_pos[1], '*', color='red', markersize=15, label='Goal')
            # Add a small circle around the goal to show the target region
            #circle = plt.Circle((goal_pos[0], goal_pos[1]), 0.1, color='purple', fill=False, linestyle='--')
            #ax.add_artist(circle)
        
        # ax.set_title(f'Time: {t:.2f}s')
        ax.legend(fontsize=24)
        
        # Convert plot to RGB array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    if src_dir:
        print("Saving animation...")
        output_path = os.path.join(src_dir, 'figures/robot_animation.mp4')
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Animation saved as '{output_path}'")
    
    return frames

def forward_kinematics(joint_angles):
    # Convert input to torch tensor if it's not already
    if not isinstance(joint_angles, torch.Tensor):
        joint_angles = torch.tensor(joint_angles, dtype=torch.float32)
    
    joint_angles = joint_angles.detach().cpu().numpy()
    x,y = forward_kinematics_chain_from_array(joint_angles)
    return x, y

def main():

    # Create sample obstacles with point cloud observations
    num_points = 50  # You can adjust this value
    obstacles = create_obstacles(num_points)

    # Create a sample arm configuration (5 joints)
    arm_angles = np.array([0, 0])

    # Plot the environment
    fig, ax = plot_environment(obstacles, arm_angles)
    plt.show()

    print("Sample obstacles created:")
    for i, obstacle in enumerate(obstacles):
        print(f"Obstacle {i+1}: {len(obstacle)} points")

    print(f"\nSample arm configuration: {arm_angles}")
    


if __name__ == "__main__":
    main()
