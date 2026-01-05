import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts
import matplotlib.pyplot as plt
import pybullet as p
import numpy as np
import argparse
import cProfile
import imageio
import pstats
import torch
import time
import json
import sys
import os

from typing import Optional, Union, List, Tuple, Dict
from datetime import datetime
from pathlib import Path

from self_collision_cdf import SelfCollisionCDF
from xarm_sim_env import XArmEnvironment
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from planner.mppi_functional import setup_mppi_controller
from utils.find_goal_pose import find_goal_configuration
from planner.bubble_planner import BubblePlanner
from models.xarm_model import XArmFK
from dataclasses import dataclass

def plot_distances(goal_distances, estimated_obstacle_distances, obst_radius, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(time_steps, goal_distances, color='red', linestyle=':', linewidth=3, label='Distance to Goal')
    
    if estimated_obstacle_distances.ndim == 1:
        plt.plot(time_steps, estimated_obstacle_distances, linewidth=3, label='Distance to Obstacle')
    else:
        if estimated_obstacle_distances.ndim == 3:
            estimated_obstacle_distances = estimated_obstacle_distances.squeeze(1)
        
        num_obstacles = estimated_obstacle_distances.shape[1]
        for i in range(num_obstacles):
            if i == 0:
                plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3, label='Distance to Obstacles')
            else:
                plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3)
    
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Distance', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

@dataclass
class PlanningMetrics:
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float

class XArmSDFVisualizer:
    def __init__(self, ee_goal, use_gui=True, initial_horizon=8, planner_type='bubble', seed=5, dynamic_obstacles=False, use_pybullet_inverse=True, early_termination=False):
        """
        Initialize XArmSDFVisualizer
        
        Args:
            ee_goal: End-effector goal position
            use_gui: Whether to use GUI visualization
            initial_horizon: Initial horizon for MPPI
            planner_type: Type of planner to use ('bubble', 'cdf_rrt', 'sdf_rrt', 'mppi')
            seed: Random seed for reproducibility
        """
        # Set random seeds first
        print(f"\nInitializing with random seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Store seed for later use in planning
        self.seed = seed
        self.dynamic_obstacles = dynamic_obstacles  
        
        # Initialize environment
        self.env = XArmEnvironment(gui=use_gui, add_dynamic_obstacles=dynamic_obstacles)
        self.physics_client = self.env.client
        self.robot_id = self.env.robot_id
        self.use_gui = use_gui

        # Precompute static point cloud for planning
        self.points_robot = self.env.get_static_point_cloud(downsample=True)
        self.points_robot = self.points_robot.to(dtype=torch.float32)
        while self.points_robot.dim() > 2:
            self.points_robot = self.points_robot.squeeze(0)

        # Robot base transform
        self.base_pos = torch.tensor(self.env.robot_base_pos, device='cuda')

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.use_pybullet_inverse = use_pybullet_inverse  # Default to random sampling
        self.early_termination = early_termination
        
        # Initialize robot FK, SDF, CDF models
        self.robot_fk = XArmFK(device=self.device)
        self.robot_sdf = RobotSDF(device=self.device)
        self.robot_cdf = RobotCDF(device=self.device)
        self.self_collision_cdf = SelfCollisionCDF(device=self.device)

        # ---- Metrics ----
        self.num_collision_checks_env = 0       # will count per-config environment collision checks (CDF evaluations)
        self.num_collision_checks_self = 0      # will count per-config self-collision CDF checks
        self.path_length = 0.0                  # you already use this for path length

        # Wrap CDF calls so that we count "one check per configuration"
        def _wrap_robot_cdf(cdf_module):
            original = cdf_module.query_cdf

            def wrapped(points, joint_angles, *args, **kwargs):
                # joint_angles is [B, 6]
                if isinstance(joint_angles, torch.Tensor):
                    B = joint_angles.shape[0]
                else:
                    B = len(joint_angles)
                self.num_collision_checks_env += B
                return original(points, joint_angles, *args, **kwargs)
            
            cdf_module.query_cdf = wrapped

        def _wrap_self_collision_cdf(cdf_module):
            original = cdf_module.query_cdf

            def wrapped(joint_angles, *args, **kwargs):
                if isinstance(joint_angles, torch.Tensor):
                    B = joint_angles.shape[0]
                else:
                    B = len(joint_angles)
                self.num_collision_checks_self += B
                return original(joint_angles, *args, **kwargs)
            
            cdf_module.query_cdf = wrapped

        _wrap_robot_cdf(self.robot_cdf)
        _wrap_self_collision_cdf(self.self_collision_cdf)

        # Store goal pos (in task space)
        self.goal = ee_goal + self.base_pos
        self.initial_horizon = initial_horizon
        
        self.planner_type = planner_type

        if planner_type == 'mppi':
            self.mppi_dt = 0.05
            self.mppi_horizon = 50
            self.mppi_samples = 100
            self.goal_in_joint_space = True 
            self.bootstrap_iters = 50  
            self.ik_failed = False
            self.goal_q = None

            # Finding joint-space goal configuration for MPPI via IK
            if self.goal_in_joint_space:
                goal_in_base = self.goal - self.base_pos
                goal_configs_with_metrics = self._find_goal_configuration(goal_in_base, threshold=0.08)

                if not goal_configs_with_metrics:
                    print("[XarmPlanning] MPPI: failed to find any IK solutions for the goal position.")
                    self.ik_failed = True
                else:
                    goal_configs = [config for config, _, _, _ in goal_configs_with_metrics]
                    task_distances = [task_dist for _, task_dist, _, _ in goal_configs_with_metrics]

                    # Pick solution closest to some nominal configuration (e.g., all zeros)
                    best_idx = np.argmin(task_distances) if task_distances is not None else 0
                    self.goal_q = torch.tensor(goal_configs[best_idx],device=self.device,dtype=torch.float32)
                    print(f"[XarmPlanning] Using joint configuration: {self.goal_q.cpu().numpy()}")

            # Initialize MPPI controller (6-DoF)
            self.mppi_controller = setup_mppi_controller(
                robot_sdf=self.robot_sdf,
                robot_cdf=self.robot_cdf,
                self_collision_cdf=self.self_collision_cdf,
                robot_n=6,
                input_size=6,
                initial_horizon=self.mppi_horizon,
                samples=self.mppi_samples,                            
                control_bound=2.0,
                dt=self.mppi_dt,
                use_GPU=(self.device == 'cuda'),
                costs_lambda=0.03,
                cost_goal_coeff=50.0,
                cost_safety_coeff=2.5,
                cost_collision_coeff=1000.0,
                cost_perturbation_coeff=0.02,
                action_smoothing=0.5,
                goal_in_joint_space=self.goal_in_joint_space
            )
            print(f"[XarmPlanning] Initialized MPPI controller with horizon {self.mppi_horizon} and {self.mppi_samples} samples")

        elif planner_type in ['cdf_rrt', 'sdf_rrt', 'lazy_rrt', 'rrt_connect']:
            # Import OMPLRRTPlanner 
            from planner.rrt_ompl import OMPLRRTPlanner

            # Initialize OMPL RRT planner with seed
            joint_limits = (
                self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                self.robot_fk.joint_limits[:, 1].cpu().numpy()
            )
            self.rrt_planner = OMPLRRTPlanner(
                robot_sdf=self.robot_sdf,
                robot_cdf=self.robot_cdf,
                robot_fk=self.robot_fk,
                joint_limits=joint_limits,
                planner_type=planner_type,  # Pass planner type to use appropriate collision checker
                check_resolution=0.002,
                device=self.device,
                seed=seed
            )
        elif planner_type in ['bubble', 'bubble_connect']:
            # Initialize Bubble Planner with both CDFs
            self.bubble_planner = BubblePlanner(
                robot_cdf=self.robot_cdf,
                self_collision_cdf=self.self_collision_cdf,
                joint_limits=(
                    self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                    self.robot_fk.joint_limits[:, 1].cpu().numpy()
                ),
                device=self.device,
                max_samples=1e4,
                seed=seed,
                planner_type=planner_type,
                early_termination=self.early_termination
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        
        # Create goal marker
        self.env.create_goal_marker(self.goal)
        
        # Add end-effector marker
        # self.ee_marker = p.createVisualShape(
        #     p.GEOM_SPHERE,
        #     radius=0.02,
        #     rgbaColor=[1, 0, 0, 0.7]  # Red, semi-transparent
        # )
        # self.ee_visual = p.createMultiBody(
        #     baseVisualShapeIndex=self.ee_marker,
        #     basePosition=[0, 0, 0],
        #     baseOrientation=[0, 0, 0, 1]
        # )

        # Add IK solver parameters
        # self.ik_iterations = 10000  # Default number of IK iterations
        # self.ik_threshold = 0.05  # Default threshold for IK solutions (in meters)
        # self.max_ik_solutions = 10  # Maximum number of IK solutions to find

        # Add goal configuration attributes
        self.goal_configs = None
        self._found_goal_configs = False

    def set_robot_configuration(self, joint_angles):
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        
        if len(joint_angles.shape) == 2:
            joint_angles = joint_angles[0]
        
        for i in range(6):  # xArm has 6 joints
            p.resetJointState(self.robot_id, i+1, joint_angles[i])

        # link_state = p.getLinkState(self.robot_id, 6)
        # pybullet_pos = link_state[0]

        # print('pybullet_pos', pybullet_pos)

    def get_current_joint_states(self):
        joint_states = []
        for i in range(1, 7):  # xArm has 6 joints, with 1 base joint
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])
        return torch.tensor(joint_states, device=self.device)


    def get_ee_position(self, joint_angles):
        """Get end-effector position in world frame"""
        if not isinstance(joint_angles, torch.Tensor):
            joint_angles = torch.tensor(joint_angles, device=self.device)
        if len(joint_angles.shape) == 1:
            joint_angles = joint_angles.unsqueeze(0)
            
        # Get end-effector position in robot base frame
        ee_pos_local = self.robot_fk.fkine(joint_angles)[:, -1]
        
        # Debug prints
        # print("Local EE position:", ee_pos_local)
        # print("Base position:", self.base_pos)
        # print("World EE position:", ee_pos_local + self.base_pos)
        
        # # Compare with PyBullet's FK
        # state_id = p.saveState()
        # for i in range(6):
        #     p.resetJointState(self.robot_id, i+1, joint_angles[0][i])
        # link_state = p.getLinkState(self.robot_id, 6)  # 5 is typically the end-effector link
        # pybullet_ee_pos = link_state[0]
        # p.restoreState(state_id)
        # print("PyBullet EE position:", pybullet_ee_pos)
        
        # Use the same transform as in MPPI test
        ee_pos_world = ee_pos_local + self.base_pos

        # print('ee_pos_world', ee_pos_world)
        return ee_pos_world

    def update_ee_marker(self, ee_pos):
        """Update the position of the end-effector marker"""
        if isinstance(ee_pos, torch.Tensor):
            ee_pos = ee_pos.cpu().numpy()
        p.resetBasePositionAndOrientation(
            self.ee_visual,
            ee_pos,
            [0, 0, 0, 1]
        )

    def _find_goal_configuration(self, goal_pos: torch.Tensor, n_samples: int = 1e6, threshold: float = 0.05) -> Optional[List[Tuple[np.ndarray, float, float, float]]]:
        """
        Find valid goal configurations for a given goal position.
        
        Returns:
            Optional[List[Tuple[np.ndarray, float, float, float]]]: 
                List of tuples (config, task_dist, min_sdf, min_cdf) for each valid solution.
                Returns None if no solutions found.
        """
        print(f"\nSearching for goal configurations:")
        print(f"Using random seed: {self.seed}")
        
        if self.use_pybullet_inverse:
            # Use environment's IK solver
            
            # Add IK parameters
            ik_max_iterations = 5000
            ik_threshold = threshold
            ik_max_solutions = 5

            self.env.set_ik_parameters(ik_max_iterations, ik_threshold, ik_max_solutions)
            solutions = self.env.find_ik_solutions(
                target_pos=goal_pos,  # Already in robot base frame
                visualize=True,
                pause_time=1.0,
                seed=self.seed
            )
            
            if not solutions:
                print("Failed to find any valid goal configurations")
                return None
            
            # Compute CDF for each solution and return tuples
            goal_configs_with_metrics = []
            for solution in solutions:
                config = np.array(solution.joint_angles, dtype=np.float32)
                config_tensor = torch.tensor(config, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                # Compute CDF (environment collision)
                min_cdf = 1.0
                if self.robot_cdf is not None:
                    cdf_values = self.robot_cdf.query_cdf(
                        points=self.points_robot.unsqueeze(0),
                        joint_angles=config_tensor,
                        return_gradients=False
                    )
                    min_cdf = cdf_values.min().item()
                
                # Compute self-collision CDF
                min_self_cdf = 1.0
                if self.self_collision_cdf is not None:
                    self_cdf_values = self.self_collision_cdf.query_cdf(
                        joint_angles=config_tensor,
                        return_gradients=False
                    )
                    min_self_cdf = self_cdf_values.min().item()
                
                # Use minimum of both CDFs
                min_cdf_overall = min(min_cdf, min_self_cdf)
                
                goal_configs_with_metrics.append((
                    config,
                    solution.task_dist,
                    solution.min_sdf,
                    min_cdf_overall
                ))
            
            print(f"\nFound {len(goal_configs_with_metrics)} valid goal configurations")
            for i, (config, task_dist, min_sdf, min_cdf) in enumerate(goal_configs_with_metrics):
                print(f"\nSolution {i+1}:")
                print(f"Task distance: {task_dist:.4f}")
                print(f"Min SDF: {min_sdf:.4f}")
                print(f"Min CDF: {min_cdf:.4f}")
                print(f"Joint angles: {config}")
            
            return goal_configs_with_metrics
        else:
            # Use existing random sampling method
            solutions = find_goal_configuration(
                goal_pos=goal_pos,
                points_robot=self.points_robot,
                robot_fk=self.robot_fk,
                robot_sdf=self.robot_sdf,
                robot_cdf=self.robot_cdf,
                n_samples=int(1e6),
                threshold=threshold,
                device=self.device,
                max_solutions=5,
                seed=self.seed
            )
            if not solutions:
                return None
            
            # Solutions already come as tuples (config, distance, min_sdf, min_cdf)
            # Just convert configs to float32
            goal_configs_with_metrics = [
                (config.astype(np.float32), task_dist, min_sdf, min_cdf)
                for config, task_dist, min_sdf, min_cdf in solutions
            ]
            return goal_configs_with_metrics
    
    def _capture_frame(self, step: int, time_steps: int, width: int, height: int) -> np.ndarray:
        """Capture a frame using a rotating camera"""

        # env1 camera parameters
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 1.5],
            distance=1.7,
            yaw=55,  # Rotating camera, -(step / time_steps) * 60 for rotate to left
            pitch=-10,
            roll=0,
            upAxisIndex=2
        )

        # env 2 camera parameters
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 1.5],
            distance=2.1,
            yaw=-40,  # Rotating camera, -(step / time_steps) * 60 for rotate to left
            pitch=-15,
            roll=0,
            upAxisIndex=2
        )

        #         camera_params = {
        #     'target': [0.0, 0.0, 1.5],
        #     'distance': 2.1,
        #     'yaw': -55,
        #     'pitch': -15,
        #     'roll': 0
        # }
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0
        )
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        return np.array(rgb)[:, :, :3]

    def run_demo(self, fps=20, execute_trajectory=True, save_snapshots=False) -> Union[PlanningMetrics, Tuple[List, List]]:
        try:
            print(f"Starting {self.planner_type.upper()} demo...")
            
            # Initialize lists to store metrics
            goal_distances = []
            sdf_distances = []
            cdf_distances = []
            trajectory = None
            dt = 1.0 / fps
            bootstrap_time = 0.0
            success = False
            
            if self.planner_type in ['sdf_rrt', 'cdf_rrt', 'lazy_rrt', 'rrt_connect']:
                # RRT/RRT-Connect planning code
                if not self._found_goal_configs:
                    goal_config_with_metrics = self._find_goal_configuration(self.goal - self.base_pos)
                    if goal_config_with_metrics is None:
                        print("Failed to find valid goal configuration!")
                        return None if not execute_trajectory else (goal_distances, sdf_distances)
                    # Extract just configs for RRT (backward compatible)
                    goal_config = [config for config, _, _, _ in goal_config_with_metrics]
                    self.goal_configs = goal_config
                else:
                    goal_config = self.goal_configs

                # Get current state and plan
                current_state = self.get_current_joint_states()
                result = self.rrt_planner.plan(
                    start_config=current_state.cpu().numpy(),
                    goal_configs=goal_config,
                    obstacle_points=self.points_robot,
                    max_time=100.0,
                    early_termination=self.early_termination
                )
                
                if not result['metrics'].success:
                    print("RRT planning failed!")
                    return None if not execute_trajectory else (goal_distances, sdf_distances)
                
                trajectory = result['waypoints']
                print(f"RRT path found with {len(trajectory)} waypoints")
                
                if not execute_trajectory:
                    return result
                
            elif self.planner_type in ['bubble', 'bubble_connect']:
                # Bubble planning code
                current_state = self.get_current_joint_states().cpu().numpy()
                
                if not self._found_goal_configs:
                    goal_config_with_metrics = self._find_goal_configuration(self.goal - self.base_pos)
                    if goal_config_with_metrics is None:
                        print("Failed to find valid goal configuration!")
                        return None if not execute_trajectory else (goal_distances, sdf_distances)
                    # Extract just configs for bubble planner (backward compatible)
                    goal_config = [config for config, _, _, _ in goal_config_with_metrics]
                    self.goal_configs = goal_config
                else:
                    goal_config = self.goal_configs

                try:
                    trajectory_result = self.bubble_planner.plan(
                        start_config=current_state,
                        goal_configs=goal_config,
                        obstacle_points=self.points_robot
                    )
                    
                    if trajectory_result is None:
                        print("Bubble planning failed!")
                        return None if not execute_trajectory else (goal_distances, sdf_distances)
                    
                    trajectory = trajectory_result['waypoints']
                    print(f"Bubble planner found path with {len(trajectory)} waypoints")
                    
                    if not execute_trajectory:
                        return trajectory_result
                    
                except Exception as e:
                    print(f"Error during bubble planning: {str(e)}")
                    return None if not execute_trajectory else (goal_distances, sdf_distances)
            
            elif self.planner_type == 'mppi':
                        # --- MPPI closed-loop control directly in PyBullet ---
                        print("\nRunning MPPI controller in PyBullet...")
                        
                        # If IK failed earlier, skip MPPI but don't crash; metrics will record failure
                        if getattr(self, "ik_failed", False) or self.goal_q is None:
                            print("MPPI: skipping this run because no valid IK solution was found for the goal.")
                            trajectory = []  # no motion
                        else:
                            current_state = self.get_current_joint_states().to(self.device).float()
                            U = torch.zeros((self.mppi_horizon, 6), device=self.device)

                            full_cloud = self.env.get_full_point_cloud()
                            static_pts = full_cloud["static"]["points"]         #[Ns, 3]
                            dynamics_pts = full_cloud["dynamic"]["points"]      #[Nd, 3]

                            if dynamics_pts.numel() > 0:
                                obstaclesX = torch.cat([static_pts, dynamics_pts], dim=0).to(self.device)
                            else:
                                obstaclesX = static_pts.to(self.device)
                            trajectory = []
                            
                            mppi_dt = getattr(self, 'mppi_dt', dt)
                            max_time = 10.0
                            max_steps = int(max_time / mppi_dt)
                            
                            # Find joint-space goal configuration for MPPI via IK
                            goal_config = self.goal_q
                            goal_workspace = self.goal.to(self.device).float()
                            if self.goal_in_joint_space:
                                goal_tensor = goal_config       # [6]
                            else:
                                goal_tensor = goal_workspace    # [3]

                            # ---- BOOSTSTRAP MPPI FOR {bootstrap_iters} ITERATIONS WITH RANDOM SAMPLING ----
                            print(f"\nBootstrapping MPPI with {self.bootstrap_iters} iterations of random sampling...")
                            timer_start = time.time()
                            for k in range(self.bootstrap_iters):
                                _, _, U = self.mppi_controller( key=None, U=U, init_state=current_state, goal=goal_tensor, obstaclesX=obstaclesX, safety_margin=0.02)
                            bootstrap_time = time.time() - timer_start
                            print(f"Bootstrap completed in {bootstrap_time:.2f} seconds. Starting MPPI iterations...")
                            prev_state = current_state.clone()

                            timer_start = time.time()
                            for step in range(max_steps):                            
                                # Run one MPPI iteration
                                states_final, action, U = self.mppi_controller(
                                    key=None,                 
                                    U=U,
                                    init_state=current_state,
                                    goal=goal_tensor,
                                    obstaclesX=obstaclesX,
                                    safety_margin=0.02,       
                                )
                                
                                # Apply first action to joint state: q_{t+1} = q_t + u_t * dt
                                current_state = current_state + action.squeeze() * mppi_dt
                                
                                # Path length: sum of |Δq| over all joints
                                self.path_length += torch.abs(current_state - prev_state).sum().item()
                                prev_state = current_state.clone()

                                # Log trajectory (numpy for plotting/video)
                                trajectory.append(current_state.detach().cpu().numpy())
                                
                                # Push to PyBullet
                                self.set_robot_configuration(current_state)
                                p.stepSimulation()
                                
                                # ----- Metrics -----
                                # Goal distance in configuration space
                                joint_diff = torch.abs(goal_config - current_state)             # [6]
                                max_joint_diff = torch.max(joint_diff)                          # scalar
                                goal_distances.append(max_joint_diff.detach().cpu().numpy())
                                
                                # Self-collision CDF
                                next_config_tensor = current_state.unsqueeze(0)
                                min_cdf = self.self_collision_cdf.query_cdf(next_config_tensor, return_gradients=False).min().detach().cpu().numpy()
                                cdf_distances.append(min_cdf)

                                # Self-collision SDF
                                sdf_values = self.robot_sdf.query_sdf( points=self.points_robot.unsqueeze(0), joint_angles=current_state.unsqueeze(0), return_gradients=False)
                                min_sdf = sdf_values.min()
                                sdf_distances.append(min_sdf.detach().cpu().numpy())
                                
                                if step % 10 == 0:
                                    goal_dist = torch.norm(goal_config - current_state)
                                    print(f"MPPI step {step}/{max_steps}")
                                    print(f"  Distance to goal: {goal_dist.item():.4f}")
                                    print(f"  Min SDF:          {min_sdf.item():.4f}")
                                    print(f"  Min CDF:          {min_cdf:.4f}")
                                    print("---")
                                
                                # GUI pacing
                                if self.use_gui:
                                    time.sleep(mppi_dt)
                                
                                # Stopping criteria
                                if max_joint_diff < 0.05: 
                                    cdf_min_distance = float(np.min(cdf_distances)) if len(cdf_distances) > 0 else 1.0 
                                    sdf_min_distance = float(np.min(sdf_distances)) if len(sdf_distances) > 0 else 1.0
                                    if cdf_min_distance < 0.0 or sdf_min_distance < 0.0:
                                        print("\nMPPI reached goal but is in self-collision! Stopping with failure.")
                                        break
                                    else:
                                        success = True
                                        print("\nMPPI reached goal!")
                                        break
                            control_time = time.time() - timer_start

                            # Optional pause at final config
                            if self.use_gui:
                                print("\nReached MPPI final configuration. Pausing for 5 seconds...")
                                time.sleep(5.0)

            # Execute trajectory and record metrics
            if trajectory is not None and execute_trajectory and self.planner_type not in ['mppi']:
                print(f"\nExecuting {len(trajectory)} waypoints...")
                
                for traj_idx in range(len(trajectory)):
                    # Update robot state
                    next_config = trajectory[traj_idx]
                    if isinstance(next_config, np.ndarray):
                        next_config = torch.tensor(next_config, device=self.device, dtype=torch.float32)
                    self.set_robot_configuration(next_config)
                    p.stepSimulation()
                    
                    # Record metrics
                    current_ee_pos = self.get_ee_position(next_config)
                    goal_dist = torch.norm(self.goal - current_ee_pos.squeeze())
                    goal_distances.append(goal_dist.detach().cpu().numpy())
                    
                    # Get workspace CDF (SDF)
                    sdf_values = self.robot_sdf.query_sdf(
                        points=self.points_robot.unsqueeze(0),
                        joint_angles=next_config.unsqueeze(0),
                        return_gradients=False
                    )
                    min_sdf = sdf_values.min()
                    sdf_distances.append(min_sdf.detach().cpu().numpy())
                    
                    # Get self-collision CDF - FIX: ensure next_config is a tensor with correct shape
                    next_config_tensor = next_config.unsqueeze(0) if next_config.dim() == 1 else next_config
                    min_self_cdf = self.self_collision_cdf.query_cdf(next_config_tensor, return_gradients=False).min().detach().cpu().numpy()
                    
                    if traj_idx % 10 == 0:
                        print(f"Waypoint {traj_idx}/{len(trajectory)}")
                        print(f"Distance to goal: {goal_dist.item():.4f}")
                        print(f"Minimum SDF value: {min_sdf.item():.4f}")
                        print(f"Minimum self-collision CDF: {min_self_cdf:.4f}")
                        print("---")
                    
                    if self.use_gui:
                        time.sleep(dt)
                
                # Pause at the final configuration for 5 seconds
                if self.use_gui:
                    print("\nReached goal configuration. Pausing for 5 seconds...")
                    time.sleep(5.0)
            
            # Save snapshots, video, and plot results
            if save_snapshots and trajectory is not None:
                self.env.record_trajectory_video(
                    trajectory=trajectory,
                    fps=fps,
                    planner_type=self.planner_type
                )
                
                print("\nPlotting trajectory metrics...")
                try:
                    self.env.plot_trajectory_metrics(
                        goal_distances=np.array(goal_distances),
                        sdf_distances=np.array(sdf_distances),
                        dt=dt,
                        planner_type=self.planner_type
                    )
                except Exception as e:
                    print(f"Warning: Could not plot trajectory metrics: {e}")
                    print("Continuing without plot...")
            
            if getattr(self, "ik_failed", False) or self.goal_q is None:
                print("\nIK failed or goal configuration not found. Skipping metrics and saving.")
            else:
                print("\n=== Planning Metrics ===")
                print(f"Planner type: {self.planner_type}")
                print(f"Success: {success}")
                print(f"Experiment completed at number of steps: {len(trajectory) if trajectory is not None else 0}")
                print(f"Collision checks (per-config environment CDF evals): {self.num_collision_checks_env}")
                print(f"Collision checks (per-config self-collision CDF evals): {self.num_collision_checks_self}")
                print(f"Total collision checks: {self.num_collision_checks_env + self.num_collision_checks_self}")
                print(f"Path length (Σ_j,t |Δq_j,t|):            {self.path_length:.6f} rad")

                if self.planner_type == 'mppi':
                    print(f"Planning time: {bootstrap_time:.2f} seconds (MPPI bootstrap) + {control_time:.2f} seconds (MPPI iterations)")
                else:
                    print("Planning time: N/A (non-MPPI planner)")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metrics_dir = Path("results/metrics")
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics = {
                    "timestamp": timestamp,
                    "planner": self.planner_type,
                    "goal": self.goal.cpu().numpy().tolist(),
                    "goal_q": goal_config.cpu().numpy().tolist() if hasattr(self, 'goal_q') else None,
                    "steps": len(trajectory) if trajectory is not None else 0,
                    "collision_checks_env": self.num_collision_checks_env,
                    "collision_checks_self": self.num_collision_checks_self,
                    "collision_checks_total": self.num_collision_checks_env + self.num_collision_checks_self,
                    "path_length": float(self.path_length),
                    "bootstrap_iters": int(self.bootstrap_iters) if self.planner_type == "mppi" else None,
                    "bootstrap_time": float(bootstrap_time) if self.planner_type == "mppi" else None,
                    "mppi_time": float(control_time) if self.planner_type == "mppi" else None,
                    "success": success,
                    "seed": int(self.seed),
                    "dynamic_obstacles": bool(self.dynamic_obstacles),
                    "use_gui": bool(self.use_gui),
                    "goal_distances": [float(x) for x in goal_distances],
                    "sdf_distances": [float(x) for x in sdf_distances],
                    "cdf_distances": [float(x) for x in cdf_distances],
                    "trajectory": [config.tolist() for config in trajectory] if trajectory is not None else None

                }
                metrics_path = metrics_dir / f"metrics_{self.planner_type}_{timestamp}.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                print(f"\nSaved metrics to: {metrics_path}")

            return goal_distances, sdf_distances
        
        except Exception as e:
            print(f"Error during demo: {str(e)}")
            return None if not execute_trajectory else ([], [])
        
        finally:
            # Only clean up if we're executing the trajectory
            if execute_trajectory and hasattr(self, 'env'):
                print("\nClosing PyBullet environment...")
                self.env.close()

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    def parse_goal_list(goal_str):
        """Convert string representation of list to float list"""
        try:
            # Remove brackets and split by comma
            goal_str = goal_str.strip('[]')
            return [float(x) for x in goal_str.split(',')]
        except:
            raise argparse.ArgumentTypeError("Goal must be a list of 3 floats: [x,y,z]")
    
    parser = argparse.ArgumentParser(description='xArm Planning Demo')
    parser.add_argument('--goal', type=parse_goal_list, default=[0.78, 0.24, 0.37], help='Goal position as list [x,y,z]')
    parser.add_argument('--planner', type=str, default='bubble', choices=['bubble', 'bubble_connect', 'sdf_rrt', 'cdf_rrt', 'lazy_rrt', 'rrt_connect','mppi'], help='Planner type')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--dynamic_obstacles', type=str, default='False', choices=['True', 'False'], help='Enable dynamic obstacles')
    parser.add_argument('--gui', type=str, default='True', choices=['True', 'False'], help='Enable PyBullet GUI')
    parser.add_argument('--early_termination', type=str, default='True', choices=['True', 'False'], help='Stop planning after finding first valid path')
    args = parser.parse_args()
    
    # Validate goal length
    if len(args.goal) != 3:
        raise ValueError("Goal must contain exactly 3 values [x,y,z]")
    
    # Convert goal to tensor
    goal_pos = torch.tensor(args.goal, device='cuda')
    
    visualizer = None
    try:
        visualizer = XArmSDFVisualizer(
            goal_pos, 
            use_gui=args.gui=='True', 
            initial_horizon=12,
            planner_type=args.planner,
            seed=args.seed,
            dynamic_obstacles=args.dynamic_obstacles=='True',
            use_pybullet_inverse=True,
            early_termination=args.early_termination=='True'
        )
        
        visualizer.run_demo(
            fps=20,
            execute_trajectory=True,
            save_snapshots=True,
        )
    
    finally:
        if visualizer is not None and hasattr(visualizer, 'env'):
            try:
                visualizer.env.close()
            except Exception as e:
                print(f"Warning while closing environment: {e}")

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats(50)
