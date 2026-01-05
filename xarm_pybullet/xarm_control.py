import pybullet as p
import torch
import numpy as np
import imageio
import argparse

from xarm_planning import XArmSDFVisualizer
from typing import List, Tuple
#from control.clf_cbf_qp import ClfCbfQpController

# Add project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


from control.pd_cbf_qp import ClfCbfQpController
from control.pd_dro_cbf import ClfCbfDrccpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
from control.reference_governor_bezier import BezierReferenceGovernor

import time
import os
import pickle

class XArmController:
    def __init__(self, planner: XArmSDFVisualizer, control_type='pd'):
        self.planner = planner
        self.env = planner.env
        self.robot_id = self.env.robot_id
        self.device = planner.device
        self.control_type = control_type
        
        # Control parameters
        self.max_velocity = 2.0  # Matches test case

        self.num_samples = 10

        self.cbf_rate = 1.0
        
        # Initialize controllers
        if control_type == 'pd':
            self.pd_controller = PDController(kp=0.8, kd=0.1, control_limits=self.max_velocity)

        elif control_type == 'clf_cbf':
            self.pd_controller = PDController(kp=0.8, kd=0.1, control_limits=self.max_velocity)
            self.clf_cbf_controller = ClfCbfQpController(
            p1=1e0,    # Control effort penalty
            p2=1e0,    # CLF slack variable penalty
            clf_rate=1.0,
            cbf_rate=self.cbf_rate,
            safety_margin=0.1
            )
        # Add DR-CLF-CBF controller
        elif control_type == 'clf_dro_cbf':

            self.pd_controller = PDController(kp=0.7, kd=0.1, control_limits=self.max_velocity)
            self.clf_dro_cbf_controller = ClfCbfDrccpController(
            p1=1e0,    # Control effort penalty
            p2=1e0,    # CLF slack variable penalty
            clf_rate=1.0,
            cbf_rate=self.cbf_rate,
            wasserstein_r=0.025,
            epsilon=0.1,
            num_samples=self.num_samples,  # Number of CBF samples to use
            state_dim=6,                #xArm6
            control_limits=self.max_velocity
        )

    def compute_control(self, current_joints: torch.Tensor, 
                       target_joints: torch.Tensor,
                       current_vel: torch.Tensor) -> torch.Tensor:
        """Compute control based on selected controller type"""
        if self.control_type == 'pd':
            velocity_cmd = self.pd_controller.compute_control(
                current_joints,
                target_joints,
                current_vel
            )
            # print('PD velocity_cmd', velocity_cmd)
            
        elif self.control_type == 'clf_cbf':
            # Ensure we need gradients
            current_joints.requires_grad_(True)
            
            # Get point clouds with new format
            point_data = self.env.get_full_point_cloud()
            static_points = point_data['static']['points']
            dynamic_points = point_data['dynamic']['points']
            dynamic_velocities = point_data['dynamic']['velocities']
            
            # Combine all points for CDF query
            all_points = torch.cat([static_points, dynamic_points], dim=0).unsqueeze(0)
            all_points.requires_grad_(True)
            
            # Get CDF values for all points
            h_values = self.planner.robot_cdf.query_cdf(
                points=all_points,
                joint_angles=current_joints.unsqueeze(0),
                return_gradients=False
            )
            
            # Create combined_h = h + dh/dt for ranking
            combined_h = h_values[0].detach().clone() * self.cbf_rate
            
            # Compute dh/dt for dynamic obstacles and add to combined_h
            num_static = static_points.shape[0]
            for idx in range(num_static, all_points.shape[1]):
                h_val = h_values[0, idx]
                
                # Compute dh/dt
                h_val.backward(retain_graph=True)
                dh_dp = all_points.grad[0, idx].clone()
                all_points.grad.zero_()
                current_joints.grad.zero_()
                
                dyn_idx = idx - num_static
                dp_dt = dynamic_velocities[dyn_idx].to(torch.float64)
                dh_dt = torch.dot(dh_dp, dp_dt).item()
                
                combined_h[idx] += dh_dt
            
            # Get the minimum value based on combined_h
            min_idx = torch.argmin(combined_h)
            h = h_values[0, min_idx] 
            
            # Print which type of obstacle is active
            # if min_idx < static_points.shape[0]:
            #     print(f"Static obstacle active - h value: {h.item():.3f}")
            # else:
            #     dyn_idx = min_idx - static_points.shape[0]
            #     print(f"Dynamic obstacle {dyn_idx} active - h value: {h.item():.3f}")
            
            # Compute final gradients for the minimum point
            h.backward(retain_graph=True)
            dh_dp = all_points.grad[0, min_idx].clone()
            all_points.grad.zero_()
            
            h.backward()
            dh_dtheta = current_joints.grad.clone()
            current_joints.grad.zero_()
            
            # Compute dh_dt for the minimum point
            if min_idx < static_points.shape[0]:
                dp_dt = torch.zeros(3, device=self.device, dtype=torch.float64)
            else:
                dyn_idx = min_idx - static_points.shape[0]
                dp_dt = dynamic_velocities[dyn_idx].to(torch.float64)
            
            dh_dt = torch.dot(dh_dp, dp_dt).item()

            u_nominal = self.pd_controller.compute_control(
                current_joints.detach().cpu().numpy(),
                target_joints.detach().cpu().numpy(),
                current_vel.detach().cpu().numpy()
            )

            # Generate control input using CLF-CBF-QP
            velocity_cmd = self.clf_cbf_controller.generate_controller(
                current_joints.detach().cpu().numpy(),
                target_joints.cpu().numpy(),
                h.item(),
                dh_dtheta.cpu().numpy(),
                dh_dt,
                u_nominal
            )
            velocity_cmd = torch.tensor(velocity_cmd, device=self.device)
        
        elif self.control_type == 'clf_dro_cbf':
            current_joints.requires_grad_(True)
            
            # Get point clouds
            point_data = self.env.get_full_point_cloud()
            static_points = point_data['static']['points']
            dynamic_points = point_data['dynamic']['points']
            dynamic_velocities = point_data['dynamic']['velocities']
            
            # Combine all points for CDF query
            all_points = torch.cat([static_points, dynamic_points], dim=0).unsqueeze(0)
            all_points.requires_grad_(True)
            
            # Get CDF values for all points
            h_values = self.planner.robot_cdf.query_cdf(
                points=all_points,
                joint_angles=current_joints.unsqueeze(0),
                return_gradients=False
            )
            
            # print('min_h_values', h_values.min().item())
            
            # Create combined_h = h + dh/dt for ranking
            combined_h = h_values[0].detach().clone() * self.cbf_rate  # Detach to prevent gradient accumulation
            
            # Compute dh/dt for dynamic obstacles and add to combined_h

            start_time = time.time()
            num_static = static_points.shape[0]
            for idx in range(num_static, all_points.shape[1]):
                h_val = h_values[0, idx]  # Use original h_values
                
                # Compute dh/dt
                h_val.backward(retain_graph=True)
                dh_dp = all_points.grad[0, idx].clone()
                all_points.grad.zero_()
                current_joints.grad.zero_()  # Also clear joint grads
                
                dyn_idx = idx - num_static
                dp_dt = dynamic_velocities[dyn_idx].to(torch.float64)
                dh_dt = torch.dot(dh_dp, dp_dt).item()
                
                # Add dh/dt to combined_h
                combined_h[idx] += dh_dt

            # print('dh/dt time taken', time.time() - start_time)
            
            # Get the k smallest values based on combined_h
            k = self.num_samples
            top_k_values, top_k_indices = torch.topk(combined_h, k, largest=False)
            
            # Count and print which types of obstacles are active
            num_static = static_points.shape[0]
            static_active = sum(1 for idx in top_k_indices if idx < num_static)
            dynamic_active = sum(1 for idx in top_k_indices if idx >= num_static)
            
            # print(f"Active obstacles in top {k} samples:")
            # if static_active > 0:
            #     print(f"  - {static_active} static obstacles")
            # if dynamic_active > 0:
            #     print(f"  - {dynamic_active} dynamic obstacles")
            # print(f"Minimum h value: {h_values[0, top_k_indices[0]].item():.3f}")
            
            # Now compute gradients and collect samples for selected points
            h_grads = []
            h_samples = []
            dh_dt_samples = []
            
            for idx in top_k_indices:
                h_val = h_values[0, idx]  # Get original h_value
                
                # First compute dh/dp for dh/dt
                h_val.backward(retain_graph=True)
                dh_dp = all_points.grad[0, idx].clone()
                all_points.grad.zero_()
                
                # Then compute dh/dtheta
                h_val.backward(retain_graph=True)
                h_grads.append(current_joints.grad.clone().cpu().numpy())
                current_joints.grad.zero_()
                
                h_samples.append(h_val.item())
                
                # Compute dh/dt
                if idx < static_points.shape[0]:
                    dp_dt = torch.zeros(3, dtype=torch.float64, device=self.device)
                else:
                    dyn_idx = idx - static_points.shape[0]
                    dp_dt = dynamic_velocities[dyn_idx].to(torch.float64)
                
                dh_dt = torch.dot(dh_dp, dp_dt).item()
                dh_dt_samples.append(dh_dt * 1.0)
                # if static obstacles: 
                # dh_dt_samples.append(0.0)
            
            # Convert to numpy arrays
            h_samples_np = np.array(h_samples)
            h_grads_np = np.stack(h_grads)
            dh_dt_samples_np = np.array(dh_dt_samples)

            u_nominal = self.pd_controller.compute_control(
                current_joints.detach().cpu().numpy(),
                target_joints.detach().cpu().numpy(),
                current_vel.detach().cpu().numpy()
            )
            
            # Generate control input using DR-CLF-CBF-QP with correct dh_dt values
            velocity_cmd = self.clf_dro_cbf_controller.generate_controller(
                current_joints.detach().cpu().numpy(),
                target_joints.cpu().numpy(),
                h_samples_np,
                h_grads_np,
                dh_dt_samples_np,  # Now using correct dh_dt values
                u_nominal
            )
            velocity_cmd = torch.tensor(velocity_cmd, device=self.device)
        
        # Clip velocities
        velocity_cmd = torch.clamp(velocity_cmd, -self.max_velocity, self.max_velocity)
        
        return velocity_cmd

    def downsample_trajectory(self, trajectory: List[np.ndarray], num_points: int = 10) -> np.ndarray:
        """Downsample trajectory to specified number of points"""
        if len(trajectory) <= num_points:
            return np.array(trajectory)
        
        # Create indices for evenly spaced samples
        indices = np.linspace(10, len(trajectory)-1, num_points, dtype=int)
        return np.array(trajectory)[indices]

    def execute_trajectory(self, trajectory_data: dict, dt: float = 0.02, use_bezier: bool = True, 
                          save_video: bool = False) -> Tuple[List[float], List[np.ndarray], bool]:
        """
        Execute trajectory using velocity control.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
                - waypoints: Discrete waypoints
                - bezier_curves: Bezier curve segments (optional)
                - times: Time parameterization
            dt: Time step for control
            use_bezier: Whether to use Bezier curves (True) or discrete waypoints (False)
            save_video: Whether to save execution video (default: False)
        
        Returns:
            Tuple containing:
            - List of distances to goal
            - List of executed joint configurations
            - Boolean indicating if execution was safe (no collisions)
        """
        goal_distances = []
        executed_configs = []
        min_sdf_values = []
        frames = []
        
        SAFETY_THRESHOLD = 0.03
        JOINT_THRESHOLD = 0.05  # Maximum allowed joint difference for considering goal reached
        is_safe = True
        
        if save_video:
            width = 1920
            height = 1080
            fps = 30
            frame_interval = max(1, int((1/dt) / fps))
        
        step = 0
        initial_state = self.planner.get_current_joint_states().cpu().numpy()
        
        # Get the final planned configuration
        if use_bezier and 'bezier_curves' in trajectory_data:
            final_planned_config = trajectory_data['waypoints'][-1]
        else:
            final_planned_config = trajectory_data[-1]
        
        # Initialize appropriate governor
        if use_bezier and 'bezier_curves' in trajectory_data:
            print("Using Bezier-based reference governor")
            governor = BezierReferenceGovernor(
                initial_state=initial_state,
                trajectory_data=trajectory_data,
                dt=dt, 
                k=0.5,
                zeta=15
            )
        else:
            print("Using discrete waypoint-based reference governor")
            waypoints = (trajectory_data['waypoints'] if isinstance(trajectory_data, dict) 
                        else trajectory_data)
            governor = ReferenceGovernor(
                initial_state=initial_state,
                path_configs=waypoints,
                dt=dt
            )

        while True:
            # Get current joint states and velocities
            current_joints = self.planner.get_current_joint_states()
            # current_velocities = torch.tensor([
            #     p.getJointState(self.robot_id, i+1)[1]
            #     for i in range(6)
            # ], device=self.device)

            current_velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

            executed_configs.append(current_joints.cpu().numpy())

            # Check safety using robot SDF
            points_robot = self.planner.points_robot.unsqueeze(0)  # Add batch dimension
            sdf_values = self.planner.robot_sdf.query_sdf(
                points=points_robot,
                joint_angles=current_joints.unsqueeze(0),
                return_gradients=False
            )
            min_sdf = sdf_values.min().item()
            min_sdf_values.append(min_sdf)
            
            # Update safety flag
            if min_sdf < SAFETY_THRESHOLD:
                is_safe = False
                print(f"Warning: Safety violation detected! Min SDF: {min_sdf:.4f}")

            # Get reference based on governor type
            if use_bezier and 'bezier_curves' in trajectory_data:
                reference_joints, s, reference_vel = governor.update(current_joints.cpu().numpy())
            else:
                reference_joints, s = governor.update(current_joints.cpu().numpy())
            
            target_joints = torch.tensor(reference_joints, device=self.device)
            
            # Compute and apply control
            velocity_cmd = self.compute_control(
                current_joints, 
                target_joints,
                current_velocities
            )

            # Apply velocity commands
            for i in range(6):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i+1,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=velocity_cmd[i].item(),
                    force=100
                )

            # Step simulation
            for _ in range(int(dt / (1/200.0))):
                self.env.step()

            # Record metrics
            current_ee_pos = self.planner.get_ee_position(current_joints)
            goal_dist = torch.norm(self.planner.goal - current_ee_pos.squeeze())
            goal_distances.append(goal_dist.item())

            # Capture frame if video saving is enabled
            if save_video and step % frame_interval == 0:
                frames.append(self.planner._capture_frame(step, 1000, width, height))
            step += 1

            current_config = current_joints.detach().cpu().numpy()
            joint_differences = np.abs(current_config - final_planned_config)
            max_joint_diff = np.max(joint_differences)

            # print('max_joint_diff', max_joint_diff)

            # Check if we're close to the end of the trajectory
            if s > 0.99 and max_joint_diff < JOINT_THRESHOLD:
                print("Goal reached successfully!")
                break

            if step > 600:
                print("Trajectory completed but goal not reached!")
                break

            time.sleep(dt)

        if save_video and frames:
            print("Saving execution video...")
            imageio.mimsave(f'execution_{self.control_type}.mp4', frames, fps=fps)

        return goal_distances, np.array(executed_configs), is_safe

if __name__ == "__main__":
    def parse_goal_list(goal_str):
        """Convert string representation of list to float list"""
        try:
            goal_str = goal_str.strip('[]')
            return [float(x) for x in goal_str.split(',')]
        except:
            raise argparse.ArgumentTypeError("Goal must be a list of 3 floats: [x,y,z]")
    
    parser = argparse.ArgumentParser(description='xArm Control Demo')

    # example pos: [0.7, 0.1, 0.66], [0.25, 0.6, 0.6], [0.22, 0.27, 0.66]
    parser.add_argument('--goal', type=parse_goal_list, default=[0.78, 0.24, 0.37],
                      help='Goal position as list [x,y,z]')
    parser.add_argument('--planner', type=str, default='bubble',
                      choices=['bubble', 'sdf_rrt', 'cdf_rrt', 'rrt_connect'],
                      help='Planner type')
    parser.add_argument('--controller', type=str, default='clf_dro_cbf',
                      choices=['pd', 'clf_cbf', 'clf_dro_cbf'],
                      help='Controller type')
    parser.add_argument('--dynamic', type=str, default='True',
                      choices=['True', 'False'],
                      help='Enable dynamic obstacles')
    parser.add_argument('--gui', type=str, default='True',
                      choices=['True', 'False'],
                      help='Enable PyBullet GUI')
    parser.add_argument('--seed', type=int, default=10,
                      help='Random seed')
    parser.add_argument('--early_termination', type=str, default='True',
                      choices=['True', 'False'],
                      help='Stop planning after finding first valid path')
    
    args = parser.parse_args()
    
    # Convert goal to tensor
    goal_pos = torch.tensor(args.goal, device='cuda')
    
    try:
        # Initialize planner
        planner = XArmSDFVisualizer(
            goal_pos, 
            use_gui=args.gui == 'True',
            planner_type=args.planner,
            seed=args.seed,
            dynamic_obstacles=args.dynamic == 'True',
            early_termination=args.early_termination == 'True'
        )
        
        # Plan trajectory
        trajectory_whole = planner.run_demo(execute_trajectory=False)
        
        if trajectory_whole is not None:
            # Initialize controller
            controller = XArmController(
                planner, 
                control_type=args.controller
            )
            
            # Execute trajectory
            if args.planner == 'bubble':
                distances, executed_configs, is_safe = controller.execute_trajectory(
                    trajectory_whole, 
                    use_bezier=True, 
                    save_video=True
                )
            else:
                distances, executed_configs, is_safe = controller.execute_trajectory(
                    trajectory_whole
                )
        else:
            print("Failed to generate trajectory!")
            
    finally:
        # Ensure proper cleanup
        if 'controller' in locals() and hasattr(controller, 'planner'):
            controller.planner.env.close() 