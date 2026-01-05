import numpy as np
import torch
import math
import sys
import os

from torch.distributions.multivariate_normal import MultivariateNormal
from pathlib import Path

# Add project root to Python path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from xarm_pybullet.self_collision_cdf import SelfCollisionCDF
from xarm_pybullet.models.xarm_model import XArmFK
from xarm_pybullet.robot_sdf import RobotSDF
from xarm_pybullet.robot_cdf import RobotCDF

def compute_robot_distances(robot_state, obstaclesX, robot_sdf, batch_size=50):
    """Compute distances between robot and obstacles using RobotSDF."""
    # Add batch dimension if not present
    if len(robot_state.shape) == 1:
        robot_state = robot_state.unsqueeze(0)      # [1, 6]
    
    # Ensure obstacles are properly shaped [N, 3]
    if len(obstaclesX.shape) > 2:
        obstaclesX = obstaclesX.reshape(-1, 3)      # Flatten to [N, 3]
    
    total_states = robot_state.shape[0]
    device = robot_state.device
    all_sdf_values = []
    
    # Process robot states in batches
    for i in range(0, total_states, batch_size):
        batch_end = min(i + batch_size, total_states)
        batch_robot_state = robot_state[i:batch_end]    # [B, 6]
        
        # Query SDF values for this batch
        batch_obstacles = obstaclesX.unsqueeze(0).expand(batch_end - i, obstaclesX.shape[0], 3) # [B, N, 3]
        
        # Query SDF values
        sdf_values = robot_sdf.query_sdf(
            batch_obstacles,    # [B, N, 3]
            batch_robot_state   # [B, 6]
        )                       # [B, N]
        
        all_sdf_values.append(sdf_values)

    return torch.cat(all_sdf_values, dim=0) 

def compute_robot_cdf_min_distance( robot_state, obstaclesX, robot_cdf, self_collision_cdf=None, safety_margin=0.02):
    """
    Vectorized CDF-based distance for each robot configuration.
    robot_state: [B, 6] or [6]
    obstaclesX: [N, 3]
    Returns: [B]
    """
    # Add batch dim if needed
    if robot_state.dim() == 1:
        robot_state = robot_state.unsqueeze(0)  # [1, 6]

    # Flatten obstacles if needed
    if obstaclesX.dim() > 2:
        obstaclesX = obstaclesX.reshape(-1, 3)  # [N, 3]

    device = robot_cdf.device
    robot_state = robot_state.to(device)
    obstaclesX = obstaclesX.to(device)

    B = robot_state.shape[0]
    safety_margin = float(safety_margin)

    with torch.no_grad():
        # Workspace CDF: [B, N]
        batch_points = obstaclesX.unsqueeze(0).expand(B, -1, -1)                           # [B, N, 3]
        workspace_cdf = robot_cdf.query_cdf(
            points=batch_points,
            joint_angles=robot_state,
            return_gradients=False
        )                                                                                  # [B, N]
        workspace_min = workspace_cdf.min(dim=1)[0]                                        # [B]

        if self_collision_cdf is not None:
            self_vals = self_collision_cdf.query_cdf(
                robot_state,
                return_gradients=False
            )                                                                              # [B, ...]
            if self_vals.dim() == 1:
                self_min = self_vals
            elif self_vals.dim() == 2:
                self_min = self_vals.min(dim=1)[0]
            else:
                self_min = self_vals.view(B, -1).min(dim=1)[0]

            combined = torch.min(workspace_min - safety_margin, self_min)                  # [B]
        else:
            combined = workspace_min - safety_margin                                       # [B]

    return combined

def setup_mppi_controller(
    robot_sdf,
    robot_cdf,
    self_collision_cdf=None,
    robot_n=6,
    input_size=6,
    initial_horizon=10,
    samples=100,
    control_bound=3.0,
    dt=0.05,
    u_guess=None,
    use_GPU=True,
    costs_lambda=0.03,
    cost_goal_coeff=50.0,
    cost_safety_coeff=0.4,
    cost_collision_coeff=100.0,
    cost_perturbation_coeff=0.02,
    action_smoothing=0.5,
    noise_sigma=None,
    goal_in_joint_space=False
):
    device = 'cuda' if use_GPU and torch.cuda.is_available() else 'cpu'

    if goal_in_joint_space: 
        pass
    else: # Goal is in workspace, we need FK to compute EE position
        # FK model 
        fk_model = XArmFK(device=device)
        # Base offset 
        base_offset = torch.tensor([-0.6, 0.0, 0.625], device=device)

    # Noise parameters
    if noise_sigma is None:
        noise_sigma = 2 * torch.eye(input_size, device=device)
    else:
        noise_sigma = torch.tensor(noise_sigma, device=device, dtype=torch.float32)
    noise_mu = torch.zeros(input_size, device=device)
    noise_dist = MultivariateNormal(noise_mu, covariance_matrix=noise_sigma)
    control_cov_inv = torch.inverse(noise_sigma)
    
    # Nominal control sequence U: [H, D]
    if u_guess is not None:
        U = u_guess.to(device)
    else:
        U = torch.zeros((initial_horizon, input_size), device=device)
    
    # Precompute constants on device
    cost_goal_coeff_t = torch.tensor(cost_goal_coeff, device=device, dtype=torch.float32)
    cost_safety_coeff_t = torch.tensor(cost_safety_coeff, device=device, dtype=torch.float32)
    cost_collision_coeff_t = torch.tensor(cost_collision_coeff, device=device, dtype=torch.float32)
    cost_perturbation_coeff_t = torch.tensor(cost_perturbation_coeff, device=device, dtype=torch.float32)

    def robot_dynamics_step(state, input_):
        return state + input_ * dt
    
    def weighted_sum(U, perturbation, costs):
        costs = costs - costs.min()
        costs = costs / (costs.max() + 1e-8)
        weights = torch.exp(-1.0/costs_lambda * costs)
        normalization_factor = weights.sum() + 1e-8
        weights = weights.view(-1, 1, 1)
        
        # Add action smoothing while keeping the original weighting scheme
        weighted_perturbation = (perturbation * weights).sum(dim=0)
        new_U = U + weighted_perturbation / normalization_factor
        return (1.0 - action_smoothing) * U + action_smoothing * new_U
    
    def compute_rollout_costs(key, U, init_state, goal, obstaclesX, safety_margin, batch_size=50):
        """MPPI rollout with smaller batches"""        
        # Make sure everything is on the right device
        U = U.to(device)
        init_state = init_state.to(device)
        goal = goal.to(device)
        obstaclesX = obstaclesX.to(device)

        with torch.no_grad():
            # Sample noise: [N, H, D]
            perturbation = noise_dist.sample((samples, initial_horizon)).detach()
            perturbation = torch.clamp(perturbation, -1., 1.)
            
            # Add perturbation to nominal control sequence: [N, H, D]
            U_expanded = U.unsqueeze(0)                                                         # [1, H, D] 
            perturbed_control = U_expanded + perturbation                                       # [N, H, D]
            perturbed_control = torch.clamp(perturbed_control, -control_bound, control_bound)
            perturbation = perturbed_control - U_expanded                                       # [N, H, D]
            
            costs = torch.empty(samples, device=device)
            idx = 0
            
            for i in range(0, samples, batch_size):
                batch_end = min(i + batch_size, samples)
                batch_size_i = batch_end - i
                
                # init_state_batch: [b, D]
                init_state_batch = init_state.unsqueeze(0).expand(batch_size_i, -1)     # [b, D]

                # u_batch: [b, H, D]
                u_batch = perturbed_control[i:batch_end, :, :]                          # [b, H, D]

                # q(t) = q0 + dt * cumsum(u_batch, dim=1), no loop over horizon
                q_batch = init_state_batch.unsqueeze(1) + torch.cumsum(u_batch * dt, dim=1)  # [b, H, D]
                q_flat = q_batch.reshape(-1, robot_n)                   # [b*H, D]

                # ---- Goal Cost ----
                if goal_in_joint_space:
                    # Goal is in joint space, compute distance directly
                    goal_dist = torch.norm(q_batch - goal.view(1, 1, robot_n), dim=-1)  # [b, H]
                else:
                    # Gaol is in workspace, compute EE position and distance
                    ee_flat = fk_model.fkine(q_flat)[:, -1]                 # [b*H, 3]
                    ee_flat_world = ee_flat + base_offset.unsqueeze(0)      # [b*H, 3]
                    ee_world = ee_flat_world.reshape(batch_size_i, initial_horizon, 3)  
                    goal_dist = torch.norm(ee_world - goal.view(1, 1, 3), dim=-1)  # [b, H]

                # ---- Safety Cost CDF ----
                # For each configuration, compute the CDF-based distance
                min_cdf_dist = compute_robot_cdf_min_distance(q_flat, obstaclesX, robot_cdf, self_collision_cdf, safety_margin)     # [b*H]
                min_cdf_dist = min_cdf_dist.reshape(batch_size_i, initial_horizon)                                                  # [b, H]
                safety_cost = cost_safety_coeff_t / torch.clamp(min_cdf_dist, min=0.01)                                             # [b, H]

                # ---- Collision Cost CDF ----
                collision_mask = (min_cdf_dist < 0.01)                                  # [b, H] boolean
                any_collision = collision_mask.any(dim=1)                               # [b] boolean: True if any step collided
                collision_cost = cost_collision_coeff_t * any_collision.float()         # [b]

                # ---- Control Cost ----
                U_batch = U.unsqueeze(0).expand(batch_size_i, -1, -1)                   # [b, H, D]
                tmp = torch.matmul(U_batch, control_cov_inv)                            # [b, H, D] @ [D, D] -> [b, H, D]
                ctrl_cost_bh = (tmp * perturbation[i:batch_end]).sum(dim=-1)            # [b, H]
                ctrl_cost = cost_perturbation_coeff_t * ctrl_cost_bh.sum(dim=1)         # [b]

                # ---- Total cost for this batch ----
                batch_costs = (cost_goal_coeff_t * goal_dist.sum(dim=1)) + (safety_cost.sum(dim=1)) + collision_cost + ctrl_cost  

                costs[idx:idx+batch_size_i] = batch_costs.detach()
                idx += batch_size_i

            # MPPI update for U 
            U_new = weighted_sum(U, perturbation, costs)    # [H, D]

            # Compute a nominal final trajectory for logging: [H+1, D]
            q_seq = init_state.unsqueeze(0) + torch.cumsum(U_new * dt, dim=0)                   # [H, D]
            states_final = torch.cat([init_state.unsqueeze(0), q_seq], dim=0).transpose(0, 1)   # [D, H+1]

            action = U_new[0].reshape(-1, 1)                        # [D, 1]
            U_next = torch.cat([U_new[1:], U_new[-1:]], dim=0)      # [H, D]
            
            # Clear all intermediate results
            del perturbation, perturbed_control, costs
        
        return states_final, action, U_next
    
    return compute_rollout_costs

def test_mppi():
    """ Test MPPI controller with RobotSDF """
    print("\n=== Starting MPPI Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store successful actions
    successful_actions = []
    
    try:
        # Initialize RobotSDF
        robot_sdf = RobotSDF(device)
        fk_model = XArmFK(device=device)

        # Initialize RobotCDF 
        robot_cdf = RobotCDF(device)
        self_collision_cdf = SelfCollisionCDF(device)
        
        # Initial and goal states
        initial_state = torch.zeros(6, device=device)
        goal_state = torch.tensor([0.3961982, -0.3110882, 0.5596867], device=device, dtype=torch.float32) + torch.tensor([-0.6, 0.0, 0.625], device=device)
        
        # End-effector offset
        ee_offset = torch.tensor([-0.6, 0.0, 0.625], device=device)

        print("\nInitial Configuration:")
        print(f"Initial joint state: {initial_state}")
        print(f"Goal state (world frame): {goal_state}")
        
        # Create test obstacles
        t = torch.linspace(0, 2*math.pi, 50, device=device)
        phi = torch.linspace(0, math.pi, 20, device=device)
        t, phi = torch.meshgrid(t, phi, indexing='ij')
        
        x = 0.3 + 0.2 * torch.sin(phi) * torch.cos(t)
        y = 0.0 + 0.2 * torch.sin(phi) * torch.sin(t)
        z = 0.5 + 0.2 * torch.cos(phi)
        obstacles = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        
        # Setup MPPI controller
        mppi = setup_mppi_controller(
            robot_sdf=robot_sdf,
            robot_cdf=robot_cdf,
            self_collision_cdf=self_collision_cdf,
            use_GPU=(device=='cuda'),
            samples=300,
            initial_horizon=10,
            dt = 0.05,
        )
        
        # Initialize control sequence
        U = torch.zeros((10, 6), device=device)
        current_state = initial_state.clone()
        
        print("\nStarting MPPI optimization...")
        
        for epoch in range(100):            
            # Run one iteration of MPPI
            states_final, action, U = mppi(
                key=None,
                U=U,
                init_state=current_state,
                goal=goal_state,
                obstaclesX=obstacles,
                safety_margin=0.02
            )
            
            # Store the action
            successful_actions.append(action.cpu().numpy())
            
            # Update current state
            with torch.no_grad():
                current_state = current_state + action.squeeze() * 0.05
            
            # FK to get EE position
            ee_pos = fk_model.fkine(current_state.unsqueeze(0))[:, -1] + ee_offset
            distance_to_goal = torch.norm(goal_state - ee_pos.squeeze())
            
            # Compute distance to obstacles
            distances = compute_robot_distances(current_state.unsqueeze(0), obstacles, robot_sdf)
            min_distance = distances.min().item()

            # Compute CDF-based distance
            cdf_distance = compute_robot_cdf_min_distance(current_state.unsqueeze(0), obstacles, robot_cdf, self_collision_cdf=self_collision_cdf, safety_margin=0.02)
            
            print(f"Epoch {epoch + 1}/100")
            print(f"Distance to goal: {distance_to_goal.item():.4f}")
            print(f"Distance to obstacle: {min_distance:.4f}")
            print(f"CDF-based distance: {cdf_distance.item():.4f}")
            print(f"action: {action}")
            print("---")
            
            if distance_to_goal < 0.01:
                print("Reached goal!")
                # Save successful actions to file
                np.save('successful_mppi_actions.npy', np.array(successful_actions))
                break
    
        return True
        
    except Exception as e:
        print(f"MPPI test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mppi()
