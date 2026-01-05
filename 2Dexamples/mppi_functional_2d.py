import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal

from robot_sdf import RobotSDF

def forward_kinematics_2d(joint_angles, link_length=2.0):
    """
    Compute 2D forward kinematics for end-effector position
    Args:
        joint_angles: [B, 2] tensor of joint angles
        link_length: length of each link (default 2.0)
    Returns:
        end_pos: [B, 2] tensor of end-effector positions
    """
    batch_size = joint_angles.shape[0]
    device = joint_angles.device
    
    # First joint position
    angle1 = joint_angles[..., 0]
    joint2_pos = torch.stack([
        link_length * torch.cos(angle1),
        link_length * torch.sin(angle1)
    ], dim=-1)
    
    # End effector position
    angle2 = angle1 + joint_angles[..., 1]
    end_pos = joint2_pos + torch.stack([
        link_length * torch.cos(angle2),
        link_length * torch.sin(angle2)
    ], dim=-1)
    
    return end_pos

def compute_robot_cdf_distances(robot_state, obstaclesX, robot_cdf, batch_size=50):
    """
    Compute CDF distances between robot and obstacles using RobotCDF.
    
    Args:
        robot_state: [B, 2] or [2] tensor of joint angles
        obstaclesX: [N, 2] tensor of obstacle points
        robot_cdf: RobotCDF instance
        batch_size: batch size for processing
    Returns:
        cdf_values: [B, N] tensor of CDF values (minimum across all obstacles)
    """
    # Add batch dimension if not present
    if len(robot_state.shape) == 1:
        robot_state = robot_state.unsqueeze(0)
    
    # Ensure obstacles are properly shaped [N, 2]
    if len(obstaclesX.shape) > 2:
        obstaclesX = obstaclesX.reshape(-1, 2)
    
    # Get device from robot_cdf (which is where the model lives)
    cdf_device = robot_cdf.device
    
    # Ensure inputs are on the correct device (these are already on device, but check anyway)
    # Most of the time these are already on the correct device, so .to() is fast
    robot_state = robot_state.to(cdf_device)
    obstaclesX = obstaclesX.to(cdf_device)
    
    total_states = robot_state.shape[0]
    all_cdf_values = []
    
    # Process robot states in batches
    for i in range(0, total_states, batch_size):
        batch_end = min(i + batch_size, total_states)
        batch_robot_state = robot_state[i:batch_end]
        
        # Query CDF values for this batch
        # expand() creates a view, so it's very fast
        batch_obstacles = obstaclesX.unsqueeze(0).expand(batch_end - i, obstaclesX.shape[0], 2)
        
        # Query CDF values - this is the main computation
        cdf_values = robot_cdf.query_cdf(
            points=batch_obstacles,  # [B, N, 2]
            joint_angles=batch_robot_state,  # [B, 2]
            return_gradients=False
        )  # Returns [B, N]
        
        all_cdf_values.append(cdf_values)
    
    # torch.cat() can be slow if there are many small tensors
    return torch.cat(all_cdf_values, dim=0)

def setup_mppi_controller_2d(
    robot_cdf,
    robot_n=2,
    input_size=2,
    initial_horizon=20,
    samples=200,
    control_bound=2.0,
    dt=0.02,
    u_guess=None,
    use_GPU=True,
    costs_lambda=0.03,
    cost_goal_coeff=50.0,
    cost_safety_coeff=0.4,
    cost_perturbation_coeff=0.02,
    action_smoothing=0.5,
    noise_sigma=None
):
    """
    Setup MPPI controller for 2D robot
    
    Args:
        robot_cdf: RobotCDF instance for safety cost
        robot_n: number of joints (2 for 2D)
        input_size: control input size (2)
        initial_horizon: planning horizon
        samples: number of MPPI samples
        control_bound: maximum control input
        dt: time step
        u_guess: initial guess for control sequence
        use_GPU: whether to use GPU
        costs_lambda: temperature parameter for MPPI
        cost_goal_coeff: weight for goal cost
        cost_safety_coeff: weight for safety cost
        cost_perturbation_coeff: weight for control cost
        action_smoothing: smoothing factor for control updates
        noise_sigma: noise covariance matrix
    """
    # Get device from robot_cdf (where the model actually lives)
    cdf_device = robot_cdf.device
    device = cdf_device  # Use CDF device for all operations
    
    # Initialize noise parameters on CDF device
    # Scale noise variance relative to control bound for better exploration
    # Use a fraction of control_bound so perturbations can meaningfully explore
    if noise_sigma is None:
        # Scale noise to be a reasonable fraction of control bound (e.g., 30-50%)
        noise_scale = 0.4 * control_bound  # 40% of control bound
        noise_sigma = (noise_scale ** 2) * torch.eye(input_size, device=cdf_device)
    else:
        noise_sigma = noise_sigma.to(cdf_device) if hasattr(noise_sigma, 'to') else noise_sigma
    noise_mu = torch.zeros(input_size, device=cdf_device)
    noise_dist = MultivariateNormal(noise_mu, covariance_matrix=noise_sigma)
    control_cov_inv = torch.inverse(noise_sigma)
    
    if u_guess is not None:
        U = u_guess.to(cdf_device)
    else:
        U = 0.0 * torch.ones((initial_horizon, input_size), device=cdf_device)
    
    def robot_dynamics_step(state, input_):
        """Simple dynamics: q_next = q_current + u * dt"""
        return state + input_ * dt
    
    def weighted_sum(U, perturbation, costs):
        """Weighted sum for MPPI update"""
        costs = costs - costs.min()
        costs = costs / (costs.max() + 1e-8)  # Avoid division by zero
        weights = torch.exp(-1.0 / costs_lambda * costs)
        normalization_factor = weights.sum()
        weights = weights.view(-1, 1, 1)
        
        # Add action smoothing while keeping the original weighting scheme
        weighted_perturbation = (perturbation * weights).sum(dim=0)
        new_U = U + weighted_perturbation / (normalization_factor + 1e-8)
        return (1.0 - action_smoothing) * U + action_smoothing * new_U
    
    def compute_rollout_costs(U, init_state, goal_pos, obstaclesX, safety_margin, batch_size=10):
        """
        MPPI rollout with cost computation
        
        Args:
            U: nominal control sequence [horizon, 2]
            init_state: initial joint angles [2]
            goal_pos: goal end-effector position [2]
            obstaclesX: obstacle points [N, 2]
            safety_margin: safety margin for CDF
            batch_size: batch size for processing
        
        Returns:
            states_final: final trajectory [2, horizon+1]
            action: first control action [2, 1]
            U_next: updated control sequence [horizon, 2]
            rollout_trajectories: sampled rollout trajectories for visualization [num_vis_samples, horizon+1, 2]
        """
        # No need for detailed timing - just return results
        
        # Get the actual device from robot_cdf (where the model lives)
        cdf_device = robot_cdf.device
        
        # Device transfers and tensor creation
        U = U.to(cdf_device)
        init_state = init_state.to(cdf_device)
        goal_pos = goal_pos.to(cdf_device)
        obstaclesX = obstaclesX.to(cdf_device)
        
        # Convert cost coefficients to tensors on CDF device
        cost_goal_coeff_tensor = torch.tensor(cost_goal_coeff, device=cdf_device, dtype=torch.float32)
        cost_safety_coeff_tensor = torch.tensor(cost_safety_coeff, device=cdf_device, dtype=torch.float32)
        cost_perturbation_coeff_tensor = torch.tensor(cost_perturbation_coeff, device=cdf_device, dtype=torch.float32)
        safety_margin_tensor = torch.tensor(safety_margin, device=cdf_device, dtype=torch.float32)
        
        # Update device variable for noise distribution and other operations
        device = cdf_device
        
        # Ensure noise distribution is on the correct device
        noise_mu = torch.zeros(input_size, device=cdf_device)
        noise_sigma_device = noise_sigma.to(cdf_device) if hasattr(noise_sigma, 'to') else noise_sigma
        noise_dist_device = MultivariateNormal(noise_mu, covariance_matrix=noise_sigma_device)
        control_cov_inv_device = torch.inverse(noise_sigma_device)
        
        # Sample noise using distribution
        perturbation = noise_dist_device.sample((samples, initial_horizon)).detach()
        perturbation = torch.clamp(perturbation, -5., 5.)
        
        # Add perturbation to nominal control sequence
        perturbed_control = U.unsqueeze(0) + perturbation
        # Only clamp the final control (not the perturbation itself) to respect control bounds
        perturbed_control = torch.clamp(perturbed_control, -control_bound, control_bound)
        # Recompute perturbation after clamping to ensure it matches the actual control
        perturbation = perturbed_control - U.unsqueeze(0)
        
        # Process samples in smaller batches
        all_costs = []
        all_trajectories = []  # Store all trajectories for visualization
        
        for i in range(0, samples, batch_size):
            batch_end = min(i + batch_size, samples)
            batch_size_i = batch_end - i

            # vectorize the rollout over the horizon
            current_state = init_state + torch.cumsum(perturbed_control[i:batch_end, ...], dim=-2) * dt # [batch_size_i, horizon, 2]

            # CDF queries
            current_state_flat = current_state.reshape(-1, 2)  # Reshape for CDF
            cdf_values_all_points = compute_robot_cdf_distances(current_state_flat, obstaclesX, robot_cdf, batch_size=200) # [batch_size_i * horizon, num_objects]
            cdf_values = torch.min(cdf_values_all_points, dim=-1)[0].reshape(current_state.shape[:-1])                              # [batch_size_i, horizon]

            safety_penalty = torch.where(
                cdf_values < safety_margin_tensor,
                cost_safety_coeff_tensor / (cdf_values + 0.01),     # Penalty when too close
                torch.zeros_like(cdf_values)                        # No penalty when safe                
            ) # [batch_size_i, horizon]

            # Forward kinematics
            ee_positions = forward_kinematics_2d(current_state)  # [batch_size_i, horiozn, 2]
            goal_dist = torch.norm(ee_positions - goal_pos, dim=-1)  # [batch_size_i, horizon]

            costs = torch.sum(cost_goal_coeff_tensor * goal_dist + safety_penalty, dim=-1) # [batch_size_i]
            
            all_costs.append(costs.detach())  # Detach costs
            # Keep trajectories on GPU - only move to CPU at the end (much faster!)
            all_trajectories.append(current_state.detach())  # Store for visualization (on GPU)
        
        # Combine results
        costs = torch.cat(all_costs)
        all_trajectories = torch.cat(all_trajectories, dim=0)  # [samples, horizon+1, 2]
        
        # Update nominal control sequence
        U_new = weighted_sum(U, perturbation, costs)
        
        # Compute final trajectory
        with torch.no_grad():
            states_final = torch.zeros((robot_n, initial_horizon+1), device=device)
            states_final[:, 0] = init_state
            for t in range(initial_horizon):
                states_final[:, t+1] = robot_dynamics_step(states_final[:, t], U_new[t])
        
        action = U_new[0].reshape(-1, 1)
        U_next = torch.cat([U_new[1:], U_new[-1:]], dim=0)
        
        # Select subset of rollouts for visualization (10 samples)
        num_vis_samples = min(10, samples)
        vis_indices = torch.linspace(0, samples-1, num_vis_samples, dtype=torch.long)
        rollout_trajectories = all_trajectories[vis_indices]  # [num_vis_samples, horizon+1, 2]
        # Only move to CPU at the very end (single transfer is much faster than many small ones)
        rollout_trajectories = rollout_trajectories.cpu()  # Move to CPU once for visualization
        
        # Clear all intermediate results
        del perturbation, perturbed_control, all_costs, costs, all_trajectories
        # Don't clear cache here - it's very slow and not needed
        
        return states_final, action, U_next, rollout_trajectories
    
    return compute_rollout_costs

