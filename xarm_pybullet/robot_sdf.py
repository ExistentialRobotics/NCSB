import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
import numpy as np
from sdf_training.network import SDFNetwork
from models.xarm6_differentiable_fk import fk_xarm6_torch, fk_xarm6_torch_batch

class RobotSDF:
    def __init__(self, device='cuda', with_gripper=True):
        self.device = device
        self.with_gripper = with_gripper
        # Start from link 1 (skip base link 0)
        self.num_links = 7 if with_gripper else 6
        
        # Load all trained SDF models
        self.models = self._load_sdf_models()
        
        # Load scaling parameters
        data_dir = project_root / "data" / "sdf_data"
        self.offsets = []
        self.scales = []
        # Loading links (6 movable arm links + optional gripper), starting from link 1
        for i in range(1, self.num_links + 1):  # Note: i now starts from 1
            if i < 7:
                data = np.load(data_dir / f"link_{i}_link{i}.npz")
            else:
                data = np.load(data_dir / f"link_{i}_gripper.npz")
            # Explicitly convert to float32
            self.offsets.append(torch.from_numpy(data['original_center']).float().to(device))
            self.scales.append(torch.tensor(data['original_scale'], dtype=torch.float32).to(device))
        
        self.offsets = torch.stack(self.offsets)  # [num_links, 3]
        self.scales = torch.stack(self.scales)     # [num_links]
    
    def _load_sdf_models(self):
        """Load trained models for all links including optional gripper"""
        models = []
        model_dir = project_root / "trained_models"
        # Loading models (6 movable arm links + optional gripper), starting from link 1
        for i in range(1, self.num_links + 1):  # Note: i now starts from 1
            model_path = model_dir / f"link_{i}" / "best_model.pt"
            checkpoint = torch.load(model_path)
            
            model = SDFNetwork().to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        return models
    

    def query_sdf(self, points, joint_angles, return_link_id=False, return_gradients=False, gradient_method='analytic'):
        """
        Query SDF for points given joint angles
        Args:
            gradient_method: 'finite_diff', 'analytic', or 'both'
        """
        points = points.to(dtype=torch.float32)
        joint_angles = joint_angles.to(dtype=torch.float32)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        if return_gradients and (gradient_method == 'analytic' or gradient_method == 'both'):
            joint_angles.requires_grad_(True)
        
        # Get transforms for each link (now only for movable links)
        transforms_list = []
        for b in range(B):
            # Get transforms from FK function
            link_transforms = fk_xarm6_torch(joint_angles[b], with_gripper=self.with_gripper)
            
            # If with_gripper, use Link 6's transform for the gripper (skip the last transform)
            if self.with_gripper:
                # Use all transforms except the last one, and repeat Link 6's transform
                transforms = link_transforms[:-1] + [link_transforms[-2]]
            else:
                transforms = link_transforms
                
            transforms_list.append(transforms)
        
        transforms = torch.stack([torch.stack(t) for t in transforms_list])
        
        # Transform points to each link's local frame (no need for base transform)
        points_expanded = points.unsqueeze(1).expand(-1, self.num_links, -1, -1)
        inv_transforms = torch.linalg.inv(transforms)
        local_points = torch.einsum('bkij,bknj->bkni', inv_transforms, 
                                  torch.cat([points_expanded, 
                                           torch.ones_like(points_expanded[...,:1])], dim=-1))
        local_points = local_points[...,:3]
        
        # Scale points
        scaled_points = ((local_points - self.offsets.view(1, self.num_links, 1, 3)) / 
                        self.scales.view(1, self.num_links, 1, 1)).to(dtype=torch.float32)
        
        # Query each link's SDF
        sdf_values = []
        for i, model in enumerate(self.models):
            points_i = scaled_points[:,i].reshape(B*N, 3)
            with torch.set_grad_enabled(True):
                sdf = model(points_i).reshape(B, N, 1)
            sdf_values.append(sdf)
        
        sdf_values = torch.stack(sdf_values, dim=1)  # [B, num_links, N, 1]
        sdf_values = sdf_values.squeeze(-1) * self.scales.view(1, self.num_links, 1)
        
        # Get minimum distance and corresponding link
        min_sdf, link_ids = torch.min(sdf_values, dim=1)  # [B, N]
        
        result = [min_sdf]
        
        if return_link_id:
            result.append(link_ids)
        
        if return_gradients:
            if gradient_method == 'analytic':
                #print("\nComputing analytic gradients...")
                analytic_grads = torch.zeros((B, N, 6), device=self.device)
                for b in range(B):
                    for i in range(N):
                        try:
                            grad = torch.autograd.grad(min_sdf[b, i], 
                                                     joint_angles,
                                                     retain_graph=True,
                                                     allow_unused=True)[0]
                            if grad is not None:
                                analytic_grads[b, i] = grad[b]
                        except Exception as e:
                            print(f"Warning: Gradient computation failed for batch {b}, point {i}: {e}")
                
                # print(f"Analytic gradients shape: {analytic_grads.shape}")
                # print(f"Analytic gradients mean: {analytic_grads.mean():.6f}")
                # print(f"Analytic gradients std: {analytic_grads.std():.6f}")
                result.append(analytic_grads)
                
            elif gradient_method == 'finite_diff':
                #print("\nComputing finite difference gradients...")
                delta = 0.001
                finite_diff_grads = []
                
                # For each joint
                for i in range(6):
                    perturbed_angles = joint_angles.clone()
                    perturbed_angles[:, i] += delta
                    perturbed_sdf = self.query_sdf(points, perturbed_angles)[0]
                    grad = (perturbed_sdf - min_sdf) / delta
                    finite_diff_grads.append(grad)
                
                finite_diff_grads = torch.stack(finite_diff_grads, dim=2)  # [B, N, 6]
                # print(f"Finite diff gradients shape: {finite_diff_grads.shape}")
                # print(f"Finite diff gradients mean: {finite_diff_grads.mean():.6f}")
                # print(f"Finite diff gradients std: {finite_diff_grads.std():.6f}")
                result.append(finite_diff_grads)
                
            elif gradient_method == 'both':
                # Compute both and return as dict
                analytic = self.query_sdf(points, joint_angles, return_link_id=False, 
                                        return_gradients=True, gradient_method='analytic')[1]
                finite = self.query_sdf(points, joint_angles, return_link_id=False, 
                                      return_gradients=True, gradient_method='finite_diff')[1]
                
                print("\nGradient comparison:")
                diff = analytic - finite
                print(f"Mean difference: {diff.abs().mean():.6f}")
                print(f"Max difference: {diff.abs().max():.6f}")
                print(f"Relative difference: {(diff.abs() / (analytic.abs() + 1e-6)).mean():.6f}")
                
                result.append({'analytic': analytic, 'finite_diff': finite})
        
        return tuple(result) if len(result) > 1 else result[0]
    
    def query_sdf_batch(self, points, joint_angles, return_link_id=False, return_gradients=False, gradient_method='analytic'):
        """
        Optimized SDF query implementation
        """
        points = points.to(dtype=torch.float32)
        joint_angles = joint_angles.to(dtype=torch.float32)
        
        B = joint_angles.shape[0]
        N = points.shape[1] if points.dim() > 2 else 1
        
        if return_gradients and gradient_method == 'analytic':
            joint_angles.requires_grad_(True)  # Ensure gradients are enabled
        
        # Vectorized FK computation
        transforms = fk_xarm6_torch_batch(joint_angles, with_gripper=self.with_gripper)
        
        if self.with_gripper:
            transforms = torch.cat([transforms[:, :-1], transforms[:, -2:-1]], dim=1)
        
        # Vectorized point transformation
        points_expanded = points.unsqueeze(1).expand(-1, self.num_links, -1, -1)
        points_homogeneous = torch.cat([
            points_expanded, 
            torch.ones_like(points_expanded[...,:1])
        ], dim=-1)
        
        inv_transforms = torch.linalg.inv(transforms)
        local_points = torch.matmul(
            inv_transforms.view(B * self.num_links, 4, 4),
            points_homogeneous.view(B * self.num_links, N, 4).transpose(1, 2)
        ).transpose(1, 2).view(B, self.num_links, N, 4)[..., :3]
        
        scaled_points = ((local_points - self.offsets.view(1, self.num_links, 1, 3)) / 
                        self.scales.view(1, self.num_links, 1, 1))
        
        # Batch SDF computation
        sdf_values = torch.zeros((B, self.num_links, N), device=self.device)
        batch_size = 10000
        
        for i, model in enumerate(self.models):
            points_i = scaled_points[:, i].reshape(-1, 3)
            
            if len(points_i) > batch_size:
                sdf_i = []
                for j in range(0, len(points_i), batch_size):
                    batch = points_i[j:j + batch_size]
                    sdf_batch = model(batch)
                    sdf_i.append(sdf_batch)
                sdf_i = torch.cat(sdf_i)
            else:
                sdf_i = model(points_i)
            
            sdf_values[:, i] = sdf_i.reshape(B, N) * self.scales[i]
        
        # Get minimum distance and corresponding link
        min_sdf, link_ids = torch.min(sdf_values, dim=1)
        
        # Map gripper indices (6) to Link 6 indices (5)
        if return_link_id:
            link_ids = torch.where(link_ids == 6, torch.tensor(5, device=link_ids.device), link_ids)
        
        result = [min_sdf]
        if return_link_id:
            result.append(link_ids)
        
        if return_gradients:
            if gradient_method == 'analytic':
                # Vectorized gradient computation
                analytic_grads = torch.zeros((B, N, 6), device=self.device)
                
                # Enable gradients for joint angles if not already enabled
                if not joint_angles.requires_grad:
                    joint_angles.requires_grad_(True)
                
                # Compute gradients for each point individually
                for b in range(B):
                    for n in range(N):
                        # Create computation graph for this specific point
                        point_sdf = min_sdf[b, n]
                        
                        try:
                            grad = torch.autograd.grad(
                                point_sdf, 
                                joint_angles,
                                retain_graph=True,
                                create_graph=True,  # Enable higher-order gradients if needed
                                allow_unused=True
                            )[0]
                            
                            if grad is not None:
                                analytic_grads[b, n] = grad[b]
                        except Exception as e:
                            print(f"Warning: Gradient computation failed for batch {b}, point {n}: {e}")
                            # Keep zero gradient for failed computations
                
                result.append(analytic_grads)
                
            elif gradient_method == 'finite_diff':
                # Vectorized finite differences
                delta = 0.001
                finite_diff_grads = torch.zeros((B, N, 6), device=self.device)
                
                for i in range(6):
                    perturbed_angles = joint_angles.clone()
                    perturbed_angles[:, i] += delta
                    perturbed_sdf = self.query_sdf(points, perturbed_angles)[0]
                    finite_diff_grads[..., i] = (perturbed_sdf - min_sdf) / delta
                
                result.append(finite_diff_grads)
        
        return tuple(result) if len(result) > 1 else result[0]
    

if __name__ == "__main__":
    device = 'cuda'
    robot_sdf = RobotSDF(device)
    
    # Fixed joint configuration
    joint_angles = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    print(f"\nTesting with joint angles: {joint_angles.cpu().numpy()[0]}")
    
    # Test points (in world frame)
    test_points = torch.tensor([
        # Point near base
        [0.0, 0.5, 0.1],
        # Point near middle (around link 2-3)
        [0.3, 0.2, 0.0],
        # Point near end-effector
        [0.3, 0.3, 0.5]
    ], device=device).unsqueeze(0)  # [1, 3, 3]
    
    print("\nTest points:")
    for i, p in enumerate(test_points[0]):
        print(f"Point {i}: {p.cpu().numpy()}")
    
    # Query SDF with both gradient methods separately
    print("\nComputing analytic gradients...")
    results_analytic = robot_sdf.query_sdf_batch(
        test_points, 
        joint_angles, 
        return_link_id=True, 
        return_gradients=True,
        gradient_method='analytic'
    )
    
    print("\nComputing finite difference gradients...")
    results_finite = robot_sdf.query_sdf_batch(
        test_points, 
        joint_angles, 
        return_link_id=True, 
        return_gradients=True,
        gradient_method='finite_diff'
    )
    
    sdf, link_ids, analytic_grads = results_analytic
    _, _, finite_grads = results_finite
    
    # Print results for each point
    print("\nResults:")
    for i in range(len(test_points[0])):
        print(f"\nPoint {i}:")
        print(f"SDF value: {sdf[0,i].item():.6f}")
        print(f"Closest link: {link_ids[0,i].item()}")
        print(f"Analytic gradients: {analytic_grads[0,i].detach().cpu().numpy()}")
        print(f"Finite diff gradients: {finite_grads[0,i].detach().cpu().numpy()}")
        print(f"Gradient difference: {(analytic_grads[0,i] - finite_grads[0,i]).abs().max().item():.6f}")
    
    # Test differentiability

    # Speed comparison test
    print("\n=== Speed Comparison Test ===")
    device = 'cuda'
    robot_sdf = RobotSDF(device)
    
    # Test different combinations of N configs and M points
    test_cases = [
        (1, 1000),    # 1 config, 1000 points
        (1, 10000),   # 1 config, 10000 points
        (100, 1000),  # 100 configs, 1000 points
        (1000, 100),  # 1000 configs, 100 points
        (10000, 1)
    ]

    import time

    for num_configs, num_points in test_cases:
        print(f"\nTesting with {num_configs} configs and {num_points} points:")
        
        # Generate random configurations and points
        configs = torch.rand((num_configs, 6), device=device) * 2 * torch.pi - torch.pi
        points = torch.rand((num_configs, num_points, 3), device=device) * 2 - 1  # Range [-1, 1]

        # Test query_sdf
        torch.cuda.synchronize()
        start = time.time()
        regular_result = robot_sdf.query_sdf(points, configs)
        torch.cuda.synchronize()
        regular_time = time.time() - start

        # Test query_sdf_batch
        torch.cuda.synchronize()
        start = time.time()
        batch_result = robot_sdf.query_sdf_batch(points, configs)
        torch.cuda.synchronize()
        batch_time = time.time() - start

        # Compare results
        max_diff = (regular_result - batch_result).abs().max().item()
        mean_diff = (regular_result - batch_result).abs().mean().item()
        
        print(f"query_sdf time:       {regular_time:.4f} seconds")
        print(f"query_sdf_batch time: {batch_time:.4f} seconds")
        print(f"Speedup factor:       {regular_time/batch_time:.2f}x")
        print(f"Max difference:       {max_diff:.8f}")
        print(f"Mean difference:      {mean_diff:.8f}")
    
    
    print("\n=== Testing Gradients in Batch vs Regular ===")
    device = 'cuda'
    robot_sdf = RobotSDF(device)
    
    # Create test inputs
    configs = torch.zeros((2, 6), device=device, requires_grad=True)  # 2 configs
    points = torch.ones((2, 3, 3), device=device)  # 2 batches, 3 points each
    
    # Test regular version
    sdf_regular = robot_sdf.query_sdf(points, configs)
    loss_regular = sdf_regular.sum()
    loss_regular.backward()
    grad_regular = configs.grad.clone()
    
    # Reset gradients
    configs.grad = None
    
    # Test batch version
    sdf_batch = robot_sdf.query_sdf_batch(points, configs)
    loss_batch = sdf_batch.sum()
    loss_batch.backward()
    grad_batch = configs.grad.clone()
    
    print("\nGradient comparison:")
    print(f"Regular gradients:\n{grad_regular}")
    print(f"Batch gradients:\n{grad_batch}")
    print(f"Max difference: {(grad_regular - grad_batch).abs().max().item():.8f}")
    
    
    