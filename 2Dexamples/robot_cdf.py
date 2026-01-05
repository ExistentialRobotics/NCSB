import sys
from pathlib import Path
import torch
import numpy as np
import time
from typing import List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cdf_training.network import CDFNetwork, CDFNetworkWithDropout

class RobotCDF:
    def __init__(self, device='cuda', use_dropout=True):
        self.device = device
        
        # Load trained CDF model
        self.model = self._load_cdf_model(use_dropout)
        
    def _load_cdf_model(self, use_dropout=True):
        """Load trained CDF model"""
        model_dir = project_root / "trained_models"
        #model_path = model_dir / "best_model_gelu.pth"

        if use_dropout:
        # dropout model
            model_path = model_dir / "dropout_0.1_best_model_gelu.pth"
            model = CDFNetworkWithDropout(input_dims=8, output_dims=1, activation='gelu', dropout_rate=0.1).to(self.device)
        else:
            model_path = model_dir / "best_model_gelu.pth"
            model = CDFNetwork(input_dims=8, output_dims=1, activation='gelu').to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def query_cdf(self, points, joint_angles, return_gradients=False):
        """
        Query CDF for points given joint angles
        Args:
            points: [B, N, 2] tensor of 2D points
            joint_angles: [B, 2] tensor of joint angles
            return_gradients: bool, whether to return gradients w.r.t. joint angles
        Returns:
            cdf_values: [B, N] tensor of CDF values
            gradients: [B, N, 2] tensor of gradients (optional)
        """
        # print("\nDebug - query_cdf:")
        # print(f"Input points shape: {points.shape}")
        # print(f"Input joint_angles shape: {joint_angles.shape}")
        
        points = points.to(dtype=torch.float32, device=self.device)
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        if return_gradients:
            joint_angles = joint_angles.detach().clone()
            joint_angles.requires_grad_(True)
        
        # Remove extra dimensions from joint_angles first
        if len(joint_angles.shape) > 2:
            joint_angles = joint_angles.squeeze(1)
        
        # Prepare input features
        sin_q = torch.sin(joint_angles)  # [B, 2]
        cos_q = torch.cos(joint_angles)  # [B, 2]

        
        # Expand joint features for each point
        joint_features = torch.cat([
            joint_angles.unsqueeze(1).expand(-1, N, -1),  # [B, N, 2]
            sin_q.unsqueeze(1).expand(-1, N, -1),        # [B, N, 2]
            cos_q.unsqueeze(1).expand(-1, N, -1)         # [B, N, 2]
        ], dim=-1)  # [B, N, 6]

        
        # Concatenate with points
        inputs = torch.cat([joint_features, points], dim=-1)  # [B, N, 8]
        
        # Forward pass
        cdf_values = self.model(inputs.reshape(B*N, -1)).reshape(B, N)
        
        if return_gradients:
            gradients = torch.zeros((B, N, 2), device=self.device)
            
            for b in range(B):
                for n in range(N):
                    scalar_output = cdf_values[b, n:n+1]
                    grad_outputs = torch.ones_like(scalar_output)
                    
                    try:
                        grad = torch.autograd.grad(
                            scalar_output,
                            joint_angles,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            retain_graph=True
                        )[0]
                        
                        if grad is not None:
                            gradients[b, n] = grad[b]
                    except Exception as e:
                        print(f"Warning: Gradient computation failed for batch {b}, point {n}: {e}")
            
            return cdf_values, gradients
        
        return cdf_values

    def query_cdf_dropout(self, points, joint_angles, num_samples=10, return_gradients=False):
        """
        Query CDF with Monte Carlo Dropout for uncertainty estimation
        Args:
            points: [B, N, 2] tensor of 2D points
            joint_angles: [B, 2] tensor of joint angles
            num_samples: int, number of forward passes for MC dropout
            return_gradients: bool, whether to return gradients w.r.t. joint angles
        Returns:
            all_cdf_values: [num_samples, B, N] tensor of all CDF values
            all_gradients: [num_samples, B, N, 2] tensor of all gradients (optional)
        """
        # Explicitly enable dropout
        self.model.train()  # Enable dropout
        torch.set_grad_enabled(True)
        
        points = points.to(dtype=torch.float32, device=self.device)
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        # Store multiple predictions
        all_cdf_values = []
        all_gradients = [] if return_gradients else None
        
        for _ in range(num_samples):
            if return_gradients:
                # Create a new copy of joint_angles for each forward pass
                joint_angles_copy = joint_angles.detach().clone()
                joint_angles_copy.requires_grad_(True)
                
                cdf, grad = self.query_cdf(points, joint_angles_copy, return_gradients=True)
                all_cdf_values.append(cdf)
                all_gradients.append(grad)
            else:
                with torch.no_grad():
                    cdf = self.query_cdf(points, joint_angles, return_gradients=False)
                    all_cdf_values.append(cdf)
        
        # Stack the values
        all_cdf_values = torch.stack(all_cdf_values, dim=0)  # [num_samples, B, N]
        
        if return_gradients:
            all_gradients = torch.stack(all_gradients, dim=0)  # [num_samples, B, N, 2]
            
            # Set model back to eval mode after we're done
            self.model.eval()
            return all_cdf_values, all_gradients
        
        # Set model back to eval mode after we're done
        self.model.eval()
        return all_cdf_values

    def test_differentiability(self):
        """Test if the model is differentiable and check eikonal constraint"""
        # Create test inputs
        points = torch.randn(1, 3, 2, device=self.device)  # [batch, num_points, 2]
        joint_angles = torch.zeros(1, 2, device=self.device, requires_grad=True)
        
        # Get CDF values and gradients
        cdf_values, gradients = self.query_cdf(points, joint_angles, return_gradients=True)
        
        # Check gradient norms (eikonal constraint)
        gradient_norms = torch.norm(gradients, dim=-1)
        mean_norm = gradient_norms.mean().item()
        norm_deviation = (gradient_norms - 1.0).abs().mean().item()
        
        print("\nEikonal Constraint Check:")
        print(f"Mean gradient norm: {mean_norm:.4f} (should be close to 1.0)")
        print(f"Mean deviation from 1.0: {norm_deviation:.4f} (should be close to 0.0)")
        
        try:
            loss = cdf_values.sum()
            loss.backward()
            print("\nGradient computation successful!")
            print(f"Joint angles gradient: {joint_angles.grad}")
        except Exception as e:
            print(f"Gradient computation failed: {str(e)}")

    def benchmark_inference_time(self, q_sizes: List[int], p_sizes: List[int], num_trials: int = 100) -> dict:
        """
        Benchmark inference time for different numbers of configurations and workspace points.
        
        Args:
            q_sizes: List of different numbers of configurations to test
            p_sizes: List of different numbers of workspace points to test
            num_trials: Number of trials to average over
        
        Returns:
            Dictionary containing timing results and statistics
        """
        results = {}
        
        for q_size in q_sizes:
            for p_size in p_sizes:
                # Generate random test data
                joint_angles = torch.rand(q_size, 2, device=self.device) * 2 * np.pi - np.pi  # [-π, π]
                points = torch.rand(q_size, p_size, 2, device=self.device) * 2 - 1  # [-1, 1]
                
                # Warm-up runs
                for _ in range(10):
                    _ = self.query_cdf(points, joint_angles)
                
                # Timing runs
                torch.cuda.synchronize()  # Ensure GPU operations are completed
                times = []
                
                for _ in range(num_trials):
                    start = time.perf_counter()
                    _ = self.query_cdf(points, joint_angles)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
                
                # Calculate statistics
                times = np.array(times)
                results[(q_size, p_size)] = {
                    'mean': np.mean(times) * 1000,  # Convert to milliseconds
                    'std': np.std(times) * 1000
                }
        
        return results

if __name__ == "__main__":
    device = 'cuda'
    robot_cdf = RobotCDF(device=device, use_dropout=False)
    
    # Define test sizes
    q_sizes = [1, 10, 100, 1000]
    p_sizes = [1, 10, 100, 1000]
    
    # Run benchmark
    print("\nRunning inference time benchmark...")
    results = robot_cdf.benchmark_inference_time(q_sizes, p_sizes)
    
    # Print results in a simple format
    print("\nInference Time Results (milliseconds):")
    print("Format: q_size, p_size: mean ± std")
    for (q_size, p_size), stats in sorted(results.items()):
        print(f"q={q_size}, p={p_size}: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    