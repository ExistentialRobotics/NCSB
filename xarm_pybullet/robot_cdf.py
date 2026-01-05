import sys
from pathlib import Path
import torch
import numpy as np
import time
from typing import List

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cdf_training.network import CDFNetwork

class RobotCDF:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load trained CDF model
        self.model = self._load_cdf_model()
        
    def _load_cdf_model(self):
        """Load trained CDF model"""
        model_dir = project_root / "trained_models" / "cdf"
        model_path = model_dir / "best_model_bfgs_gelu_3.pth"
        
        model = CDFNetwork(activation='gelu').to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model
    
    def query_cdf(self, points, joint_angles, return_gradients=False):
        """
        Query CDF for points given joint angles
        Args:
            points: [B, N, 3] tensor of 3D points
            joint_angles: [B, 6] tensor of joint angles
            return_gradients: bool, whether to return gradients w.r.t. joint angles
        Returns:
            cdf_values: [B, N] tensor of CDF values
            gradients: [B, N, 6] tensor of gradients (optional)
        """
        points = points.to(dtype=torch.float32, device=self.device)
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        if return_gradients:
            # Enable gradient tracking for joint angles only
            joint_angles = joint_angles.detach().clone()
            joint_angles.requires_grad_(True)
        
        # Prepare input features
        sin_q = torch.sin(joint_angles)
        cos_q = torch.cos(joint_angles)
        
        # Expand joint features for each point
        joint_features = torch.cat([
            joint_angles.unsqueeze(1).expand(-1, N, -1),
            sin_q.unsqueeze(1).expand(-1, N, -1),
            cos_q.unsqueeze(1).expand(-1, N, -1)
        ], dim=-1)  # [B, N, 18]
        
        # Concatenate with points
        inputs = torch.cat([joint_features, points], dim=-1)  # [B, N, 21]
        
        # Forward pass
        cdf_values = self.model(inputs.reshape(B*N, -1)).reshape(B, N)
        
        if return_gradients:
            gradients = torch.zeros((B, N, 6), device=self.device)
            
            for b in range(B):
                for n in range(N):
                    # The key fix: ensure scalar output for gradient computation
                    scalar_output = cdf_values[b, n:n+1]  # Shape: [1]
                    grad_outputs = torch.ones_like(scalar_output)  # Shape: [1]
                    
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

    def test_differentiability(self):
        """Test if the model is differentiable and check eikonal constraint"""
        # Create test inputs
        points = torch.randn(1, 3, 3, device=self.device)  # [batch, num_points, 3]
        joint_angles = torch.zeros(1, 6, device=self.device, requires_grad=True)
        
        # Get CDF values and gradients
        cdf_values, gradients = self.query_cdf(points, joint_angles, return_gradients=True)
        
        # Check gradient norms (eikonal constraint)
        gradient_norms = torch.norm(gradients, dim=-1)  # [B, N]
        mean_norm = gradient_norms.mean().item()
        norm_deviation = (gradient_norms - 1.0).abs().mean().item()
        
        print("\nEikonal Constraint Check:")
        print(f"Mean gradient norm: {mean_norm:.4f} (should be close to 1.0)")
        print(f"Mean deviation from 1.0: {norm_deviation:.4f} (should be close to 0.0)")
        
        # Original differentiability test
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
                joint_angles = torch.rand(q_size, 6, device=self.device) * 2 * np.pi - np.pi  # [-π, π]
                points = torch.rand(q_size, p_size, 3, device=self.device) * 2 - 1  # [-1, 1]
                
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
    robot_cdf = RobotCDF(device)
    
    # Test points (3D)
    test_points = torch.tensor([
        [0.3, 0.2, 0.1],
        [0.4, 0.3, 0.2],
        [0.5, 0.4, 0.3]
    ], device=device).unsqueeze(0)  # [1, 3, 3]
    
    # Test joint angles
    joint_angles = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    
    print("\nTesting CDF query...")
    print(f"Test points shape: {test_points.shape}")
    print(f"Joint angles shape: {joint_angles.shape}")
    
    # Query CDF values and gradients
    cdf_values, gradients = robot_cdf.query_cdf(test_points, joint_angles, return_gradients=True)
    
    print("\nResults:")
    for i in range(len(test_points[0])):
        print(f"\nPoint {test_points[0,i].cpu().numpy()}:")
        print(f"CDF value: {cdf_values[0,i].item():.6f}")
        print(f"Gradients: {gradients[0,i].detach().cpu().numpy()}")
    
    # Test differentiability
    robot_cdf.test_differentiability() 
    
    # Add eikonal constraint visualization for test points
    print("\nEikonal Constraint at Test Points:")
    gradient_norms = torch.norm(gradients, dim=-1)  # [B, N]
    for i in range(len(test_points[0])):
        print(f"\nPoint {test_points[0,i].cpu().numpy()}:")
        print(f"CDF value: {cdf_values[0,i].item():.6f}")
        print(f"Gradient norm: {gradient_norms[0,i].item():.6f} (should be close to 1.0)")
        print(f"Gradients: {gradients[0,i].detach().cpu().numpy()}")
    
    # Add benchmarking
    print("\n" + "="*50)
    print("Running inference time benchmark...")
    print("="*50)
    
    # Define test sizes
    q_sizes = [1, 10, 100]
    p_sizes = [1, 10, 100, 1000, 10000]
    
    # Run benchmark
    results = robot_cdf.benchmark_inference_time(q_sizes, p_sizes)
    
    # Print results in a simple format
    print("\nInference Time Results (milliseconds):")
    print("Format: q_size (6D configs), p_size (3D points): mean ± std")
    for (q_size, p_size), stats in sorted(results.items()):
        print(f"q={q_size}, p={p_size}: {stats['mean']:.3f} ± {stats['std']:.3f} ms") 