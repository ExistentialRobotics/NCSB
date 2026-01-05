import sys
from pathlib import Path
import torch
import numpy as np
import time

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from self_collision_cdf_training.network import SelfCollisionCDFNetwork

class SelfCollisionCDF:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load trained CDF model
        self.model = self._load_cdf_model()
        
    def _load_cdf_model(self):
        """Load trained CDF model"""
        model_dir = project_root / "trained_models" / "self_collision_cdf"
        model_path = model_dir / "best_model.pth"
        
        model = SelfCollisionCDFNetwork(activation='gelu').to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model
    
    def query_cdf(self, joint_angles, return_gradients=False):
        """
        Query CDF for joint configurations
        Args:
            joint_angles: [B, 6] tensor of joint angles
            return_gradients: bool, whether to return gradients
        Returns:
            cdf_values: [B] tensor of CDF values (distances to nearest collision)
            gradients: [B, 6] tensor of gradients (optional)
        """
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        if return_gradients:
            joint_angles = joint_angles.detach().clone()
            joint_angles.requires_grad_(True)
        
        # Prepare input features
        inputs = torch.cat([
            joint_angles,
            torch.sin(joint_angles),
            torch.cos(joint_angles)
        ], dim=-1)  # [B, 18]
        
        # Forward pass
        cdf_values = self.model(inputs).squeeze(-1)  # [B]
        
        if return_gradients:
            gradients = torch.zeros_like(joint_angles)  # [B, 6]
            
            for b in range(len(joint_angles)):
                scalar_output = cdf_values[b:b+1]
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
                        gradients[b] = grad[b]
                except Exception as e:
                    print(f"Warning: Gradient computation failed for batch {b}: {e}")
            
            return cdf_values, gradients
        
        return cdf_values

    def test_differentiability(self, epsilon=1e-4):
        """
        Test if the model is differentiable and check eikonal constraint
        Also validates gradients using finite differences
        
        Args:
            epsilon: Step size for finite difference computation
        """
        # Create test inputs
        joint_angles = torch.zeros(5, 6, device=self.device, requires_grad=True)
        
        # Get CDF values and analytical gradients
        cdf_values, analytical_gradients = self.query_cdf(joint_angles, return_gradients=True)
        
        # Compute finite difference gradients
        finite_diff_gradients = torch.zeros_like(analytical_gradients)
        
        with torch.no_grad():
            for b in range(len(joint_angles)):
                for j in range(6):  # 6 joints
                    # Create positive and negative perturbations
                    pos_perturb = joint_angles[b].clone()
                    neg_perturb = joint_angles[b].clone()
                    
                    pos_perturb[j] += epsilon
                    neg_perturb[j] -= epsilon
                    
                    # Query CDF for perturbed configurations
                    pos_value = self.query_cdf(pos_perturb.unsqueeze(0))
                    neg_value = self.query_cdf(neg_perturb.unsqueeze(0))
                    
                    # Central difference
                    finite_diff_gradients[b, j] = (pos_value - neg_value) / (2 * epsilon)
        
        # Compare gradients
        grad_diff = torch.norm(analytical_gradients - finite_diff_gradients, dim=-1)
        mean_diff = grad_diff.mean().item()
        max_diff = grad_diff.max().item()
        
        print("\nGradient Validation:")
        print(f"Mean difference between analytical and finite diff: {mean_diff:.6f}")
        print(f"Max difference between analytical and finite diff: {max_diff:.6f}")
        
        # Example comparison for first configuration
        print("\nDetailed comparison for first configuration:")
        print("Joint | Analytical | Finite Diff | Difference")
        print("-" * 50)
        for j in range(6):
            print(f"{j:5d} | {analytical_gradients[0,j]:10.6f} | {finite_diff_gradients[0,j]:10.6f} | {abs(analytical_gradients[0,j] - finite_diff_gradients[0,j]):10.6f}")
        
        # Check gradient norms (eikonal constraint)
        gradient_norms = torch.norm(analytical_gradients, dim=-1)
        mean_norm = gradient_norms.mean().item()
        norm_deviation = (gradient_norms - 1.0).abs().mean().item()
        
        print("\nEikonal Constraint Check:")
        print(f"Mean gradient norm: {mean_norm:.4f} (should be close to 1.0)")
        print(f"Mean deviation from 1.0: {norm_deviation:.4f} (should be close to 0.0)")

    def benchmark_inference_time(self, batch_sizes, num_trials=100):
        """Benchmark inference time for different batch sizes"""
        results = {}
        
        for batch_size in batch_sizes:
            # Generate random test data
            joint_angles = torch.rand(batch_size, 6, device=self.device) * 2 * np.pi - np.pi
            
            # Warm-up runs
            for _ in range(10):
                _ = self.query_cdf(joint_angles)
            
            # Timing runs
            torch.cuda.synchronize()
            times = []
            
            for _ in range(num_trials):
                start = time.perf_counter()
                _ = self.query_cdf(joint_angles)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            # Calculate statistics
            times = np.array(times)
            results[batch_size] = {
                'mean': np.mean(times) * 1000,  # Convert to milliseconds
                'std': np.std(times) * 1000
            }
        
        return results

if __name__ == "__main__":
    device = 'cuda'
    self_collision_cdf = SelfCollisionCDF(device)
    
    # Test configurations
    test_configs = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Zero configuration
        [1.0, 0.5, 0.3, 0.2, 0.1, 0.0],  # Random configuration
        [-0.5, 0.8, -0.3, 0.4, -0.2, 0.1]  # Another random configuration
    ], device=device)
    
    print("\nTesting CDF query...")
    print(f"Test configurations shape: {test_configs.shape}")
    
    # Query CDF values and gradients
    cdf_values, gradients = self_collision_cdf.query_cdf(test_configs, return_gradients=True)
    
    print("\nResults:")
    for i in range(len(test_configs)):
        print(f"\nConfiguration {i}:")
        print(f"Joint angles: {test_configs[i].cpu().numpy()}")
        print(f"Distance to collision: {cdf_values[i].item():.6f}")
        print(f"Gradient (direction to move away from collision):\n{gradients[i].detach().cpu().numpy()}")
    
    # Test differentiability
    # self_collision_cdf.test_differentiability()
    
    # Benchmark inference time
    print("\nRunning inference time benchmark...")
    batch_sizes = [1, 10, 100, 1000, 10000]
    results = self_collision_cdf.benchmark_inference_time(batch_sizes)
    
    print("\nInference Time Results (milliseconds):")
    print("Format: batch_size (6D configs): mean ± std")
    for batch_size, stats in sorted(results.items()):
        print(f"batch={batch_size}: {stats['mean']:.3f} ± {stats['std']:.3f} ms") 