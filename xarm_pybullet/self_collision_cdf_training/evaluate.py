import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import os

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from self_collision_cdf import SelfCollisionCDF
from models.xarm_model import XArmFK

def create_evaluation_dataset(n_configs=1000, save_path=None, device='cuda'):
    """Create dataset of configurations and their ground truth CDF values"""
    # Load the template dataset
    data_path = project_root / "data/self_cdf_data/refined_self_collision_data.npy"
    template_data = np.load(data_path, allow_pickle=True).item()
    
    # Convert templates to tensors
    template_configs = torch.tensor(template_data['colliding_configs'], dtype=torch.float32, device=device)
    template_joints = template_data['relevant_joints']
    
    # Group templates by joint combinations
    joint_groups = {}
    for config, joints in zip(template_configs, template_joints):
        key = tuple(sorted(joints))
        if key not in joint_groups:
            joint_groups[key] = []
        joint_groups[key].append(config)
    
    # Sample random configurations
    robot_fk = XArmFK(device=device)  # Initialize with device
    configs = robot_fk.joint_limits[:, 0] + torch.rand(n_configs, 6, device=device) * (
        robot_fk.joint_limits[:, 1] - robot_fk.joint_limits[:, 0]
    )
    
    # Compute ground truth CDF values
    gt_values = torch.zeros(n_configs, device=device)
    
    for i, config in enumerate(configs):
        min_dist = float('inf')
        # Check distance to templates in each joint group
        for joints, templates in joint_groups.items():
            templates = torch.stack(templates)  # Already on correct device
            joint_indices = list(joints)
            
            # Compute distances only for relevant joints
            diff = config[joint_indices] - templates[:, joint_indices]
            distances = torch.norm(diff, dim=1)
            min_dist = min(min_dist, torch.min(distances).item())
        
        gt_values[i] = min_dist
    
    # Save dataset (move to CPU for saving)
    if save_path is None:
        save_path = project_root / "data/self_cdf_data/evaluation_dataset.pt"
    
    torch.save({
        'configs': configs.cpu(),
        'gt_cdf_values': gt_values.cpu(),
    }, save_path)
    
    return configs, gt_values

def evaluate_quantitative(eval_dataset_path=None, device='cuda'):
    """Evaluate trained model against ground truth"""
    # Load or create evaluation dataset
    if eval_dataset_path is None:
        eval_dataset_path = project_root / "data/self_cdf_data/evaluation_dataset.pt"
    
    if not os.path.exists(eval_dataset_path):
        print("Creating new evaluation dataset...")
        configs, gt_values = create_evaluation_dataset(save_path=eval_dataset_path, device=device)
    else:
        print("Loading evaluation dataset...")
        data = torch.load(eval_dataset_path)
        configs = data['configs'].to(device)
        gt_values = data['gt_cdf_values'].to(device)
    
    # Initialize model
    model = SelfCollisionCDF(device=device)
    
    # Get predictions
    print("Computing model predictions...")
    with torch.no_grad():
        pred_values = model.query_cdf(configs)
    
    # Compute metrics
    mse = torch.nn.functional.mse_loss(pred_values, gt_values).item()
    mae = torch.nn.functional.l1_loss(pred_values, gt_values).item()
    correlation = torch.corrcoef(torch.stack([pred_values, gt_values]))[0,1].item()
    
    print("\nQuantitative Evaluation Results:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Correlation Coefficient: {correlation:.6f}")
    
    # Visualize results
    plt.figure(figsize=(10, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(gt_values.cpu(), pred_values.cpu(), alpha=0.1)
    plt.plot([0, gt_values.max().item()], [0, gt_values.max().item()], 'r--')
    plt.xlabel('Ground Truth CDF')
    plt.ylabel('Predicted CDF')
    plt.title('Predicted vs Ground Truth CDF')
    
    # Error histogram
    plt.subplot(122)
    errors = (pred_values - gt_values).cpu().numpy()
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'configs': configs,
        'predictions': pred_values,
        'ground_truth': gt_values
    }

if __name__ == "__main__":
    metrics = evaluate_quantitative() 