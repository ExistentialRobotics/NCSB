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

from robot_models import Robot2D
from cdf_computation import CDFDataProcessor, compute_cdf_and_gradients
from network import CDFNetwork
from robot_cdf import RobotCDF

def create_evaluation_dataset(
    batch_q_size=200,
    batch_x_size=1000,
    device='cuda',
    save_path='data/evaluation_dataset_2d.pt',
    mini_batch_size=100
):
    """Create evaluation dataset by processing mini-batches of points and configurations"""

    src_dir = Path(__file__).parent
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"
    processor = CDFDataProcessor(contact_db_path, device=device)
    
    # Sample all points at once
    point_indices = torch.randint(0, len(processor.valid_points), (batch_x_size,))
    all_points = processor.valid_points[point_indices]
    
    # Sample all configurations at once
    all_configs = processor.sample_q(batch_q=batch_q_size)
    
    # Initialize final CDF values tensor
    final_cdf_values = torch.zeros(batch_x_size, batch_q_size)
    
    # Process in mini-batches
    for i in range(0, batch_x_size, mini_batch_size):
        for j in range(0, batch_q_size, mini_batch_size):
            x_start, x_end = i, min(i + mini_batch_size, batch_x_size)
            q_start, q_end = j, min(j + mini_batch_size, batch_q_size)
            
            start_time = time.time()
            print(f"Processing batch: points [{x_start}:{x_end}], configs [{q_start}:{q_end}]")
            
            # Get mini-batch data
            points_batch = all_points[x_start:x_end]
            configs_batch = all_configs[q_start:q_end]
            
            # Get corresponding contact data
            contact_configs_batch = [processor.contact_configs[idx] for idx in point_indices[x_start:x_end]]
            link_indices_batch = [processor.link_indices[idx] for idx in point_indices[x_start:x_end]]
            
            # Compute CDF values for this mini-batch
            cdf_values, _ = compute_cdf_and_gradients(
                points=points_batch,
                configs=configs_batch,
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
            
            # Store in final tensor
            final_cdf_values[x_start:x_end, q_start:q_end] = cdf_values.cpu()
            
            torch.cuda.empty_cache()
            print(f"Completed {(x_end-x_start)*(q_end-q_start)} values in {time.time() - start_time:.2f} seconds")
    
    # Save dataset
    torch.save({
        'points': all_points.cpu(),
        'configs': all_configs.cpu(),
        'gt_cdf_values': final_cdf_values,
    }, save_path)
    
    print(f"Evaluation dataset saved to {save_path}")
    print(f"Final shapes - Points: {all_points.shape}, Configs: {all_configs.shape}, "
          f"CDF Values: {final_cdf_values.shape}")
    
    return all_points, all_configs, final_cdf_values

def evaluate_quantitative(
    eval_dataset_path='data/evaluation_dataset_2d.pt',
    device='cuda'
):
    """Perform quantitative evaluation using a fixed evaluation dataset"""
    print("Loading evaluation dataset...")

    src_dir = Path(__file__).parent
    eval_dataset_path = src_dir / "data" / "evaluation_dataset_2d.pt"
    data = torch.load(eval_dataset_path)
    points = data['points'].to(device)      # [N, 2]
    configs = data['configs'].to(device)    # [M, 2]
    gt_cdf_values = data['gt_cdf_values'].to(device)  # [N, M]
    
    print(f"\nDataset shapes:")
    print(f"Points shape: {points.shape}")
    print(f"Configs shape: {configs.shape}")
    print(f"Ground truth values shape: {gt_cdf_values.shape}")
    
    print("\nLoading trained model...")
    robot_cdf = RobotCDF(device=device)
    
    print("\nComputing model predictions...")
    with torch.no_grad():
        batch_size = 100
        pred_cdf_values = torch.zeros_like(gt_cdf_values)
        
        for i in range(0, points.shape[0], batch_size):
            for j in range(0, configs.shape[0], batch_size):
                i_end = min(i + batch_size, points.shape[0])
                j_end = min(j + batch_size, configs.shape[0])
                
                # Get batch of points and configs
                points_batch = points[i:i_end]  # [batch_N, 2]
                configs_batch = configs[j:j_end]  # [batch_M, 2]
                
                # For each config in the batch, compute CDF for all points
                for k, config in enumerate(configs_batch):
                    points_input = points_batch.unsqueeze(0)  # [1, batch_N, 2]
                    config_input = config.unsqueeze(0)  # [1, 2]
                    
                    # print(f"\nBatch shapes:")
                    # print(f"Points input shape: {points_input.shape}")
                    # print(f"Config input shape: {config_input.shape}")
                    
                    pred_batch = robot_cdf.query_cdf(points_input, config_input)
                    pred_cdf_values[i:i_end, j+k] = pred_batch.squeeze(0)
                    
                # print(f"Processed batch: points [{i}:{i_end}], configs [{j}:{j_end}]")
    
    # Compute metrics
    mse = torch.nn.functional.mse_loss(pred_cdf_values, gt_cdf_values).item()
    mae = torch.nn.functional.l1_loss(pred_cdf_values, gt_cdf_values).item()
    
    # Compute correlation coefficient
    pred_flat = pred_cdf_values.flatten()
    gt_flat = gt_cdf_values.flatten()
    correlation = torch.corrcoef(torch.stack([pred_flat, gt_flat]))[0,1].item()
    
    # Print results
    print("\nQuantitative Evaluation Results:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Correlation Coefficient: {correlation:.6f}")
    
    # Visualize predictions vs ground truth
    plt.figure(figsize=(10, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(gt_flat.detach().cpu(), pred_flat.detach().cpu(), alpha=0.1)
    plt.plot([0, gt_flat.max().item()], [0, gt_flat.max().item()], 'r--')
    plt.xlabel('Ground Truth CDF')
    plt.ylabel('Predicted CDF')
    plt.title('Predicted vs Ground Truth CDF')
    
    # Error histogram
    plt.subplot(122)
    errors = (pred_cdf_values - gt_cdf_values).detach().cpu().numpy()
    plt.hist(errors.flatten(), bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'points': points,
        'configs': configs,
        'predictions': pred_cdf_values,
        'ground_truth': gt_cdf_values
    }

def visualize_test_cases(device='cuda'):
    """Visualize specific test cases with the trained model"""
    # Initialize robot_cdf
    robot_cdf = RobotCDF(device=device)
    robot = Robot2D(device=device)
    
    # Define test configurations
    test_configs = [
        torch.tensor([0.0, 0.0], device=device, dtype=torch.float32),
        torch.tensor([np.pi/4, np.pi/4], device=device, dtype=torch.float32),
        torch.tensor([np.pi/2, -np.pi/2], device=device, dtype=torch.float32),
    ]
    
    # Create a grid of test points
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                         device=device, dtype=torch.float32)  # [N, 2]
    
    # Plot for each configuration
    for i, config in enumerate(test_configs):
        plt.figure(figsize=(8, 8))
        
        # Prepare inputs for robot_cdf
        points_input = points.unsqueeze(0)  # [1, N, 2]
        config_input = config.unsqueeze(0)  # [1, 2]
        
        # Get CDF predictions using robot_cdf
        with torch.no_grad():
            cdf_values = robot_cdf.query_cdf(points_input, config_input).squeeze(0).cpu().numpy()
        
        # Plot CDF heatmap
        plt.scatter(X.flatten(), Y.flatten(), c=cdf_values, cmap='viridis')
        plt.colorbar(label='CDF Value')
        
        # Plot robot configuration
        joints = robot.forward_kinematics(config.unsqueeze(0))[0]
        joints = joints.cpu().numpy()
        plt.plot(joints[:, 0], joints[:, 1], 'r-', linewidth=2, label='Robot')
        plt.scatter(joints[:, 0], joints[:, 1], c='red', s=50)
        
        plt.title(f'Configuration {i+1}: [{config[0]:.2f}, {config[1]:.2f}]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

def compare_computation_efficiency(model_path, device='cuda', num_trials=100):
    """Compare computation time between learned model and template-based approach"""
    print("\nComparing computation efficiency...")
    
    # Initialize models and processor
    model = CDFNetwork(input_dims=8, output_dims=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    src_dir = Path(__file__).parent
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"
    processor = CDFDataProcessor(contact_db_path, device=device)
    
    # Test cases: (num_points, num_configs)
    test_cases = [
        (1, 1),
        (1, 10),
        (1, 100),
        (10, 1),
    ]
    
    results = {}
    
    for num_points, num_configs in test_cases:
        key = f"points_{num_points}_configs_{num_configs}"
        results[key] = {'learned': [], 'template': []}
        
        print(f"\nTest case: {num_points} points, {num_configs} configs")
        
        for _ in range(num_trials):
            # Sample points and configs
            point_indices = torch.randint(0, len(processor.valid_points), (num_points,))
            points = processor.valid_points[point_indices]
            configs = processor.sample_q(batch_q=num_configs)
            
            # Get contact data and ensure it's on the correct device
            contact_configs_batch = []
            link_indices_batch = []
            for idx in point_indices:
                # Reshape contact configs to 2D tensor [N, 2]
                contact_configs = torch.tensor(processor.contact_configs[idx], device=device)
                if contact_configs.dim() == 1:
                    contact_configs = contact_configs.reshape(-1, 2)
                contact_configs_batch.append(contact_configs)
                
                # Convert link indices to tensor
                link_indices = torch.tensor(processor.link_indices[idx], device=device)
                link_indices_batch.append(link_indices)
            
            # Time learned model
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # Prepare inputs with positional encoding
                points_exp = points.unsqueeze(1).expand(-1, configs.shape[0], -1)
                configs_exp = configs.unsqueeze(0).expand(points.shape[0], -1, -1)
                configs_sin = torch.sin(configs_exp)
                configs_cos = torch.cos(configs_exp)
                
                inputs = torch.cat([
                    configs_exp.reshape(-1, 2),
                    configs_sin.reshape(-1, 2),
                    configs_cos.reshape(-1, 2),
                    points_exp.reshape(-1, 2)
                ], dim=1)
                
                _ = model(inputs)
            
            torch.cuda.synchronize()
            learned_time = time.time() - start_time
            results[key]['learned'].append(learned_time)
            
            # Time template-based approach
            torch.cuda.synchronize()
            start_time = time.time()
            
            from cdf_computation import compute_cdf_and_gradients
            _, _ = compute_cdf_and_gradients(
                points=points,
                configs=configs,
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
            
            torch.cuda.synchronize()
            template_time = time.time() - start_time
            results[key]['template'].append(template_time)
        
        # Compute statistics
        learned_mean = np.mean(results[key]['learned'])
        learned_std = np.std(results[key]['learned'])
        template_mean = np.mean(results[key]['template'])
        template_std = np.std(results[key]['template'])
        speedup = template_mean / learned_mean
        
        print(f"Learned model:   {learned_mean*1000:.2f} ± {learned_std*1000:.2f} ms")
        print(f"Template-based:  {template_mean*1000:.2f} ± {template_std*1000:.2f} ms")
        print(f"Speedup factor:  {speedup:.1f}x")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(test_cases))
    width = 0.35
    
    learned_means = [np.mean(results[f"points_{p}_configs_{c}"]['learned']) * 1000 
                    for p, c in test_cases]
    template_means = [np.mean(results[f"points_{p}_configs_{c}"]['template']) * 1000 
                     for p, c in test_cases]
    
    learned_stds = [np.std(results[f"points_{p}_configs_{c}"]['learned']) * 1000 
                   for p, c in test_cases]
    template_stds = [np.std(results[f"points_{p}_configs_{c}"]['template']) * 1000 
                    for p, c in test_cases]
    
    plt.bar(x - width/2, learned_means, width, label='Learned Model',
            yerr=learned_stds, capsize=5)
    plt.bar(x + width/2, template_means, width, label='Template-based',
            yerr=template_stds, capsize=5)
    
    plt.xlabel('Test Cases')
    plt.ylabel('Computation Time (ms)')
    plt.title('Computation Time Comparison')
    plt.xticks(x, [f'{p}p,{c}c' for p, c in test_cases], rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

def visualize_cdf_slice(device='cuda'):
    """Visualize a single slice of training data showing a robot config, point, and contact configs"""
    # Initialize robot and data processor
    robot = Robot2D(device=device)
    src_dir = Path(__file__).parent
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"
    processor = CDFDataProcessor(contact_db_path, device=device)
    
    # Sample a random point and its contact configs
    point_idx = torch.randint(0, len(processor.valid_points), (1,)).item()
    point = processor.valid_points[point_idx]
    contact_configs = torch.tensor(processor.contact_configs[point_idx], device=device)
    if contact_configs.dim() == 1:
        contact_configs = contact_configs.reshape(-1, 2)
    
    # Randomly select 30 contact configs if we have more than that
    if len(contact_configs) > 30:
        indices = torch.randperm(len(contact_configs))[:30]
        contact_configs = contact_configs[indices]
    
    # Sample a test configuration
    test_config = processor.sample_q(batch_q=1)[0]
    
    # Compute CDF values for the point with respect to all contact configs
    distances = torch.norm(contact_configs - test_config.unsqueeze(0), dim=1)
    closest_idx = torch.argmin(distances)
    closest_config = contact_configs[closest_idx]
    
    # Create visualization with larger figure size
    plt.figure(figsize=(12, 12))
    
    # Plot contact configurations in light gray
    # Plot first contact config with label, rest without to avoid duplicate legends
    joints = robot.forward_kinematics(contact_configs[0].unsqueeze(0))[0]
    joints = joints.cpu().numpy()
    plt.plot(joints[:, 0], joints[:, 1], 'lightgray', alpha=0.3, linewidth=2, label='Contact Configs')
    
    # Plot remaining contact configs
    for config in contact_configs[1:]:
        joints = robot.forward_kinematics(config.unsqueeze(0))[0]
        joints = joints.cpu().numpy()
        plt.plot(joints[:, 0], joints[:, 1], 'gray', alpha=0.6, linewidth=2)
    
    # Plot the test configuration in blue
    test_joints = robot.forward_kinematics(test_config.unsqueeze(0))[0]
    test_joints = test_joints.detach().cpu().numpy()
    plt.plot(test_joints[:, 0], test_joints[:, 1], 'b-', linewidth=3, label='Test Config')
    plt.scatter(test_joints[:, 0], test_joints[:, 1], c='blue', s=100)
    
    # Plot the closest configuration in red
    closest_joints = robot.forward_kinematics(closest_config.unsqueeze(0))[0]
    closest_joints = closest_joints.detach().cpu().numpy()
    plt.plot(closest_joints[:, 0], closest_joints[:, 1], 'r-', linewidth=3, label='Closest Contact Config')
    plt.scatter(closest_joints[:, 0], closest_joints[:, 1], c='red', s=100)
    
    # Plot the query point as a purple star
    point = point.detach().cpu().numpy()
    plt.plot(point[0], point[1], '*', color='purple', markersize=20, label='Query Point')
    
    plt.title(f'Neural CDF Training Data', fontsize=22)
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.axis('equal')
    plt.grid(True)
    
    # Increase legend and tick size
    plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('cdf_slice_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    src_dir = project_root
    
    # Create evaluation dataset (only need to run once)
    save_path = src_dir / 'cdf_training' / 'data' / 'evaluation_dataset_2d.pt'
    # create_evaluation_dataset(batch_q_size=200, batch_x_size=400, device='cuda', save_path=save_path)
    
    # Run efficiency comparison
    # efficiency_results = compare_computation_efficiency(model_path=model_path)
    
    # Run other evaluations
    # metrics = evaluate_quantitative()
    # visualize_test_cases()
    
    # Add this line to run the new visualization
    visualize_cdf_slice() 