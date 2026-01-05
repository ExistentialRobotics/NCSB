import torch
import numpy as np
from pathlib import Path
import trimesh
import matplotlib.pyplot as plt
import time
import os
# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from models.xarm_model import XArmFK
from utils.visualization import SDFVisualizer
from train_online_batch import compute_cdf_values

from train_online_batch import CDFTrainer

src_dir = project_root

def evaluate_sdf_cdf_correlation(device='cuda'):
    """
    Evaluate correlation between SDF and CDF values with visualization
    """
    print("Initializing models...")
    robot_sdf = RobotSDF(device=device)
    robot_cdf = RobotCDF(device=device)
    robot_fk = XArmFK(device=device)
    visualizer = SDFVisualizer(device=device)
    
    # Load contact database
    contact_db_path = src_dir / "data/cdf_data/refined_bfgs_100_contact_db.npy"
    contact_db = np.load(contact_db_path, allow_pickle=True).item()
    valid_points = torch.tensor(contact_db['points'], device=device)
    contact_configs = contact_db['contact_configs']
    link_indices = contact_db['link_indices']
    
    # Define test configurations
    test_configs = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device),  # Home pose
        torch.tensor([np.pi/4, np.pi/3, -np.pi/4, np.pi/6, np.pi/4, 0.0], device=device),  # Bent pose
    ]
    
    # Define test points for each configuration
    test_points = [
        # Points around the robot
        torch.tensor([
            [0.3, 0.0, 0.3],   # Front of robot
            [0.0, 0.3, 0.3],   # Side of robot
            [0.3, 0.3, 0.3],   # Diagonal from robot
            [0.2, 0.0, 0.5],   # Above robot
            [0.4, 0.0, 0.2],   # Near base
        ], device=device)
    ]
    
    # Evaluate each configuration
    for i, q in enumerate(test_configs):
        print(f"\n=== Configuration {i+1} ===")
        print(f"Joint angles: {q.cpu().numpy()}")
        
        points = test_points[0].unsqueeze(0)
        
        # Find closest points in contact database
        closest_indices = []
        for point in points[0]:
            distances = torch.norm(valid_points - point, dim=1)
            closest_idx = torch.argmin(distances).item()
            closest_indices.append(closest_idx)
        
        # Get corresponding contact configs and link indices
        contact_configs_batch = [contact_configs[idx] for idx in closest_indices]
        link_indices_batch = [link_indices[idx] for idx in closest_indices]
        
        with torch.no_grad():
            sdf_values = robot_sdf.query_sdf(points, q.unsqueeze(0))
            cdf_values = robot_cdf.query_cdf(points, q.unsqueeze(0))
            gt_cdf_values = compute_cdf_values(
                points=points[0],
                configs=q.unsqueeze(0),
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
        
        # Show each point individually
        for j, point in enumerate(points[0]):
            print(f"\nPoint {j+1} at {point.cpu().numpy()}:")
            print(f"  SDF: {sdf_values[0,j].item():.4f}")
            print(f"  CDF (predicted): {cdf_values[0,j].item():.4f}")
            print(f"  CDF (ground truth): {gt_cdf_values[j,0].item():.4f}")
            print(f"  Contact Link Index: {link_indices_batch[j][0]}")  # Show link index for closest contact
            
            # Visualize current and closest contact config
            print("\nShowing visualizations... (close windows to continue)")
            
            # Current configuration
            scene_current = visualizer.visualize_sdf(q.unsqueeze(0), show_meshes=False, resolution=64)
            sphere_current = trimesh.primitives.Sphere(radius=0.02)
            sphere_current.visual.face_colors = [255, 0, 0, 255]  # Red color
            sphere_current.apply_translation(point.cpu().numpy())
            scene_current.add_geometry(sphere_current)
            scene_current.show()
            
            # Closest contact configuration
            contact_q = contact_configs_batch[j][0]  # Take first (closest) contact config
            contact_q_tensor = torch.tensor(contact_q, device=device).unsqueeze(0)
            scene_contact = visualizer.visualize_sdf(contact_q_tensor, show_meshes=False, resolution=64)
            sphere_contact = trimesh.primitives.Sphere(radius=0.02)
            sphere_contact.visual.face_colors = [0, 255, 0, 255]  # Green color
            sphere_contact.apply_translation(point.cpu().numpy())
            scene_contact.add_geometry(sphere_contact)
            scene_contact.show()

def create_evaluation_dataset(batch_q_size=200, batch_x_size=300, device='cuda', 
                            save_path='data/cdf_data/evaluation_dataset.pt', 
                            mini_batch_size=50):
    """
    Create evaluation dataset by processing mini-batches of points and configurations
    
    Args:
        batch_q_size: Total number of configurations (200)
        batch_x_size: Total number of points (1000)
        mini_batch_size: Size of mini-batches to process at once
        device: Computing device
        save_path: Where to save the dataset
    """

    contact_db_path = src_dir / "data/cdf_data/refined_bfgs_100_contact_db.npy"
    trainer = CDFTrainer(contact_db_path, device=device)
    
    # Sample all points at once
    point_indices = torch.randint(0, len(trainer.valid_points), (batch_x_size,))
    all_points = trainer.valid_points[point_indices]
    
    # Sample all configurations at once
    all_configs = trainer.sample_q(batch_q=batch_q_size)
    
    # Initialize final CDF values tensor
    final_cdf_values = torch.zeros(batch_x_size, batch_q_size)
    
    # Process in mini-batches
    for i in range(0, batch_x_size, mini_batch_size):
        for j in range(0, batch_q_size, mini_batch_size):
            # Get current mini-batch indices
            x_start, x_end = i, min(i + mini_batch_size, batch_x_size)
            q_start, q_end = j, min(j + mini_batch_size, batch_q_size)

            start_time = time.time()
            
            print(f"Processing batch: points [{x_start}:{x_end}], configs [{q_start}:{q_end}]")
            
            # Get mini-batch data
            points_batch = all_points[x_start:x_end]
            configs_batch = all_configs[q_start:q_end]
            
            # Get corresponding contact data
            contact_configs_batch = [trainer.contact_configs[idx] for idx in point_indices[x_start:x_end]]
            link_indices_batch = [trainer.link_indices[idx] for idx in point_indices[x_start:x_end]]
            
            # Compute CDF values for this mini-batch
            cdf_values = compute_cdf_values(
                points=points_batch,
                configs=configs_batch,
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
            
            # Store in final tensor
            final_cdf_values[x_start:x_end, q_start:q_end] = cdf_values.cpu()
            
            # Clear CUDA cache
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

def evaluate_quantitative(eval_dataset_path='data/cdf_data/evaluation_dataset.pt', device='cuda'):
    """
    Perform quantitative evaluation using a fixed evaluation dataset
    """
    print("Loading evaluation dataset...")
    data = torch.load(eval_dataset_path)
    points = data['points'].to(device)      # [N, 3]
    configs = data['configs'].to(device)    # [M, 6]
    gt_cdf_values = data['gt_cdf_values'].to(device)  # [N, M]
    
    print("Initializing models...")
    robot_sdf = RobotSDF(device=device)
    robot_cdf = RobotCDF(device=device)
    
    print("Computing model predictions...")
    with torch.no_grad():
        # Reshape points and configs to match expected dimensions
        points_expanded = points.unsqueeze(0).expand(configs.shape[0], -1, -1)  # [M, N, 3]
        pred_cdf_values = robot_cdf.query_cdf(points_expanded, configs)         # [M, N]
        # Transpose to match ground truth shape
        pred_cdf_values = pred_cdf_values.t()  # [N, M]
    
    # both pred_cdf_values and gt_cdf_values are shape [N, M]
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
    plt.plot([0, gt_flat.max().item()], [0, gt_flat.max().item()], 'r--')  # Perfect prediction line
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

def evaluate_all_datasets(base_path, device='cuda'):
    """
    Evaluate all 5 evaluation datasets and compute aggregate statistics
    """
    all_metrics = []
    
    for i in range(1, 5):
        eval_path = os.path.join(base_path, f'data/cdf_data/evaluation_dataset_{i}.pt')
        if not os.path.exists(eval_path):
            print(f"Warning: Dataset {i} not found at {eval_path}")
            continue
            
        print(f"\n=== Evaluating Dataset {i} ===")
        metrics = evaluate_quantitative(eval_dataset_path=eval_path, device=device)
        all_metrics.append(metrics)
    
    # Compute aggregate statistics
    if all_metrics:
        print("\n=== Aggregate Results ===")
        mean_mse = np.mean([m['mse'] for m in all_metrics])
        std_mse = np.std([m['mse'] for m in all_metrics])
        mean_mae = np.mean([m['mae'] for m in all_metrics])
        std_mae = np.std([m['mae'] for m in all_metrics])
        mean_corr = np.mean([m['correlation'] for m in all_metrics])
        std_corr = np.std([m['correlation'] for m in all_metrics])
        
        print(f"MSE: {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"MAE: {mean_mae:.6f} ± {std_mae:.6f}")
        print(f"Correlation: {mean_corr:.6f} ± {std_corr:.6f}")
    
    return all_metrics

if __name__ == "__main__":
    # Create evaluation dataset (only need to run once)
    
    # save_path = os.path.join(src_dir, 'data/cdf_data/evaluation_dataset_4.pt')
    # create_evaluation_dataset(batch_q_size=100, batch_x_size=100, device='cuda', save_path=save_path)
    
    # Run evaluation on all datasets
    metrics = evaluate_all_datasets(src_dir)
    
    # # Qualitative evaluation (commented out by default)
    # evaluate_sdf_cdf_correlation()
