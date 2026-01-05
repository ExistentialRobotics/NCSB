import sys
from pathlib import Path
import glob

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
import trimesh
import mcubes  # for marching cubes
from network import SDFNetwork
import matplotlib.pyplot as plt

def load_trained_model(model_path, device='cuda'):
    """Load a trained SDF model"""
    checkpoint = torch.load(model_path)
    
    model = SDFNetwork(
        input_size=3,
        output_size=1,
        hidden_size=128,
        num_layers=5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def extract_mesh_from_sdf(model, resolution=128, device='cuda'):
    """Extract mesh from SDF using marching cubes"""
    # Create a grid of points
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Reshape to (N, 3) array
    points = torch.FloatTensor(np.stack([X, Y, Z], axis=-1)).reshape(-1, 3).to(device)
    
    # Evaluate SDF in batches
    sdf_values = []
    batch_size = 32768
    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            sdf_values.append(model(batch_points).cpu().numpy())
    
    sdf_values = np.concatenate(sdf_values, axis=0)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    # Extract mesh using marching cubes
    vertices, triangles = mcubes.marching_cubes(sdf_grid, 0.0)
    
    # Scale vertices back to [-1, 1]
    vertices = vertices / (resolution - 1) * 2 - 1
    
    return vertices, triangles

def visualize_comparison(original_mesh, predicted_vertices, predicted_triangles, link_id, original_center, original_scale):
    """Visualize original mesh and predicted mesh side by side"""
    fig = plt.figure(figsize=(12, 6))
    
    # Original mesh plot
    ax1 = fig.add_subplot(121, projection='3d')
    # Scale original mesh to match SDF normalization
    vertices_centered = original_mesh.vertices - original_center
    vertices_normalized = vertices_centered / original_scale
    
    ax1.plot_trisurf(vertices_normalized[:, 0],
                     vertices_normalized[:, 1],
                     vertices_normalized[:, 2],
                     triangles=original_mesh.faces,
                     color='blue', alpha=0.5)
    ax1.set_title(f'Original Mesh (Link {link_id})')
    
    # Predicted mesh plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(predicted_vertices[:, 0],
                     predicted_vertices[:, 1],
                     predicted_vertices[:, 2],
                     triangles=predicted_triangles,
                     color='red', alpha=0.5)
    ax2.set_title(f'Predicted Mesh (Link {link_id})')
    
    # Set same limits and labels for both plots
    for ax in [ax1, ax2]:
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    plt.tight_layout()
    return fig

def evaluate_all_links(model_dir, mesh_dir, sdf_data_dir, device='cuda'):
    """Evaluate all trained models and compare with original meshes"""
    model_dir = Path(model_dir)
    mesh_dir = Path(mesh_dir)
    sdf_data_dir = Path(sdf_data_dir)
    
    # For each link
    for link_id in range(7):
        print(f"\nEvaluating Link {link_id}")
        
        # Load trained model
        model_path = model_dir / f"link_{link_id}" / "best_model.pt"
        print(f"Loading model from: {model_path}")
        model = load_trained_model(model_path, device)
        
        # Load original mesh
        mesh_file = mesh_dir / f"link{link_id}.stl"
        print(f"Loading mesh from: {mesh_file}")
        original_mesh = trimesh.load(str(mesh_file))
        
        # Load SDF data to get original scaling parameters
        sdf_files = sorted(glob.glob(str(sdf_data_dir / f"link_{link_id}_*.npz")))
        if not sdf_files:
            print(f"No SDF data found for link {link_id}")
            continue
            
        sdf_data = np.load(sdf_files[0])
        original_center = sdf_data['original_center']
        original_scale = sdf_data['original_scale']
        
        print(f"Original center: {original_center}")
        print(f"Original scale: {original_scale}")
        
        # Extract mesh from SDF
        vertices, triangles = extract_mesh_from_sdf(model, resolution=128, device=device)
        
        # Visualize comparison
        fig = visualize_comparison(original_mesh, vertices, triangles, link_id, 
                                 original_center, original_scale)
        plt.show()

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "trained_models"
    mesh_dir = project_root / "xarm_description" / "meshes" / "xarm6" / "visual"
    sdf_data_dir = project_root / "data" / "sdf_data"
    
    evaluate_all_links(model_dir, mesh_dir, sdf_data_dir) 