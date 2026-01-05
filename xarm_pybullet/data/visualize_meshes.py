import trimesh
import glob
from pathlib import Path
import numpy as np
import torch


import sys
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from models.xarm6_differentiable_fk import fk_xarm6_torch

def visualize_robot_meshes(mesh_dir, gripper_dir, q, gripper_angle=0.0):
    """
    Visualize robot mesh at a specific joint configuration
    Args:
        mesh_dir: directory containing arm .stl files
        gripper_dir: directory containing gripper .stl files
        q: joint angles in radians [q1, q2, q3, q4, q5, q6]
        gripper_angle: gripper opening angle in radians (0.0 = closed, 0.85 = fully open)
    """
    # Create a scene
    scene = trimesh.Scene()
    
    # Load arm meshes
    mesh_files = sorted(glob.glob(str(mesh_dir / "*.stl")))
    meshes = [trimesh.load(mf) for mf in mesh_files]
    
    # Load gripper meshes
    gripper_files = sorted(glob.glob(str(gripper_dir / "*.STL")))
    gripper_meshes = [trimesh.load(gf) for gf in gripper_files]
    
    # Colors for visualization
    colors = [[128,128,128,150], [255,0,0,150], [0,255,0,150], 
             [0,0,255,150], [255,255,0,150], [255,0,255,150], 
             [0,255,255,150]]
    gripper_color = [192,192,192,150]  # Silver color for gripper
    
    # Get transforms using our FK function with gripper
    transforms = fk_xarm6_torch(q, with_gripper=True)
    transforms_dict = {
        'base': np.eye(4),
        'link1': transforms[0].numpy(),
        'link2': transforms[1].numpy(),
        'link3': transforms[2].numpy(),
        'link4': transforms[3].numpy(),
        'link5': transforms[4].numpy(),
        'link6': transforms[5].numpy(),
        'gripper_base': transforms[6].numpy()  # Gripper base transform from FK
    }
    
    # Define gripper parts transforms relative to gripper base
    gripper_parts = {
        'base_link': np.eye(4),
        'left_outer_knuckle': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.035],
            [0, 0, 1, 0.059098],
            [0, 0, 0, 1]
        ]),
        'right_outer_knuckle': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.035],
            [0, 0, 1, 0.059098],
            [0, 0, 0, 1]
        ]),
        'left_finger': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.035465],
            [0, 0, 1, 0.042039],
            [0, 0, 0, 1]
        ]),
        'right_finger': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.035465],
            [0, 0, 1, 0.042039],
            [0, 0, 0, 1]
        ]),
        'left_inner_knuckle': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.02],
            [0, 0, 1, 0.074098],
            [0, 0, 0, 1]
        ]),
        'right_inner_knuckle': np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.02],
            [0, 0, 1, 0.074098],
            [0, 0, 0, 1]
        ])
    }
    
    # Add arm meshes with transforms
    for i, mesh in enumerate(meshes):
        link_name = Path(mesh_files[i]).stem.split('_')[0]
        if link_name in transforms_dict:
            mesh.apply_transform(transforms_dict[link_name])
        mesh.visual.face_colors = colors[i % len(colors)]
        scene.add_geometry(mesh)
    
    # Add gripper meshes using FK transform combined with part-specific transforms
    for i, mesh in enumerate(gripper_meshes):
        part_name = Path(gripper_files[i]).stem
        if part_name in gripper_parts:
            # Combine FK transform with part-specific transform
            transform = transforms_dict['gripper_base'] @ gripper_parts[part_name]
            mesh.apply_transform(transform)
            mesh.visual.face_colors = gripper_color
            scene.add_geometry(mesh)
    
    # Show the scene
    scene.show()

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    mesh_dir = project_root / "xarm_description" / "meshes" / "xarm6" / "visual"
    gripper_dir = project_root / "xarm_description" / "xarm_gripper" / "meshes"
    
    # Try different poses
    poses = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Home pose
        # torch.tensor([np.pi/4, 0, -np.pi/2, 0.0, 0.0, 0.0]),  # Bent pose
        # torch.tensor([np.pi/2, 0.0, -np.pi/4, np.pi/4, np.pi/2, 0.0])  # Another pose
    ]
    
    # Visualize each pose
    for i, pose in enumerate(poses):
        print(f"\nVisualizing pose {i+1}:")
        print(f"Joint angles (radians): {pose.numpy()}")
        visualize_robot_meshes(mesh_dir, gripper_dir, pose, gripper_angle=0.0) 