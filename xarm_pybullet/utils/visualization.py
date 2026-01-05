import torch
import numpy as np
import trimesh
import mcubes
from pathlib import Path
import glob
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.xarm6_differentiable_fk import fk_xarm6_torch
from data.generate_sdf import combine_gripper_meshes
from robot_sdf import RobotSDF
from sdf_training.network import SDFNetwork

class SDFVisualizer:
    def __init__(self, device='cuda'):
        self.device = device
        self.robot_sdf = RobotSDF(device)
        
        # Load base SDF model separately
        model_dir = Path(__file__).parent.parent / "trained_models"
        base_checkpoint = torch.load(model_dir / "link_0" / "best_model.pt")
        self.base_sdf_model = SDFNetwork().to(device)
        self.base_sdf_model.load_state_dict(base_checkpoint['model_state_dict'])
        self.base_sdf_model.eval()
        
        # Load base scaling parameters
        data_dir = Path(__file__).parent.parent / "data" / "sdf_data"
        base_data = np.load(data_dir / "link_0_link0.npz")
        self.base_offset = torch.from_numpy(base_data['original_center']).float().to(device)
        self.base_scale = torch.tensor(base_data['original_scale'], dtype=torch.float32).to(device)
        
        # Load meshes for comparison
        project_root = Path(__file__).parent.parent
        mesh_dir = project_root / "xarm_description" / "meshes" / "xarm6" / "visual"
        gripper_dir = project_root / "xarm_description" / "xarm_gripper" / "meshes"
        
        # Load arm meshes
        mesh_files = sorted(glob.glob(str(mesh_dir / "*.stl")))
        self.meshes = [trimesh.load(mf) for mf in mesh_files]
        
        # Load gripper mesh
        self.gripper_mesh = combine_gripper_meshes(gripper_dir)
        self.meshes.append(self.gripper_mesh)
        
        # Colors for visualization (extended for gripper)
        self.mesh_colors = [[128,128,128,100], [255,0,0,100], [0,255,0,100], 
                          [0,0,255,100], [255,255,0,100], [255,0,255,100], 
                          [0,255,255,100], [192,192,192,100]]  # Added gripper color
        self.sdf_colors = [[128,128,128,255], [255,0,0,255], [0,255,0,255], 
                          [0,0,255,255], [255,255,0,255], [255,0,255,255], 
                          [0,255,255,255], [192,192,192,255]]  # Added gripper color
    
    def extract_level_surface_for_link(self, model, resolution=64):
        """Extract zero-level surface for a single link"""
        # Create grid of points
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        z = np.linspace(-1, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Reshape to (N, 3) array
        points = torch.FloatTensor(np.stack([X, Y, Z], axis=-1)).reshape(-1, 3).to(self.device)
        
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
    
    def visualize_sdf(self, joint_angles, show_meshes=False, resolution=64):
        """Visualize SDF zero-level surface and optionally show meshes"""
        scene = trimesh.Scene()
        
        # First visualize base SDF (static)
        vertices, triangles = self.extract_level_surface_for_link(self.base_sdf_model, resolution)
        vertices = vertices * self.base_scale.cpu().numpy()
        vertices = vertices + self.base_offset.cpu().numpy()
        base_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        base_mesh.visual.face_colors = self.sdf_colors[0]  # Base color
        scene.add_geometry(base_mesh)
        
        # Get transforms using FK
        q = joint_angles.cpu().reshape(-1)
        if len(q) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(q)}")
        transforms = fk_xarm6_torch(q, with_gripper=True)
        transforms_dict = {
            'link1': transforms[0].cpu().numpy(),
            'link2': transforms[1].cpu().numpy(),
            'link3': transforms[2].cpu().numpy(),
            'link4': transforms[3].cpu().numpy(),
            'link5': transforms[4].cpu().numpy(),
            'link6': transforms[5].cpu().numpy(),
            'link7': transforms[5].cpu().numpy()   # gripper base frame
        }
        
        # Extract and transform level surface for each movable link
        for i, model in enumerate(self.robot_sdf.models):
            vertices, triangles = self.extract_level_surface_for_link(model, resolution)
            vertices = vertices * self.robot_sdf.scales[i].cpu().numpy()
            vertices = vertices + self.robot_sdf.offsets[i].cpu().numpy()
            
            link_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            link_name = f'link{i+1}'
            if link_name in transforms_dict:
                link_mesh.apply_transform(transforms_dict[link_name])
            link_mesh.visual.face_colors = self.sdf_colors[i+1]
            scene.add_geometry(link_mesh)
        
        if show_meshes:
            # Add original meshes - skip base mesh (index 0)
            for i, mesh in enumerate(self.meshes[1:], start=1):  # Start from index 1
                link_name = Path(mesh.metadata['file_name']).stem.split('_')[0]
                if link_name in transforms_dict:
                    mesh_copy = mesh.copy()
                    mesh_copy.apply_transform(transforms_dict[link_name])
                    mesh_copy.visual.face_colors = self.mesh_colors[i]
                    scene.add_geometry(mesh_copy)

        return scene

    def create_scene(self, joint_angles, show_meshes=True, resolution=64):
        """Create scene with robot SDF visualization without showing it"""
        # Create scene
        scene = trimesh.Scene()
        
        # Get transforms using FK
        q = joint_angles.cpu().reshape(-1)
        if len(q) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(q)}")
        transforms = fk_xarm6_torch(q)
        transforms_dict = {
            'base': np.eye(4),
            'link1': transforms[0].cpu().numpy(),
            'link2': transforms[1].cpu().numpy(),
            'link3': transforms[2].cpu().numpy(),
            'link4': transforms[3].cpu().numpy(),
            'link5': transforms[4].cpu().numpy(),
            'link6': transforms[5].cpu().numpy(),
            'link7': transforms[5].cpu().numpy()
        }
        
        # Extract and transform level surface for each link
        for i, model in enumerate(self.robot_sdf.models):
            vertices, triangles = self.extract_level_surface_for_link(model, resolution)
            vertices = vertices * self.robot_sdf.scales[i].cpu().numpy()
            vertices = vertices + self.robot_sdf.offsets[i].cpu().numpy()
            
            link_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            link_name = f'link{i}'
            if link_name in transforms_dict:
                link_mesh.apply_transform(transforms_dict[link_name])
            link_mesh.visual.face_colors = self.sdf_colors[i]
            scene.add_geometry(link_mesh)
        
        if show_meshes:
            for i, mesh in enumerate(self.meshes):
                link_name = Path(mesh.metadata['file_name']).stem.split('_')[0]
                if link_name in transforms_dict:
                    mesh_copy = mesh.copy()
                    mesh_copy.apply_transform(transforms_dict[link_name])
                    mesh_copy.visual.face_colors = self.mesh_colors[i]
                    scene.add_geometry(mesh_copy)
        
        return scene

if __name__ == "__main__":
    device = 'cuda'
    visualizer = SDFVisualizer(device)
    
    # Test different poses
    poses = [
        torch.tensor([0.0, 0., 0., 0.0, 0.0, 0.0], device=device),  # Home pose
        torch.tensor([-2.6487477, -1.0901253, -1.1788788, 2.7880065, 1.3651353, 3.2989674], device=device),  # Bent pose
        #torch.tensor([np.pi/2, 0.0, -np.pi/4, np.pi/4, np.pi/2, 0.0], device=device)  # Another pose
    ]
    
    # Visualize each pose
    for i, pose in enumerate(poses):
        print(f"\nVisualizing pose {i+1}:")
        print(f"Joint angles (radians): {pose.cpu().numpy()}")
        scene = visualizer.visualize_sdf(pose, show_meshes=False, resolution=64)
        scene.show()
