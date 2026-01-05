import torch
import numpy as np

class RobotSDF:
    def __init__(self, device='cuda'):
        self.device = device
        self.link_length = 2.0  # Fixed link length for each segment
        
    def forward_kinematics(self, joint_angles):
        """
        Compute robot link positions given joint angles
        Args:
            joint_angles: [B, 2] tensor of joint angles
        Returns:
            positions: tuple of (joint1_pos, joint2_pos, end_pos) each [B, 2]
        """
        batch_size = joint_angles.shape[0]
        
        # Base position
        joint1_pos = torch.zeros(batch_size, 2, device=self.device)
        
        # First joint position
        angle1 = joint_angles[:, 0]
        joint2_pos = torch.stack([
            self.link_length * torch.cos(angle1),
            self.link_length * torch.sin(angle1)
        ], dim=1)
        
        # End effector position
        angle2 = angle1 + joint_angles[:, 1]
        end_pos = joint2_pos + torch.stack([
            self.link_length * torch.cos(angle2),
            self.link_length * torch.sin(angle2)
        ], dim=1)
        
        return joint1_pos, joint2_pos, end_pos
    
    def point_to_segment_distance(self, points, segment_start, segment_end):
        """
        Compute minimum distance from points to line segment
        Args:
            points: [B, N, 2] points
            segment_start: [B, 2] segment start positions
            segment_end: [B, 2] segment end positions
        Returns:
            distances: [B, N] distances from points to segment
        """
        # Convert to [B, 1, 2] for broadcasting
        segment_start = segment_start.unsqueeze(1)  # [B, 1, 2]
        segment_end = segment_end.unsqueeze(1)      # [B, 1, 2]
        
        # Vector from start to end
        segment = segment_end - segment_start       # [B, 1, 2]
        
        # Vector from start to point
        point_vec = points - segment_start         # [B, N, 2]
        
        # Length of segment squared
        segment_length_sq = torch.sum(segment**2, dim=2, keepdim=True)  # [B, 1, 1]
        
        # Projection of point_vec onto segment
        t = torch.sum(point_vec * segment, dim=2, keepdim=True) / (segment_length_sq + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        
        # Closest point on segment
        closest_point = segment_start + t * segment
        
        # Distance to closest point
        distances = torch.norm(points - closest_point, dim=2)
        
        return distances
    
    def query_sdf(self, points, joint_angles):
        """
        Query SDF for points given joint angles
        Args:
            points: [B, N, 2] tensor of workspace points
            joint_angles: [B, 2] tensor of joint angles
        Returns:
            sdf_values: [B, N] tensor of SDF values
        """
        points = points.to(dtype=torch.float32, device=self.device)
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        # Get robot link positions
        joint1_pos, joint2_pos, end_pos = self.forward_kinematics(joint_angles)
        
        # Compute distances to both segments
        distances1 = self.point_to_segment_distance(points, joint1_pos, joint2_pos)
        distances2 = self.point_to_segment_distance(points, joint2_pos, end_pos)
        
        # Take minimum distance
        sdf_values = torch.minimum(distances1, distances2)
        
        return sdf_values

    def test_sdf(self):
        """Test the SDF computation"""
        # Create test inputs with batches
        points = torch.tensor([
            [[0.3, 0.2],
             [1.5, 1.5],
             [3.0, 0.0]],
            [[0.0, 1.0],
             [2.0, 2.0],
             [1.0, 3.0]]
        ], device=self.device)  # [2, 3, 2]
        
        joint_angles = torch.tensor([
            [0.0, 0.0],
            [np.pi/4, np.pi/3]
        ], device=self.device)  # [2, 2]
        
        # Get SDF values
        sdf_values = self.query_sdf(points, joint_angles)
        
        print("\nTest Results:")
        print(f"SDF values shape: {sdf_values.shape}")  # Should be [2, 3]
        
        print("\nSample values:")
        for b in range(points.shape[0]):
            print(f"\nBatch {b}:")
            for i in range(points.shape[1]):
                print(f"\nPoint {points[b,i].cpu().numpy()}:")
                print(f"SDF value: {sdf_values[b,i].item():.6f}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    robot_sdf = RobotSDF(device)
    robot_sdf.test_sdf()