import torch
import numpy as np

class Robot2D:
    def __init__(self, link_lengths=[2.0, 2.0], device='cpu'):
        """
        Initialize 2D 2-link robot
        Args:
            link_lengths: List of link lengths [l1, l2]
            device: torch device
        """
        self.device = device
        self.link_lengths = torch.tensor(link_lengths, device=device)
    
    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics for 2D 2-link robot
        Args:
            joint_angles: [batch_size, 2] joint angles
        Returns:
            joints: [batch_size, 3, 2] joint positions (including base)
        """
        batch_size = joint_angles.shape[0]
        
        # Initialize joint positions
        joints = torch.zeros(batch_size, 3, 2, device=self.device)
        
        # Compute cumulative angles
        theta1 = joint_angles[:, 0]
        theta2 = theta1 + joint_angles[:, 1]
        
        # Compute joint positions
        joints[:, 1, 0] = self.link_lengths[0] * torch.cos(theta1)
        joints[:, 1, 1] = self.link_lengths[0] * torch.sin(theta1)
        
        joints[:, 2, 0] = joints[:, 1, 0] + self.link_lengths[1] * torch.cos(theta2)
        joints[:, 2, 1] = joints[:, 1, 1] + self.link_lengths[1] * torch.sin(theta2)
        
        return joints

class RobotSDF:
    def __init__(self, robot, device='cpu'):
        """
        Initialize robot SDF calculator
        Args:
            robot: Robot2D instance
            device: torch device
        """
        self.robot = robot
        self.device = device

    def point_to_segment_distance(self, points, segment_start, segment_end):
        """
        Calculate minimum distance from points to line segment
        Args:
            points: [batch_size, num_points, 2] points
            segment_start: [batch_size, 2] segment start points
            segment_end: [batch_size, 2] segment end points
        Returns:
            distances: [batch_size, num_points] distances
        """
        segment = segment_end - segment_start  # [B, 2]
        length_sq = torch.sum(segment**2, dim=1, keepdim=True)  # [B, 1]
        
        # Project points onto line
        point_vec = points - segment_start.unsqueeze(1)  # [B, N, 2]
        t = torch.sum(point_vec * segment.unsqueeze(1), dim=2) / length_sq  # [B, N]
        t = torch.clamp(t, min=0.0, max=1.0)  # [B, N]
        
        # Compute closest points on segments
        closest = segment_start.unsqueeze(1) + t.unsqueeze(2) * segment.unsqueeze(1)  # [B, N, 2]
        
        # Compute distances
        distances = torch.norm(points - closest, dim=2)  # [B, N]
        return distances

    def query_sdf(self, points, joint_angles):
        """
        Query SDF for points given joint angles
        Args:
            points: [batch_size, num_points, 2] points
            joint_angles: [batch_size, 2] joint angles
        Returns:
            sdf: [batch_size, num_points] distances
            link_ids: [batch_size, num_points] closest link indices
        """
        # Get joint positions
        joints = self.robot.forward_kinematics(joint_angles)
        
        # Compute distances to each segment
        distances = []
        for i in range(2):
            segment_dist = self.point_to_segment_distance(
                points,
                joints[:, i],
                joints[:, i + 1]
            )
            distances.append(segment_dist)
        
        # Stack and find minimum distances
        distances = torch.stack(distances, dim=1)  # [B, 2, N]
        sdf, link_ids = torch.min(distances, dim=1)  # [B, N]
        
        return sdf, link_ids

