import numpy as np
from numpy import pi
import torch

class XArmFK:
    def __init__(self, device='cuda', with_gripper=True):
        self.device = device
        self.with_gripper = with_gripper
        # DH parameters [a, alpha, d, theta_offset]
        self.dh_params = [
            [0,      -pi/2,  0.267,   0],      # Joint 1: z-offset from base
            [0.2895, 0,      0,      -1.385],  # Joint 2: x offset, rotated
            [0.0775, -pi/2,  0,      1.385],   # Joint 3: length of link2
            [0,      pi/2,   0.3425, 0],       # Joint 4: z offset
            [0.076,  -pi/2,  0,      0],       # Joint 5: wrist rotation
            [0,      0,      0.097,  0]        # Joint 6: end-effector length
        ]
        
        if with_gripper:
            # Add gripper DH parameters (fixed transform relative to link6)
            # The gripper extends about 0.145m from link6 
            self.dh_params.append([0, 0, 0.145, 0])  # Gripper: additional length in z direction
        
        # Joint limits in radians - move to device
        self.joint_limits = torch.tensor([
            [-2*pi,    2*pi],     # Joint 1
            [-2.059,   2.059],    # Joint 2
            [-3.927,   0.191],    # Joint 3
            [-1.745,   3.927],    # Joint 4
            [-2.059,   2.059],    # Joint 5
            [-6.283,   6.283]     # Joint 6
        ], device=self.device)
        
        self.num_joints = 6
        self.num_links = 7 if with_gripper else 6  # Include gripper if specified
        self.fk_mask = [True] * self.num_links  # All joints included
        
    def dh_transform(self, a, alpha, d, theta):
        """
        Compute the homogeneous transformation matrix from DH parameters
        """
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(torch.tensor(alpha, device=self.device))
        sa = torch.sin(torch.tensor(alpha, device=self.device))
        
        return torch.stack([
            torch.stack([ct, -st*ca, st*sa, a*ct], dim=-1),
            torch.stack([st, ct*ca, -ct*sa, a*st], dim=-1),
            torch.stack([torch.zeros_like(theta), sa.repeat(len(theta)), ca.repeat(len(theta)), d*torch.ones_like(theta)], dim=-1),
            torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1)
        ], dim=-2)

    def fkine(self, q):
        """
        Forward kinematics for xArm
        Args:
            q: joint angles [batch_size, 6]
        Returns:
            Positions of each joint [batch_size, num_links, 3]
        """
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32, device=self.device)
            
        if q.dim() == 1:
            q = q.unsqueeze(0)  # Add batch dimension
            
        batch_size = q.shape[0]
        positions = []
        T = torch.eye(4, device=self.device).repeat(batch_size, 1, 1)
        
        # Add base joint position
        positions.append(torch.tensor([0., 0., 0.267], device=self.device).repeat(batch_size, 1))
        
        # Process arm joints
        for i in range(self.num_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            curr_transform = self.dh_transform(a, alpha, d, q[:, i] + theta_offset)
            T = torch.bmm(T, curr_transform)
            positions.append(T[:, :3, 3])
        
        # Add gripper position if needed
        if self.with_gripper:
            a, alpha, d, theta_offset = self.dh_params[-1]
            T = torch.bmm(T, self.dh_transform(a, alpha, d, torch.zeros_like(q[:, 0])))
            positions.append(T[:, :3, 3])
            
        return torch.stack(positions, dim=1)  # [batch_size, num_links, 3]

    def check_joint_limits(self, q):
        """
        Check if joint angles are within limits
        Args:
            q: joint angles [batch_size, 6]
        Returns:
            Boolean tensor indicating if joints are within limits
        """
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)
            
        if q.dim() == 1:
            q = q.unsqueeze(0)
            
        within_limits = torch.logical_and(
            q >= self.joint_limits[:, 0],
            q <= self.joint_limits[:, 1]
        )
        return within_limits.all(dim=1)
    


def compute_joint_space_diagonal():
    # Extract the joint limits from the code
    joint_ranges = [
        [(-2*pi),    (2*pi)],    # Joint 1: range of 4π
        [-2.059,     2.059],     # Joint 2: range of 4.118
        [-3.927,     0.191],     # Joint 3: range of 4.118
        [-1.745,     3.927],     # Joint 4: range of 5.672
        [-2.059,     2.059],     # Joint 5: range of 4.118
        [-6.283,     6.283]      # Joint 6: range of 12.566
    ]
    
    # Calculate the length of each dimension (upper - lower)
    ranges = np.array([upper - lower for lower, upper in joint_ranges])
    
    # Compute diagonal length using Euclidean norm
    diagonal_length = np.sqrt(np.sum(ranges**2))
    
    return diagonal_length

if __name__ == "__main__":
    diagonal = compute_joint_space_diagonal()
    print(f"Joint space diagonal length: {diagonal:.3f} radians")