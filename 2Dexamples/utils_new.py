import numpy as np
from typing import Tuple, List


def forward_kinematics_analytical(theta1: float, theta2: float, l1: float = 2.0, l2: float = 2.0) -> Tuple[float, float]:
    """
    Compute forward kinematics for a 2-link planar arm
    
    Args:
        theta1: angle of first joint (rad)
        theta2: angle of second joint (rad)
        l1: length of first link (default 2.0)
        l2: length of second link (default 2.0)
    
    Returns:
        (x, y): end-effector position
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

def forward_kinematics_ee_from_array(joint_angles: np.ndarray, l1: float = 2.0, l2: float = 2.0) -> np.ndarray:
    theta1, theta2 = float(joint_angles[0]), float(joint_angles[1])
    x,y = forward_kinematics_analytical(theta1, theta2, l1, l2)
    return np.array([x, y], dtype=np.float32)

def forward_kinematics_chain_from_array(joint_angles: np.ndarray, l1: float = 2.0, l2: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    theta1, theta2 = float(joint_angles[0]), float(joint_angles[1])
    
    x = np.zeros(3, dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)

    # Joint 1
    x[1] = l1 * np.cos(theta1)
    y[1] = l1 * np.sin(theta1)

    # End-Effector
    x[2], y[2] = forward_kinematics_analytical(theta1, theta2, l1, l2)
    return x, y

def inverse_kinematics_analytical(x: float, y: float, l1: float = 2.0, l2: float = 2.0) -> List[np.ndarray]:
    """
    Compute inverse kinematics for a 2-link planar arm
    
    Args:
        x: desired end-effector x position
        y: desired end-effector y position
        l1: length of first link (default 2.0)
        l2: length of second link (default 2.0)
    
    Returns:
        List of possible configurations [theta1, theta2] that reach the target
        Empty list if no solution exists
    """
    solutions = []
    
    # Distance from base to target
    r = np.sqrt(x**2 + y**2)
    
    # Check if target is reachable
    if r > l1 + l2 or r < abs(l1 - l2):
        print(f"Target position ({x}, {y}) is not reachable!")
        return solutions
    
    # Compute theta2 using cosine law
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if cos_theta2 > 1 or cos_theta2 < -1:
        print(f"No solution exists for target ({x}, {y})")
        return solutions
    
    # Two possible solutions for theta2 (elbow-up and elbow-down)
    theta2_1 = np.arccos(cos_theta2)
    theta2_2 = -theta2_1
    
    # Compute corresponding theta1 values
    for theta2 in [theta2_1, theta2_2]:
        # Using atan2 for correct quadrant
        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        # Normalize angles to [-pi, pi]
        theta1 = np.mod(theta1 + np.pi, 2 * np.pi) - np.pi
        theta2 = np.mod(theta2 + np.pi, 2 * np.pi) - np.pi
        
        solutions.append(np.array([theta1, theta2], dtype=np.float32))
    
    return solutions

def test_kinematics():
    """Test the kinematics implementations"""
    # Test some positions
    test_positions = [
        (2.0, 2.0),
        (0.0, 3.0),
        (-2.0, 2.0),
        (1.0, -1.0)
    ]
    
    print("\nTesting kinematics functions:")
    for x, y in test_positions:
        print(f"\nTarget position: ({x}, {y})")
        solutions = inverse_kinematics_analytical(x, y)
        
        for i, sol in enumerate(solutions):
            print(f"Solution {i+1}: theta1 = {sol[0]:.3f}, theta2 = {sol[1]:.3f}")
            
            # Verify solution using forward kinematics
            x_check, y_check = forward_kinematics_analytical(sol[0], sol[1])
            error = np.sqrt((x - x_check)**2 + (y - y_check)**2)
            print(f"FK check: ({x_check:.3f}, {y_check:.3f}), error: {error:.6f}")

if __name__ == "__main__":
    test_kinematics()
