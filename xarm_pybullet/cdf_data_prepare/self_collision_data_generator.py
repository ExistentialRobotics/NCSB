import numpy as np
import torch
import pybullet as p
import pybullet_data
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add parent directory to path to import xarm modules
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, '..'))

from models.xarm_model import XArmFK

class SelfCollisionDataGenerator:
    def __init__(self, device='cuda', gui=False):
        # Initialize PyBullet
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load xArm with self-collision enabled
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.robot_id = p.loadURDF(
            os.path.join(CUR_DIR, "..", "xarm_description/xarm6_with_gripper.urdf"),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=flags
        )
        
        # Print robot information
        # self.print_robot_info()
        
        # Include base through gripper_base (8 links total)
        self.num_arm_links = 8  # base(0) + link1-6(1-6) + gripper_base(7)
        
        print("\nChecking collisions between the following links:")
        for i in range(self.num_arm_links):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8') if i > 0 else "base_link"
            print(f"Link {i}: {link_name}")
        
        # Enable collisions only between arm links
        for joint_a in range(self.num_arm_links):
            for joint_b in range(joint_a + 1, self.num_arm_links):
                p.setCollisionFilterPair(self.robot_id, self.robot_id, joint_a, joint_b, 1)
        
        # Disable collisions for gripper links
        num_joints = p.getNumJoints(self.robot_id)
        for joint_a in range(self.num_arm_links, num_joints):
            for joint_b in range(joint_a + 1, num_joints):
                p.setCollisionFilterPair(self.robot_id, self.robot_id, joint_a, joint_b, 0)
        
        # Initialize FK model for additional computations if needed
        self.robot_fk = XArmFK(device=device)
        self.joint_limits = self.robot_fk.joint_limits.cpu().numpy()
        
        # Parameters for data generation
        self.epsilon = 0.01  # Distance threshold for collision
        self.batch_size = 10000  # Number of configurations to test at once
        
        # Print joint limits from PyBullet
        print("\nJoint limits from PyBullet:")
        for i in range(1, 7):  # 6 arm joints
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            lower_limit = joint_info[8]  # lower limit
            upper_limit = joint_info[9]  # upper limit
            print(f"Joint {i} ({joint_name}): [{lower_limit:.3f}, {upper_limit:.3f}] rad")
        
    def check_self_collision(self, q: np.ndarray) -> Tuple[bool, List[List[int]]]:
        """Check if a single configuration causes self-collision and return relevant joint indices"""
        # Set joint positions
        for i in range(6):
            p.resetJointState(self.robot_id, i+1, q[i])
        
        p.stepSimulation()
        
        relevant_joints_list = []
        for link_a in range(self.num_arm_links):
            for link_b in range(link_a + 1, self.num_arm_links):
                points = p.getContactPoints(self.robot_id, self.robot_id, link_a, link_b)
                if points:
                    # Determine relevant joints based on link indices
                    if link_a == 0:
                        joint_start = 0
                    else:
                        joint_start = link_a - 1
                    
                    joint_end = min(5, link_b - 1)
                    relevant_joints = list(range(joint_start, joint_end + 1))
                    
                    # Only store the relevant joints
                    relevant_joints_list.append(relevant_joints)
        
        return len(relevant_joints_list) > 0, relevant_joints_list
    
    def check_self_collisions_batch(self, configs: np.ndarray) -> Tuple[np.ndarray, List[List[List[int]]]]:
        """Check multiple configurations and return collision masks and link info"""
        collision_mask = np.zeros(len(configs), dtype=bool)
        collision_info_list = []
        
        for i, q in enumerate(configs):
            has_collision, info = self.check_self_collision(q)
            collision_mask[i] = has_collision
            collision_info_list.append(info)
        
        return collision_mask, collision_info_list
    
    def sample_configurations(self, n_samples: int) -> np.ndarray:
        """Sample random configurations within joint limits"""
        configs = np.random.uniform(
            low=self.joint_limits[:, 0],
            high=self.joint_limits[:, 1],
            size=(n_samples, 6)
        )
        return configs
    
    def generate_collision_dataset(self, n_samples: int = 100000, n_refined: int = 500, save_path: str = None) -> Dict:
        """Generate dataset of self-colliding configurations and refine using FPS
        
        Args:
            n_samples: Number of initial samples to test
            n_refined: Number of configurations to keep per joint combination after FPS
            save_path: Path to save the dataset
        """
        print(f"\nGenerating self-collision dataset with {n_samples} samples...")
        
        joint_combo_configs = {}  # key: tuple of joints, value: list of configs
        
        start_time = time.time()
        batches_processed = 0
        total_collected = 0
        
        while total_collected < n_samples:
            configs = self.sample_configurations(self.batch_size)
            collision_mask, collision_info = self.check_self_collisions_batch(configs)
            
            for config, is_colliding, joints_info in zip(configs, collision_mask, collision_info):
                if is_colliding:
                    min_joints = tuple(min(joints_info, key=len))
                    if min_joints not in joint_combo_configs:
                        joint_combo_configs[min_joints] = []
                    joint_combo_configs[min_joints].append(config)
                    total_collected += 1
            
            batches_processed += 1
            if batches_processed % 10 == 0:
                print(f"\nProcessed {batches_processed} batches ({total_collected} configs)")
        
        # Apply FPS to each joint combination
        print("\nApplying FPS to each joint combination:")
        colliding_configs = []
        relevant_joints = []
        
        for joints, configs in joint_combo_configs.items():
            configs = np.array(configs)
            n_to_sample = min(n_refined, len(configs))
            refined = self.farthest_point_sampling(configs, n_to_sample)
            print(f"Joint combination {list(joints)}: {len(configs)} → {len(refined)} configs")
            
            # Add to the flat lists
            colliding_configs.extend(refined)
            relevant_joints.extend([list(joints)] * len(refined))
        
        # Convert to numpy arrays
        colliding_configs = np.array(colliding_configs)
        
        # Save dataset in the format expected by self_collision_cdf.py
        if save_path:
            np.save(save_path, {
                'colliding_configs': colliding_configs,
                'relevant_joints': relevant_joints
            })
            print(f"\nDataset saved to: {save_path}")
        
        return {
            'colliding_configs': colliding_configs,
            'relevant_joints': relevant_joints
        }
    
    def farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Farthest point sampling algorithm to get diverse configurations
        
        Args:
            points: Array of configurations [N, 6]
            n_samples: Number of samples to select
        Returns:
            selected_points: Array of selected configurations [n_samples, 6]
        """
        n_points = len(points)
        if n_points <= n_samples:
            return points
        
        # Initialize with random point
        selected_indices = [np.random.randint(n_points)]
        min_distances = np.full(n_points, np.inf)
        
        # Iteratively select farthest points
        for i in range(1, n_samples):
            # Compute distances to last selected point
            last_point = points[selected_indices[-1]]
            
            # Compute distances in configuration space (considering joint angle wrapping)
            diff = points - last_point
            # Wrap angle differences to [-pi, pi]
            diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
            distances = np.linalg.norm(diff, axis=1)
            
            # Update minimum distances
            min_distances = np.minimum(min_distances, distances)
            
            # Select point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
            min_distances[next_idx] = 0  # Prevent reselection
            
            if i % 100 == 0:
                print(f"Selected {i}/{n_samples} diverse configurations")
        
        return points[selected_indices]

    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect(self.client)

    def visualize_configuration(self, q: np.ndarray, pause_time: float = 0.5):
        """Visualize a configuration and print collision information"""
        # Set joint positions
        for i in range(6):
            p.resetJointState(self.robot_id, i+1, q[i])
        
        p.stepSimulation()
        
        # Get collision info only between arm links
        contact_points = []
        for joint_a in range(self.num_arm_links):
            for joint_b in range(joint_a + 1, self.num_arm_links):
                points = p.getContactPoints(self.robot_id, self.robot_id, joint_a, joint_b)
                if points:
                    contact_points.extend(points)
        
        if contact_points:
            print("\nCollision detected!")
            print("Contact points:")
            for point in contact_points:
                linkA = point[3]  # Index of first colliding link
                linkB = point[4]  # Index of second colliding link
                print(f"Link {linkA} colliding with Link {linkB}")
                # Get link names for better understanding
                linkA_name = p.getJointInfo(self.robot_id, linkA)[12].decode('utf-8')
                linkB_name = p.getJointInfo(self.robot_id, linkB)[12].decode('utf-8')
                print(f"({linkA_name} colliding with {linkB_name})")
        else:
            print("\nNo collision detected")
            
        # Print current joint angles
        print("\nJoint angles (radians):")
        for i, angle in enumerate(q):
            print(f"Joint {i+1}: {angle:.3f}")
        
        time.sleep(pause_time)

    def print_robot_info(self):
        """Print detailed information about robot links and joints from PyBullet"""
        print("\nRobot Link and Joint Information:")
        print("-" * 50)
        
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Total number of joints: {num_joints}")
        
        print("\nDetailed Joint/Link Information:")
        print("Index  Joint Name        Link Name        Parent Link  Joint Type")
        print("-" * 70)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_index = joint_info[0]
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            parent_idx = joint_info[16]
            joint_type = joint_info[2]  # 0:REVOLUTE, 1:PRISMATIC, 4:FIXED
            
            joint_type_str = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}
            
            print(f"{joint_index:5d}  {joint_name:16s} {link_name:16s} {parent_idx:11d}  {joint_type_str.get(joint_type, str(joint_type))}")
        
        print("\nLink States:")
        print("Index  Link Name        World Position          World Orientation")
        print("-" * 70)
        
        for i in range(-1, num_joints):  # -1 is the base link
            if i == -1:
                link_state = p.getBasePositionAndOrientation(self.robot_id)
                link_name = "base_link"
            else:
                link_state = p.getLinkState(self.robot_id, i)
                link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            
            pos = link_state[0]
            orn = link_state[1]
            print(f"{i:5d}  {link_name:16s} {pos!r:20s} {orn!r}")



def main():
    generator = SelfCollisionDataGenerator(gui=False)
    try:
        save_path = os.path.join(CUR_DIR, '..', 'data', 'refined_self_collision_data.npy')
        dataset = generator.generate_collision_dataset(
            n_samples=100000,
            n_refined=500,
            save_path=save_path
        )
    finally:
        generator.close()

if __name__ == "__main__":
    main() 