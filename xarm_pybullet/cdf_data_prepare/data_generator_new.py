import torch
import os
import numpy as np
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
from torch.quasirandom import SobolEngine

# Add parent directory to path to import xarm modules
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, '..'))

from models.xarm_model import XArmFK
from robot_sdf import RobotSDF

class XArmCDFDataGeneratorNew:
    def __init__(self, device: str = 'cuda'):
        # Initialize robot models
        self.robot_fk = XArmFK(device=device)
        self.robot_sdf = RobotSDF(device=device)
        
        # Store joint limits
        self.q_max = self.robot_fk.joint_limits[:, 1]
        self.q_min = self.robot_fk.joint_limits[:, 0]
        
        # Device
        self.device = device
        
        # Data generation parameters
        self.workspace = [[-0.6, -0.6, 0.0],  # min x,y,z
                         [0.6, 0.6, 1.0]]      # max x,y,z
        
        # Grid parameters
        self.n_grid = 50  # points per dimension for workspace discretization
        self.contact_threshold = 0.005  # threshold for contact detection
        self.min_contacts_per_point = 100  # minimum number of contact configs needed
        
        # Update save directory
        self.save_dir = os.path.join(CUR_DIR, '..', 'data', 'cdf_data')
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_contact_database(self, n_samples: int = 10000) -> Dict[Tuple[float, float, float], List[np.ndarray]]:
        """
        Generate database of contact configurations using Sobol sequences
        """
        print("\n=== Generating Contact Configuration Database ===")
        print(f"Sampling {n_samples} configurations using Sobol sequence...")
        
        # Initialize Sobol sequence generator
        sobol_engine = SobolEngine(dimension=6, scramble=True)
        
        # Create grid of points in workspace
        x = torch.linspace(self.workspace[0][0], self.workspace[1][0], self.n_grid, device=self.device)
        y = torch.linspace(self.workspace[0][1], self.workspace[1][1], self.n_grid, device=self.device)
        z = torch.linspace(self.workspace[0][2], self.workspace[1][2], self.n_grid, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        
        # Initialize contact database
        contact_db = defaultdict(list)
        start_time = time.time()
        batch_size = 5  # Process configurations in batches
        total_contacts = 0
        
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            
            # Generate quasi-random configurations
            sobol_samples = sobol_engine.draw(current_batch).to(self.device)
            q_batch = sobol_samples * (self.q_max - self.q_min) + self.q_min
            
            # Compute SDF values
            with torch.no_grad():
                points_expanded = grid_points.unsqueeze(0).expand(current_batch, -1, -1)
                sdf_values = self.robot_sdf.query_sdf(
                    points=points_expanded,
                    joint_angles=q_batch,
                    return_gradients=False
                )
            
            contact_mask = torch.abs(sdf_values) < self.contact_threshold
            
            # Store contact configurations
            for b in range(current_batch):
                point_indices = torch.where(contact_mask[b])[0]
                total_contacts += len(point_indices)
                for idx in point_indices:
                    point_tuple = tuple(grid_points[idx].cpu().numpy())
                    contact_db[point_tuple].append(q_batch[b].cpu().numpy())
            
            # Print progress
            if (i + current_batch) % 10000 == 0 or (i + current_batch) == n_samples:
                elapsed_time = time.time() - start_time
                progress = (i + current_batch) / n_samples * 100
                print(f"\nProgress: {progress:.1f}%")
                print(f"Configurations processed: {i + current_batch}/{n_samples}")
                print(f"Points with contacts: {len(contact_db)}")
                print(f"Total contact configs found: {total_contacts}")
                print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        
        # Filter points with insufficient contacts
        filtered_db = {
            point: configs 
            for point, configs in contact_db.items() 
            if len(configs) >= self.min_contacts_per_point
        }
        
        print(f"\n=== Contact Database Generation Complete ===")
        print(f"Total points with sufficient contacts: {len(filtered_db)}")
        print(f"Average contacts per point: {np.mean([len(configs) for configs in filtered_db.values()]):.1f}")
        
        return filtered_db

    def generate_training_data(self, 
                             contact_db: Dict[Tuple[float, float, float], List[np.ndarray]], 
                             n_train_configs: int = 1000,
                             cdf_threshold: float = 0.8) -> List[Dict]:
        """
        Generate training data using the contact configuration database
        Args:
            contact_db: Database of contact configurations
            n_train_configs: Number of training configurations to sample
            cdf_threshold: Maximum allowed CDF value (exclude pairs with higher values)
        """
        print("\n=== Generating Training Data ===")
        
        # 1. Sample training configurations
        q_train = torch.rand(n_train_configs, 6, device=self.device) * (self.q_max - self.q_min) + self.q_min
        
        # 2. Generate training pairs
        training_pairs = []
        excluded_pairs = 0
        start_time = time.time()
        
        for i, q in enumerate(q_train):
            q_np = q.cpu().numpy()
            valid_pairs_this_config = 0
            
            # Process each point in the database
            for point, contact_configs in contact_db.items():
                contact_configs = np.array(contact_configs)
                
                # Compute minimum distance to any contact configuration
                distances = np.linalg.norm(contact_configs - q_np, axis=1)
                min_distance = np.min(distances)
                
                # Only store if CDF value is below threshold
                if min_distance < cdf_threshold:
                    training_pairs.append({
                        'joint_angles': q_np,
                        'point': np.array(point),
                        'cdf_value': min_distance
                    })
                    valid_pairs_this_config += 1
                else:
                    excluded_pairs += 1
            
            # Print progress
            if (i + 1) % 100 == 0 or (i + 1) == n_train_configs:
                elapsed_time = time.time() - start_time
                progress = (i + 1) / n_train_configs * 100
                print(f"\nProgress: {progress:.1f}%")
                print(f"Configuration {i + 1}/{n_train_configs}")
                print(f"Valid pairs in this config: {valid_pairs_this_config}")
                print(f"Total valid pairs so far: {len(training_pairs)}")
                print(f"Total excluded pairs: {excluded_pairs}")
                print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
        
        print(f"\n=== Training Data Generation Complete ===")
        print(f"Total valid training pairs: {len(training_pairs)}")
        print(f"Total excluded pairs: {excluded_pairs}")
        print(f"Percentage kept: {len(training_pairs)/(len(training_pairs) + excluded_pairs)*100:.1f}%")
        
        return training_pairs

    def save_contact_database(self, filtered_db, save_path):
        """Save contact database in memory-efficient format"""
        compressed_db = {
            'points': np.array(list(filtered_db.keys())),  # [N, 3]
            'contact_configs': [
                np.array(configs)  # List of [M_i, 6] arrays
                for configs in filtered_db.values()
            ]
        }
        np.save(save_path, compressed_db)
        print(f"\nSaved contact database to: {save_path}")
        print(f"Number of valid points: {len(compressed_db['points'])}")
        print(f"Average configs per point: {np.mean([len(configs) for configs in compressed_db['contact_configs']]):.1f}")

if __name__ == "__main__":
    generator = XArmCDFDataGeneratorNew()
    
    # 1. Generate contact configuration database
    contact_db = generator.generate_contact_database(n_samples=100000)

    # 2. Save the contact database in compressed format
    save_path = os.path.join(generator.save_dir, 'contact_db.npy')
    generator.save_contact_database(contact_db, save_path)
    
    # Note: We're not generating training data anymore since we'll do that online
    print("\nContact database generation and saving complete!")

    
    # 2. Generate training data with CDF threshold
    # training_data = generator.generate_training_data(
    #     contact_db, 
    #     n_train_configs=2000,
    #     cdf_threshold=0.8  # Exclude pairs with CDF > 0.8
    # )
    
    # # 3. Save the data
    # if training_data:  # Only save if we have valid pairs
    #     save_path = os.path.join(generator.save_dir, 'cdf_training_data_new_large.npy')
    #     np.save(save_path, training_data)
    #     print(f"\nData saved to: {save_path}")
    # else:
    #     print("\nNo valid training pairs generated!") 