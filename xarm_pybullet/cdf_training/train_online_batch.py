import torch
import torch.optim as optim
import numpy as np
import sys
import random
from pathlib import Path
from network import CDFNetwork
from losses import compute_total_loss
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.xarm_model import XArmFK

src_dir = PROJECT_ROOT 

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_cdf_values(points, configs, contact_configs, link_indices, device='cuda'):
    """
    Compute ground truth CDF values for given points and configurations.
    
    Args:
        points (torch.Tensor): (N, 3) tensor of points
        configs (torch.Tensor): (M, 6) tensor of robot configurations
        contact_configs (list): List of contact configurations for each point
        link_indices (list): List of link indices for each point
        device (str): Device to use for computation
    
    Returns:
        torch.Tensor: (N, M) tensor of CDF values
    """
    batch_x = len(points)
    batch_q = len(configs)
    cdf_values = torch.zeros(batch_x, batch_q, device=device)
    
    for i in range(batch_x):
        contact_configs_i = torch.tensor(contact_configs[i], device=device)
        point_link_indices = torch.tensor(link_indices[i], device=device)
        
        # For each sampled config, compute minimum distance to contact configs
        distances = torch.zeros(batch_q, len(contact_configs_i), device=device)
        for config_idx, link_idx in enumerate(point_link_indices):
            # Only use joints up to the contact link
            relevant_joints = slice(0, link_idx + 1)
            dist = torch.norm(
                configs[:, relevant_joints] - contact_configs_i[config_idx, relevant_joints],
                dim=1
            )
            distances[:, config_idx] = dist
        
        # CDF value is the minimum distance
        cdf_values[i] = distances.min(dim=1)[0]
    
    return cdf_values

class CDFTrainer:
    def __init__(self, contact_db_path, device='cuda'):
        # Load compressed contact database
        self.db = np.load(contact_db_path, allow_pickle=True).item()
        self.valid_points = torch.tensor(self.db['points'], device=device)
        self.contact_configs = self.db['contact_configs']
        self.link_indices = self.db['link_indices']
        
        # Add diagnostic prints
        print("\nContact Database Statistics:")
        print(f"Total number of points: {len(self.valid_points)}")
        print(f"Number of points with no configurations: {sum(len(configs) == 0 for configs in self.contact_configs)}")
        print(f"Configuration counts per point: min={min(len(configs) for configs in self.contact_configs)}, "
              f"max={max(len(configs) for configs in self.contact_configs)}, "
              f"mean={np.mean([len(configs) for configs in self.contact_configs]):.1f}")
        
        # Link indices statistics
        # all_link_indices = np.concatenate(self.link_indices)
        # unique_links, link_counts = np.unique(all_link_indices, return_counts=True)
        # print("\nLink Index Distribution:")
        # for link, count in zip(unique_links, link_counts):
        #     print(f"Link {link}: {count} contacts ({count/len(all_link_indices)*100:.1f}%)")
        
        
        # Initialize robot model for joint limits
        self.robot_fk = XArmFK(device=device)
        self.q_max = self.robot_fk.joint_limits[:, 1]
        self.q_min = self.robot_fk.joint_limits[:, 0]
        
        self.device = device
        self.batch_x = 50
        self.batch_q = 10
        self.max_q_per_link = 100

    def sample_q(self, batch_q=None):
        if batch_q is None:
            batch_q = self.batch_q
        # Use XArm joint limits
        q_sampled = self.robot_fk.joint_limits[:, 0] + torch.rand(batch_q, 6).to(self.device) * (
            self.robot_fk.joint_limits[:, 1] - self.robot_fk.joint_limits[:, 0]
        )
        q_sampled.requires_grad = True
        return q_sampled

    def sample_batch(self):
        """Sample batch of points and configs, compute CDF values online"""
        # 1. Sample random points from valid points
        point_indices = torch.randint(0, len(self.valid_points), (self.batch_x,))
        points = self.valid_points[point_indices]
        
        # 2. Sample random configurations
        configs = self.sample_q()
        
        # 3. Compute CDF values using helper function
        contact_configs_batch = [self.contact_configs[idx] for idx in point_indices]
        link_indices_batch = [self.link_indices[idx] for idx in point_indices]
        
        cdf_values = compute_cdf_values(
            points=points,
            configs=configs,
            contact_configs=contact_configs_batch,
            link_indices=link_indices_batch,
            device=self.device
        )
        
        return points, configs, cdf_values

def train_cdf_network(
    contact_db_path,
    model_save_dir,
    num_epochs=50000,
    learning_rate=0.001,
    device='cuda',
    loss_threshold=0.001,
    activation='relu',
    pretrained_model=None
):
    # Convert paths to absolute paths
    contact_db_path = Path(contact_db_path)
    if not contact_db_path.is_absolute():
        contact_db_path = src_dir / contact_db_path
    
    model_save_dir = Path(model_save_dir)
    if not model_save_dir.is_absolute():
        model_save_dir = src_dir / model_save_dir
    
    # Create save directory
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading contact database from: {contact_db_path}")
    print(f"Saving models to: {model_save_dir}")
    print(f"Using activation function: {activation}")
    
    # Initialize trainer
    trainer = CDFTrainer(contact_db_path, device)
    
    # Initialize model and load pretrained weights if specified
    model = CDFNetwork(activation=activation).to(device)
    if pretrained_model:
        print(f"Loading pretrained model from: {pretrained_model}")
        model.load_state_dict(torch.load(pretrained_model))
        # Create new filenames for continued training
        model_filename = f'best_model_bfgs_{activation}_4.pth'
        final_model_filename = f'final_model_bfgs_{activation}_4.pth'
    else:
        model_filename = f'best_model_bfgs_{activation}.pth'
        final_model_filename = f'final_model_bfgs_{activation}_final.pth'
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2000,
        threshold=0.01, verbose=True
    )
    
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        
        # Sample batch and compute CDF values
        points, configs, cdf_values = trainer.sample_batch()
        
        # Prepare inputs (all combinations of points and configs)
        points_exp = points.unsqueeze(1).expand(-1, trainer.batch_q, -1)
        configs_exp = configs.unsqueeze(0).expand(trainer.batch_x, -1, -1)
        
        inputs = torch.cat([
            configs_exp.reshape(-1, 6),  # Configs first
            points_exp.reshape(-1, 3)    # Points last
        ], dim=1)
        
        targets = cdf_values.reshape(-1)
        
        # Forward pass and loss computation using helper
        optimizer.zero_grad()
        loss, mse_loss, eikonal_loss = compute_total_loss(
            model=model,
            inputs=inputs,
            targets=targets,
            eikonal_weight=0.04
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Update scheduler
        scheduler.step(loss)
        
        # Print progress
        if epoch % 10 == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {loss.item():.6f} "
                  f"MSE: {mse_loss.item():.6f} "
                  f"Eikonal: {eikonal_loss.item():.6f} "
                  f"Time: {epoch_time:.3f}s")
        
        # Save both best model and final model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_dir / model_filename)

        if loss.item() < loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break
    
    # Save final model state regardless of performance
    torch.save(model.state_dict(), model_save_dir / final_model_filename)
    print(f"Final model saved as: {final_model_filename}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_loss

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Use relative paths from project root
    contact_db_path = src_dir / "data/cdf_data/refined_bfgs_100_contact_db.npy"
    model_save_dir = src_dir / "trained_models/cdf"


    pretrained_model = src_dir / "trained_models/cdf/best_model_bfgs_gelu_3.pth"
    
    model, final_loss = train_cdf_network(
        contact_db_path=contact_db_path,
        model_save_dir=model_save_dir,
        num_epochs=30000,
        learning_rate=0.001,  
        device='cuda',
        loss_threshold=1e-4,
        activation='gelu',
        pretrained_model=pretrained_model
    ) 