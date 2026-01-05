import torch
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
import random
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.xarm_model import XArmFK
from network import SelfCollisionCDFNetwork

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SelfCollisionCDFTrainer:
    def __init__(self, data_path, device='cuda'):
        self.device = device
        self.robot_fk = XArmFK(device=device)
        
        # Load prepared dataset
        data = np.load(data_path, allow_pickle=True).item()
        self.colliding_configs = data['colliding_configs']
        self.relevant_joints = data['relevant_joints']
        
        # Group templates by relevant joints
        self.joint_groups = {}
        for config, joints in zip(self.colliding_configs, self.relevant_joints):
            key = tuple(sorted(joints))  # Convert to tuple for dict key
            if key not in self.joint_groups:
                self.joint_groups[key] = []
            self.joint_groups[key].append(config)
        
        # Convert configs to tensors for each group
        for key in self.joint_groups:
            self.joint_groups[key] = torch.tensor(
                self.joint_groups[key], 
                dtype=torch.float32,  # Explicitly set dtype
                device=device
            )
        
        print(f"\nLoaded collision templates:")
        for joints, templates in self.joint_groups.items():
            print(f"Joint group {joints}: {len(templates)} templates")
        
        # Training parameters
        self.batch_size = 1000
        
    def sample_configs(self):
        """Sample random configurations within joint limits"""
        configs = (self.robot_fk.joint_limits[:, 0] + 
                  torch.rand(self.batch_size, 6, dtype=torch.float32, device=self.device) * 
                  (self.robot_fk.joint_limits[:, 1] - self.robot_fk.joint_limits[:, 0]))
        configs.requires_grad = True
        return configs
    
    def compute_cdf_values(self, configs):
        """Compute ground truth CDF values for configurations"""
        batch_size = len(configs)
        min_distances = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Check each joint group
        for joints, templates in self.joint_groups.items():
            # Compute distances only for relevant joints
            joint_indices = list(joints)
            
            # Extract relevant joint values [batch_size, n_relevant_joints]
            q_relevant = configs[:, joint_indices]
            
            # Extract template values [n_templates, n_relevant_joints]
            templates_relevant = templates[:, joint_indices]
            
            # Compute differences and distances
            diff = q_relevant.unsqueeze(1) - templates_relevant.unsqueeze(0)
            distances = torch.norm(diff, dim=2)  # [batch_size, n_templates]
            group_min_distances = torch.min(distances, dim=1)[0]  # [batch_size]
            
            # Update minimum distances
            min_distances = torch.minimum(min_distances, group_min_distances)
        
        return min_distances

def train_self_collision_cdf(
    data_path,
    model_save_dir,
    num_epochs=50000,
    learning_rate=0.001,
    device='cuda',
    loss_threshold=0.001,
    activation='relu'
):
    # Setup paths and trainer
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = SelfCollisionCDFTrainer(data_path, device=device)
    
    # Initialize model
    model = SelfCollisionCDFNetwork(activation=activation).to(device)
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
        
        # Sample configurations and compute CDF values
        configs = trainer.sample_configs()
        cdf_values = trainer.compute_cdf_values(configs)
        
        # Prepare inputs (add sin/cos encoding)
        inputs = torch.cat([
            configs,
            torch.sin(configs),
            torch.cos(configs)
        ], dim=1)
        
        # Forward pass
        optimizer.zero_grad()
        pred_distances = model(inputs)
        
        # Compute losses
        mse_loss = torch.nn.functional.mse_loss(pred_distances.squeeze(), cdf_values)
        
        # Compute eikonal loss
        grad_outputs = torch.ones_like(pred_distances)
        gradients = torch.autograd.grad(
            outputs=pred_distances,
            inputs=configs,
            grad_outputs=grad_outputs,
            create_graph=True
        )[0]
        eikonal_loss = torch.mean((torch.norm(gradients, dim=1) - 1) ** 2)
        
        # Total loss
        loss = mse_loss + 0.1 * eikonal_loss
        
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
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_dir / 'best_model.pth')
        
        if loss.item() < loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break
    
    # Save final model
    torch.save(model.state_dict(), model_save_dir / 'final_model.pth')
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_loss

if __name__ == "__main__":
    set_random_seed(42)
    
    data_path = PROJECT_ROOT / "data/self_cdf_data/refined_self_collision_data.npy"
    model_save_dir = PROJECT_ROOT / "trained_models/self_collision_cdf"
    
    model, final_loss = train_self_collision_cdf(
        data_path=data_path,
        model_save_dir=model_save_dir,
        num_epochs=30000,
        learning_rate=0.001,
        device='cuda',
        loss_threshold=1e-4,
        activation='gelu'
    ) 