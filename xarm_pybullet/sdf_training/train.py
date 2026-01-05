import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import random

def set_seed(seed):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from network import SDFNetwork
from sdf_training.losses import sdf_loss, eikonal_loss
from torch.nn import  ReLU

class SDFTrainer:
    def __init__(self, 
                 link_id,
                 data_path,
                 model_save_path,
                 device='cuda',
                 hidden_size=128,
                 num_layers=5,
                 lr=1e-4,
                 seed=42,
                 activation=ReLU):
        """
        Trainer for single link SDF
        Args:
            link_id: ID of the link to train
            data_path: Path to SDF data
            model_save_path: Where to save models
            device: Device to train on
            hidden_size: Hidden layer size
            num_layers: Number of layers
            lr: Learning rate
            seed: Random seed for reproducibility
            activation: Activation function to use (default: ReLU)
        """
        set_seed(seed)
        
        self.device = device
        self.link_id = link_id
        
        # Load data
        data = np.load(data_path)
        self.near_points = torch.from_numpy(data['near_points']).float()
        self.near_sdf = torch.from_numpy(data['near_sdf']).float()
        self.random_points = torch.from_numpy(data['random_points']).float()
        self.random_sdf = torch.from_numpy(data['random_sdf']).float()
        
        # Create model with specified activation
        self.model = SDFNetwork(
            input_size=3,
            output_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            act_fn=activation
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20,
            threshold=0.01, verbose=True
        )
        
        # Create save directory
        self.save_path = Path(model_save_path) / f"link_{link_id}"
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def train_step(self, points, sdfs, lambda_eikonal=0.1):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with gradient computation for eikonal loss
        points.requires_grad_(True)
        pred_sdf = self.model(points)
        
        # SDF loss
        sdf_loss_val = sdf_loss(pred_sdf, sdfs)
        
        # Eikonal loss
        grad_outputs = torch.ones_like(pred_sdf, requires_grad=False, device=self.device)
        gradients = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        eik_loss_val = eikonal_loss(gradients)
        
        # Combined loss
        total_loss = sdf_loss_val + lambda_eikonal * eik_loss_val
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), sdf_loss_val.item(), eik_loss_val.item()
    
    def train(self, num_epochs=2000, batch_size=256, lambda_eikonal=0.1, threshold=1e-6):
        """Full training loop with early stopping"""
        samples_per_epoch = len(self.near_points) + len(self.random_points)
        batches_per_epoch = samples_per_epoch // batch_size
        
        print(f"Training config:")
        print(f"- Total points: {samples_per_epoch}")
        print(f"- Batch size: {batch_size}")
        print(f"- Batches per epoch: {batches_per_epoch}")
        print(f"- Eikonal loss weight: {lambda_eikonal}")
        print(f"- Early stopping threshold: {threshold}")
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_sdf_losses = []
            epoch_eikonal_losses = []
            
            # Shuffle data
            near_indices = torch.randperm(len(self.near_points))
            random_indices = torch.randperm(len(self.random_points))
            
            near_batch_size = int(0.5 * batch_size)
            random_batch_size = batch_size - near_batch_size
            
            for batch_idx in range(batches_per_epoch):
                # Sample points (same as before)
                near_start = (batch_idx * near_batch_size) % len(self.near_points)
                random_start = (batch_idx * random_batch_size) % len(self.random_points)
                
                near_idx = near_indices[near_start:near_start + near_batch_size]
                random_idx = random_indices[random_start:random_start + random_batch_size]
                
                # Handle wraparound
                if len(near_idx) < near_batch_size:
                    extra_needed = near_batch_size - len(near_idx)
                    near_idx = torch.cat([near_idx, near_indices[:extra_needed]])
                
                if len(random_idx) < random_batch_size:
                    extra_needed = random_batch_size - len(random_idx)
                    random_idx = torch.cat([random_idx, random_indices[:extra_needed]])
                
                points = torch.cat([
                    self.near_points[near_idx],
                    self.random_points[random_idx]
                ]).to(self.device)
                
                sdfs = torch.cat([
                    self.near_sdf[near_idx],
                    self.random_sdf[random_idx]
                ]).to(self.device)
                
                total_loss, sdf_loss_val, eik_loss_val = self.train_step(
                    points, sdfs, lambda_eikonal)
                
                epoch_losses.append(total_loss)
                epoch_sdf_losses.append(sdf_loss_val)
                epoch_eikonal_losses.append(eik_loss_val)
            
            # Compute average losses
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_sdf_loss = sum(epoch_sdf_losses) / len(epoch_sdf_losses)
            avg_eikonal_loss = sum(epoch_eikonal_losses) / len(epoch_eikonal_losses)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_model("best_model.pt")
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {avg_loss:.6f}, "
                      f"SDF Loss: {avg_sdf_loss:.6f}, "
                      f"Eikonal Loss: {avg_eikonal_loss:.6f}")
            
            # Check stopping conditions
            if avg_loss < threshold:
                print(f"Reached loss threshold {threshold}. Stopping training.")
                break
            
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break
            
            self.scheduler.step(avg_loss)
    
    def save_model(self, filename):
        """Save model and metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, self.save_path / filename)

if __name__ == "__main__":
    # Set global seed
    SEED = 42
    set_seed(SEED)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "sdf_data"
    model_dir = project_root / "trained_models"
    
    # Train each link
    # for link_id in range(7):  # 0 to 6 for base + 6 links
    #     print(f"\nTraining Link {link_id}")
    #     data_path = data_dir / f"link_{link_id}_link{link_id}.npz"
        
    #     trainer = SDFTrainer(
    #         link_id=link_id,
    #         data_path=data_path,
    #         model_save_path=model_dir,
    #         device='cuda',
    #         seed=SEED
    #     )
    #     trainer.train(num_epochs=300)

    # Train only the gripper (link 7)
    link_id = 7  # gripper
    print(f"\nTraining Gripper (Link {link_id})")
    data_path = data_dir / f"link_{link_id}_gripper.npz"
    
    trainer = SDFTrainer(
        link_id=link_id,
        data_path=data_path,
        model_save_path=model_dir,
        device='cuda',
        seed=SEED
    )
    trainer.train(num_epochs=300)