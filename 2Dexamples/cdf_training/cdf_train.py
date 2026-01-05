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

from cdf_computation import CDFDataProcessor
from network import CDFNetwork, CDFNetworkWithDropout
from losses import compute_total_loss_with_gradients

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_cdf_network(
    contact_db_path,
    model_save_dir,
    num_epochs=500,
    batch_x=10,
    batch_q=10,
    learning_rate=0.001,
    device='cuda',
    loss_threshold=0.001,
    activation='relu',
    pretrained_model=None,
    dropout_rate=0.1,
    use_dropout=False
):
    # Convert paths to absolute paths
    contact_db_path = Path(contact_db_path)
    if not contact_db_path.is_absolute():
        contact_db_path = PROJECT_ROOT / contact_db_path
    
    model_save_dir = Path(model_save_dir)
    if not model_save_dir.is_absolute():
        model_save_dir = PROJECT_ROOT / model_save_dir
    
    # Create save directory
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading contact database from: {contact_db_path}")
    print(f"Saving models to: {model_save_dir}")
    print(f"Using activation function: {activation}")
    
    # Initialize data processor
    processor = CDFDataProcessor(contact_db_path, device=device, batch_x=batch_x, batch_q=batch_q)
    
    # Initialize model and load pretrained weights if specified
    if use_dropout:
        model = CDFNetworkWithDropout(
            input_dims=8, 
            output_dims=1, 
            activation=activation,
            dropout_rate=dropout_rate
        ).to(device)
        model_prefix = f'dropout_{dropout_rate}_'
    else:
        model = CDFNetwork(
            input_dims=8, 
            output_dims=1, 
            activation=activation
        ).to(device)
        model_prefix = ''
    
    if pretrained_model:
        print(f"Loading pretrained model from: {pretrained_model}")
        model.load_state_dict(torch.load(pretrained_model))
        model_filename = f'{model_prefix}best_model_{activation}_continued.pth'
        final_model_filename = f'{model_prefix}final_model_{activation}_continued.pth'
    else:
        model_filename = f'{model_prefix}best_model_{activation}.pth'
        final_model_filename = f'{model_prefix}final_model_{activation}.pth'
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000,
        threshold=0.01, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        
        # Sample batch with gradients
        points, configs, cdf_values, cdf_gradients = processor.sample_batch()
        
        # Prepare inputs (just concatenate raw configs and points)
        points_exp = points.unsqueeze(1).expand(-1, processor.batch_q, -1)
        configs_exp = configs.unsqueeze(0).expand(processor.batch_x, -1, -1)
        
        inputs = torch.cat([
            configs_exp.reshape(-1, 2),  # raw configurations
            points_exp.reshape(-1, 2)    # 2D points
        ], dim=1)
        
        targets = cdf_values.reshape(-1)
        target_gradients = cdf_gradients.reshape(-1, 2)
        
        # Compute loss with gradients
        loss, value_loss, gradient_loss, eikonal_loss = compute_total_loss_with_gradients(
            model=model,
            inputs=inputs,
            targets=targets,
            target_gradients=target_gradients,
            value_weight=1.0,
            gradient_weight=0.05,
            eikonal_weight=0.02
        )
        
        # Backward pass and optimization
        optimizer.zero_grad()
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
                  f"Value: {value_loss.item():.6f} "
                  f"Gradient: {gradient_loss.item():.6f} "
                  f"Eikonal: {eikonal_loss.item():.6f} "
                  f"Time: {epoch_time:.3f}s")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_dir / model_filename)
        
        # Periodic save every 10000 epochs
        if (epoch + 1) % 10000 == 0:
            periodic_filename = f'{model_prefix}model_{activation}_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_dir / periodic_filename)
            print(f"Saved periodic model at epoch {epoch+1}")
        
        if loss.item() < loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break
    
    # Save final model
    torch.save(model.state_dict(), model_save_dir / final_model_filename)
    print(f"Final model saved as: {final_model_filename}")
    
    return model, best_loss

if __name__ == "__main__":
    set_random_seed(42)

    src_dir = Path(__file__).parent
    
    # Example usage
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"

    src_dir_parent = Path(__file__).parent.parent
    model_save_dir = src_dir_parent / "trained_models" 
    
    model, final_loss = train_cdf_network(
        contact_db_path=contact_db_path,
        model_save_dir=model_save_dir,
        num_epochs=30000,
        batch_x=40,
        batch_q=20,
        learning_rate=0.002,
        device='cuda',
        loss_threshold=1e-4,
        activation='gelu',
        use_dropout=True,
        dropout_rate=0.1
    ) 