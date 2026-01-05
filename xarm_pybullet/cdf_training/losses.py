import torch
import torch.nn.functional as F

def cdf_loss(pred_cdf, gt_cdf):
    """
    Basic MSE loss for CDF values
    Args:
        pred_cdf: Predicted CDF values [B, 1]
        gt_cdf: Ground truth CDF values [B]
    Returns:
        loss: MSE loss between predicted and ground truth CDFs
    """
    return F.mse_loss(pred_cdf.squeeze(), gt_cdf, reduction='mean')

def eikonal_loss(grad_outputs):
    """
    Eikonal loss to enforce gradient norm = 1 for joint angles
    Args:
        grad_outputs: Gradient of CDF with respect to joint angles [B, 6]
    Returns:
        loss: Mean squared error between gradient norm and 1
    """
    grad_norm = torch.norm(grad_outputs, dim=-1)
    eikonal_term = (grad_norm - 1) ** 2
    return eikonal_term.mean()

def compute_total_loss(model, inputs, targets, eikonal_weight=0.04):
    """
    Compute total loss including MSE and eikonal terms
    Args:
        model: Neural network model
        inputs: Input tensor [B, D]
        targets: Ground truth CDF values [B]
        eikonal_weight: Weight for eikonal loss term
    Returns:
        total_loss: Combined loss
        mse_loss: MSE loss term
        eikonal_loss: Eikonal loss term
    """
    batch_size = inputs.shape[0]
    num_links = 6  # for xArm
    
    # Enable gradient tracking for input configurations
    input_configs = inputs[:, :num_links].detach().clone()
    input_configs.requires_grad = True
    
    # Reconstruct full input tensor with gradient tracking
    full_inputs = torch.cat((
        input_configs,
        torch.sin(input_configs),
        torch.cos(input_configs),
        inputs[:, -3:]  # 3D points
    ), dim=1)
    
    # Forward pass
    outputs = model(full_inputs)
    
    # MSE loss
    mse = cdf_loss(outputs, targets)
    
    # Compute eikonal loss
    grad_outputs_inputs = torch.ones_like(outputs)
    grad_cdf_inputs = torch.autograd.grad(
        outputs=outputs,
        inputs=input_configs,
        grad_outputs=grad_outputs_inputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    eik = eikonal_loss(grad_cdf_inputs)
    
    # Combine losses
    total = mse + eikonal_weight * eik
    
    return total, mse, eik 

def compute_total_loss_with_gradients(model, inputs, targets, target_gradients, 
                                    value_weight=1.0, gradient_weight=0.1, eikonal_weight=0.04):
    """
    Compute total loss including value, gradient, and eikonal terms
    Args:
        model: Neural network model
        inputs: Input tensor [B, D]
        targets: Ground truth CDF values [B]
        target_gradients: Ground truth gradients [B, 6]
        value_weight: Weight for CDF value loss
        gradient_weight: Weight for gradient supervision loss
        eikonal_weight: Weight for eikonal loss term
    Returns:
        total_loss: Combined loss
        value_loss: MSE loss term for CDF values
        gradient_loss: MSE loss term for gradients
        eikonal_loss: Eikonal loss term
    """
    batch_size = inputs.shape[0]
    num_links = 6  # for xArm
    
    # Enable gradient tracking for input configurations
    input_configs = inputs[:, :num_links].detach().clone()
    input_configs.requires_grad = True
    
    # Reconstruct full input tensor with gradient tracking
    full_inputs = torch.cat((
        input_configs,
        torch.sin(input_configs),
        torch.cos(input_configs),
        inputs[:, -3:]  # 3D points
    ), dim=1)
    
    # Forward pass
    outputs = model(full_inputs)
    
    # Value loss (MSE)
    value_loss = cdf_loss(outputs, targets)
    
    # Compute gradients for both supervision and eikonal loss
    grad_outputs_inputs = torch.ones_like(outputs)
    grad_cdf_inputs = torch.autograd.grad(
        outputs=outputs,
        inputs=input_configs,
        grad_outputs=grad_outputs_inputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient supervision loss
    gradient_loss = F.mse_loss(grad_cdf_inputs, target_gradients)
    
    # Eikonal loss
    eik_loss = eikonal_loss(grad_cdf_inputs)
    
    # Combine losses
    total = (value_weight * value_loss + 
             gradient_weight * gradient_loss + 
             eikonal_weight * eik_loss)
    
    return total, value_loss, gradient_loss, eik_loss 