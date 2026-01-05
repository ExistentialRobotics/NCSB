import torch
import torch.nn.functional as F

def collision_cdf_loss(pred_distances, gt_configs, relevant_joints):
    """
    Loss function for self-collision CDF
    Args:
        pred_distances: Predicted distances [B, 1]
        gt_configs: Ground truth configurations [B, 6]
        relevant_joints: List of relevant joint indices for each config
    """
    # Basic MSE loss for now - can be enhanced with additional terms
    return F.mse_loss(pred_distances.squeeze(), torch.zeros_like(pred_distances.squeeze()))

def eikonal_loss(grad_outputs, relevant_joints):
    """
    Eikonal loss considering only relevant joints
    Args:
        grad_outputs: Gradient of CDF with respect to joint angles [B, 6]
        relevant_joints: List of relevant joint indices for each config
    """
    batch_size = grad_outputs.shape[0]
    grad_norms = []
    
    for i in range(batch_size):
        # Only consider gradients of relevant joints
        relevant_grads = grad_outputs[i, relevant_joints[i]]
        grad_norm = torch.norm(relevant_grads)
        grad_norms.append((grad_norm - 1) ** 2)
    
    return torch.mean(torch.stack(grad_norms))

def compute_total_loss(model, inputs, gt_configs, relevant_joints, eikonal_weight=0.1):
    """
    Compute total loss including collision CDF and eikonal terms
    """
    batch_size = inputs.shape[0]
    
    # Enable gradient tracking for input configurations
    input_configs = inputs[:, :6].detach().clone()
    input_configs.requires_grad = True
    
    # Reconstruct full input tensor
    full_inputs = torch.cat((
        input_configs,
        torch.sin(input_configs),
        torch.cos(input_configs),
        inputs[:, -6:]  # joint mask
    ), dim=1)
    
    # Forward pass
    distances = model(full_inputs)
    
    # CDF loss
    cdf_loss = collision_cdf_loss(distances, gt_configs, relevant_joints)
    
    # Compute gradients for eikonal loss
    grad_outputs = torch.ones_like(distances)
    grad_cdf = torch.autograd.grad(
        outputs=distances,
        inputs=input_configs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Eikonal loss
    eik_loss = eikonal_loss(grad_cdf, relevant_joints)
    
    # Total loss
    total_loss = cdf_loss + eikonal_weight * eik_loss
    
    return total_loss, cdf_loss, eik_loss 