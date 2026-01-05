import torch
import torch.nn.functional as F

def sdf_loss(pred_sdf, gt_sdf):
    """
    Basic MSE loss for SDF values
    Args:
        pred_sdf: Predicted SDF values [B, 1]
        gt_sdf: Ground truth SDF values [B]
    Returns:
        loss: MSE loss between predicted and ground truth SDFs
    """
    return F.mse_loss(pred_sdf.squeeze(), gt_sdf, reduction='mean')

def eikonal_loss(grad_outputs):
    """
    Eikonal loss to enforce gradient norm = 1
    Args:
        grad_outputs: Gradient of SDF with respect to input points [B, 3]
    Returns:
        loss: Mean squared error between gradient norm and 1
    """
    grad_norm = torch.norm(grad_outputs, dim=-1)
    eikonal_term = (grad_norm - 1) ** 2
    return eikonal_term.mean()
