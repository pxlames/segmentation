import torch
import torch.nn.functional as F
import numpy as np

softmax_helper = lambda x: F.softmax(x, 1)

def torch_dice_fn_bce(pred, target): #pytorch tensors NCDHW # should ideally do some thresholding but this approx is fine
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection) / (m1.sum() + m2.sum())


import torch
import cc3d

def torch_betti_error_loss(pred, target):
    """
    Compute Betti-error loss between predicted and target masks.
    
    Args:
        pred (torch.Tensor): Prediction tensor, assumed to be of shape (N, C, H, W) or (N, C, D, H, W)
        target (torch.Tensor): Target tensor, assumed to be of shape (N, C, H, W) or (N, C, D, H, W)
        
    Returns:
        torch.Tensor: Betti-error loss
    """
    # Ensure that pred and target are either 4D or 5D
    assert pred.dim() in {4, 5}, "pred tensor must be 4D or 5D"
    assert target.dim() in {4, 5}, "target tensor must be 4D or 5D"
    
    # Handle the case for both 2D and 3D segmentation
    num = pred.size(0)
    
    # Check if input is 4D (2D segmentation) or 5D (3D segmentation)
    if pred.dim() == 4:
        # For 2D: NCHW format
        pred_bin = (pred > 0.5).squeeze(1).cpu().numpy()  # Remove channel dimension for connected components
        target_bin = target.squeeze(1).cpu().numpy()  # Remove channel dimension for connected components
    elif pred.dim() == 5:
        # For 3D: NCDHW format
        pred_bin = (pred > 0.5).squeeze(1).cpu().numpy()
        target_bin = target.squeeze(1).cpu().numpy()

    betti_error_sum = 0.0

    for i in range(num):
        # Calculate the number of connected components (0-dimensional Betti number) in predictions and targets
        pred_components = cc3d.connected_components(pred_bin[i], connectivity=26)
        target_components = cc3d.connected_components(target_bin[i], connectivity=26)
        
        pred_betti0 = pred_components.max()  # Prediction 0-dim Betti number
        target_betti0 = target_components.max()  # Target 0-dim Betti number
        
        # Calculate Betti-error as the absolute difference in 0-dim Betti numbers
        betti_error = abs(pred_betti0 - target_betti0)
        
        # Accumulate Betti-error for all samples in the batch
        betti_error_sum += betti_error

    # Return average Betti-error loss
    return torch.tensor(betti_error_sum / num, requires_grad=True)