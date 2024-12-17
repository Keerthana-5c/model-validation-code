
import torch

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def calculate_accuracy(pred, target):
    """
    Calculate the accuracy of predictions compared to the target labels.
    Accuracy is defined as the proportion of true results (both true positives and true negatives)
    among the total number of cases examined.

    Args:
    pred (torch.Tensor): Predicted outputs from the model, expected shape (N, *)
    target (torch.Tensor): Ground truth labels, expected shape (N, *)

    Returns:
    float: Accuracy as a percentage.
    """
    # Ensure predictions are binary (0 or 1)
    _, pred = torch.max(pred, 1)
    # Calculate correct predictions
    correct = (pred == target).float().sum()
    # Calculate accuracy
    accuracy = correct / pred.numel()
    
    return accuracy
