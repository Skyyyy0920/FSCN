import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    
    Args:
        alpha: Class weights, can be float or list
               - float: Use same weight for all classes
               - list: Specify weight for each class
        gamma: Focusing parameter, increases weight on hard-to-classify samples (default 2.0)
        ignore_index: Target value to ignore (e.g., -1 for unlabeled)
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] Model output logits
            targets: [batch_size] Target classes
        """
        # Create mask for valid samples (excluding ignore_index)
        valid_mask = targets != self.ignore_index
        
        if valid_mask.sum() == 0:
            # If no valid samples, return 0
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha weights
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha is a list, specify weight for each class
                alpha = torch.tensor(self.alpha, device=inputs.device)
                # Only apply alpha to valid targets
                alpha_t = torch.ones_like(focal_loss)
                alpha_t[valid_mask] = alpha[targets[valid_mask]]
            focal_loss = alpha_t * focal_loss
        
        # Only compute loss for valid samples
        focal_loss = focal_loss[valid_mask]
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with automatic weight calculation based on class distribution
    
    Args:
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] Model output logits
            targets: [batch_size] Target classes
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate pt
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='ce', alpha=None, gamma=2.0, class_counts=None, ignore_index=-100):
    """
    Get loss function
    
    Args:
        loss_type: Loss function type
                   - 'ce': CrossEntropyLoss
                   - 'focal': FocalLoss
                   - 'weighted_ce': Weighted CrossEntropyLoss
                   - 'weighted_focal': Automatically weighted FocalLoss
        alpha: Alpha parameter for Focal Loss
        gamma: Gamma parameter for Focal Loss
        class_counts: Class sample count dictionary for computing weights {class_id: count}
        ignore_index: Target value to ignore (e.g., -1 for unlabeled)
    
    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    elif loss_type == 'weighted_ce':
        if class_counts is None:
            print("Warning: Using weighted_ce but class_counts not provided, using standard CrossEntropyLoss")
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Calculate class weights (inversely proportional to class frequency)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        weights = torch.tensor([
            total / (num_classes * class_counts[i]) 
            for i in range(num_classes)
        ], dtype=torch.float32)
        
        print(f"  Class weights: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    
    elif loss_type == 'weighted_focal':
        if class_counts is None:
            print("Warning: Using weighted_focal but class_counts not provided, using standard FocalLoss")
            return FocalLoss(gamma=gamma, ignore_index=ignore_index)
        
        # Calculate class weights
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        alpha = [
            total / (num_classes * class_counts[i]) 
            for i in range(num_classes)
        ]
        
        print(f"  Focal Loss alpha weights: {alpha}")
        print(f"  Focal Loss gamma: {gamma}")
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")


def print_loss_info(loss_type, gamma=2.0):
    """Print loss function information"""
    print(f"\nLoss function: {loss_type}")
    
    if loss_type == 'ce':
        print("  Type: Standard Cross Entropy Loss (CrossEntropyLoss)")
        print("  Features: Suitable for balanced datasets")
    
    elif loss_type == 'focal':
        print("  Type: Focal Loss")
        print(f"  gamma: {gamma}")
        print("  Features: Reduces weight on easy-to-classify samples, focuses on hard samples")
        print("  Use case: Class imbalance scenarios")
    
    elif loss_type == 'weighted_ce':
        print("  Type: Weighted Cross Entropy Loss (Weighted CrossEntropyLoss)")
        print("  Features: Automatically computes weights based on class frequency")
        print("  Use case: Class imbalance scenarios")
    
    elif loss_type == 'weighted_focal':
        print("  Type: Weighted Focal Loss")
        print(f"  gamma: {gamma}")
        print("  Features: Combines class weights and hard example mining")
        print("  Use case: Severe class imbalance scenarios")

