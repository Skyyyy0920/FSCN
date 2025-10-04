"""
损失函数定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    论文: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    
    参数:
        alpha: 类别权重，可以是float或list
               - float: 对所有类别使用相同权重
               - list: 为每个类别指定权重
        gamma: 聚焦参数，增加难分类样本的权重 (默认2.0)
        ignore_index: 忽略的目标值（例如-1表示无标签）
        reduction: 'mean', 'sum' 或 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        参数:
            inputs: [batch_size, num_classes] 模型输出logits
            targets: [batch_size] 目标类别
        """
        # 创建有效样本的mask（排除ignore_index）
        valid_mask = targets != self.ignore_index
        
        if valid_mask.sum() == 0:
            # 如果没有有效样本，返回0
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # 计算pt (预测正确类别的概率)
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用alpha权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha是一个列表，为每个类别指定权重
                alpha = torch.tensor(self.alpha, device=inputs.device)
                # 只对有效目标应用alpha
                alpha_t = torch.ones_like(focal_loss)
                alpha_t[valid_mask] = alpha[targets[valid_mask]]
            focal_loss = alpha_t * focal_loss
        
        # 只对有效样本计算loss
        focal_loss = focal_loss[valid_mask]
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    自动根据类别分布计算权重的Focal Loss
    
    参数:
        gamma: 聚焦参数 (默认2.0)
        reduction: 'mean', 'sum' 或 'none'
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        参数:
            inputs: [batch_size, num_classes] 模型输出logits
            targets: [batch_size] 目标类别
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='ce', alpha=None, gamma=2.0, class_counts=None, ignore_index=-100):
    """
    获取损失函数
    
    参数:
        loss_type: 损失函数类型
                   - 'ce': CrossEntropyLoss
                   - 'focal': FocalLoss
                   - 'weighted_ce': 加权CrossEntropyLoss
                   - 'weighted_focal': 自动加权FocalLoss
        alpha: Focal Loss的alpha参数
        gamma: Focal Loss的gamma参数
        class_counts: 类别样本数量字典，用于计算权重 {class_id: count}
        ignore_index: 忽略的目标值（例如-1表示无标签）
    
    返回:
        loss_fn: 损失函数
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    elif loss_type == 'weighted_ce':
        if class_counts is None:
            print("警告: 使用weighted_ce但未提供class_counts，使用标准CrossEntropyLoss")
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # 计算类别权重 (inversely proportional to class frequency)
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        weights = torch.tensor([
            total / (num_classes * class_counts[i]) 
            for i in range(num_classes)
        ], dtype=torch.float32)
        
        print(f"  类别权重: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    
    elif loss_type == 'weighted_focal':
        if class_counts is None:
            print("警告: 使用weighted_focal但未提供class_counts，使用标准FocalLoss")
            return FocalLoss(gamma=gamma, ignore_index=ignore_index)
        
        # 计算类别权重
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        alpha = [
            total / (num_classes * class_counts[i]) 
            for i in range(num_classes)
        ]
        
        print(f"  Focal Loss alpha权重: {alpha}")
        print(f"  Focal Loss gamma: {gamma}")
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


def print_loss_info(loss_type, gamma=2.0):
    """打印损失函数信息"""
    print(f"\n损失函数: {loss_type}")
    
    if loss_type == 'ce':
        print("  类型: 标准交叉熵损失 (CrossEntropyLoss)")
        print("  特点: 适用于平衡数据集")
    
    elif loss_type == 'focal':
        print("  类型: Focal Loss")
        print(f"  gamma: {gamma}")
        print("  特点: 降低易分类样本的权重，关注难分类样本")
        print("  适用: 类别不平衡场景")
    
    elif loss_type == 'weighted_ce':
        print("  类型: 加权交叉熵损失 (Weighted CrossEntropyLoss)")
        print("  特点: 根据类别频率自动计算权重")
        print("  适用: 类别不平衡场景")
    
    elif loss_type == 'weighted_focal':
        print("  类型: 加权Focal Loss")
        print(f"  gamma: {gamma}")
        print("  特点: 结合类别权重和难例挖掘")
        print("  适用: 严重类别不平衡场景")

