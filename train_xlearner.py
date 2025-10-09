"""
X-Learner 模型的训练脚本

训练策略：
Stage 1: 预训练 FC-only 和 SC-only 基础预测器
Stage 2: 训练效应学习器（冻结 Stage 1）
Stage 3: 训练倾向性网络并端到端微调（可选冻结 Stage 1 & 2）
"""

import os
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
from collections import Counter

from model_xlearner import XLearnerModel
from data_utils import (
    load_raw_data,
    load_abide_data,
    load_abide_split_data,
    prepare_task_data,
    prepare_task_data_from_dict,
    get_num_classes,
    print_task_summary,
    BrainDataset,
    BalancedBatchSampler
)
from losses import get_loss_function, print_loss_info


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sensitivity_specificity(y_true, y_pred):
    """计算敏感度和特异度"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < 2:
        return 0.0, 0.0
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity


def train_stage1(model, dataloader, criterion, optimizer, device):
    """
    Stage 1: 训练基础预测器
    同时训练 FC-only 和 SC-only 预测器
    """
    model.train()
    model.set_training_stage(1)
    
    total_fc_loss = 0
    total_sc_loss = 0
    all_fc_preds = []
    all_sc_preds = []
    all_labels = []
    all_fc_probs = []
    all_sc_probs = []
    
    for fc, sc, labels in tqdm(dataloader, desc="Stage 1 Training", leave=False):
        fc = fc.to(device)
        sc = sc.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(fc, sc)
        fc_logits = outputs['fc_logits']
        sc_logits = outputs['sc_logits']
        
        # 计算损失
        fc_loss = criterion(fc_logits, labels)
        sc_loss = criterion(sc_logits, labels)
        
        # 组合损失
        loss = fc_loss + sc_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 记录
        total_fc_loss += fc_loss.item()
        total_sc_loss += sc_loss.item()
        
        # 预测
        fc_probs = torch.softmax(fc_logits, dim=1)
        sc_probs = torch.softmax(sc_logits, dim=1)
        fc_preds = torch.argmax(fc_probs, dim=1)
        sc_preds = torch.argmax(sc_probs, dim=1)
        
        # 转换为 numpy
        labels_np = labels.cpu().numpy()
        valid_mask = labels_np != -1
        
        all_fc_preds.extend(fc_preds.cpu().numpy()[valid_mask])
        all_sc_preds.extend(sc_preds.cpu().numpy()[valid_mask])
        all_labels.extend(labels_np[valid_mask])
        all_fc_probs.extend(fc_probs[:, 1].detach().cpu().numpy()[valid_mask])
        all_sc_probs.extend(sc_probs[:, 1].detach().cpu().numpy()[valid_mask])
    
    avg_fc_loss = total_fc_loss / len(dataloader)
    avg_sc_loss = total_sc_loss / len(dataloader)
    
    # 计算指标
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        fc_acc = accuracy_score(all_labels, all_fc_preds)
        sc_acc = accuracy_score(all_labels, all_sc_preds)
        fc_recall = recall_score(all_labels, all_fc_preds, average='binary', zero_division=0)
        sc_recall = recall_score(all_labels, all_sc_preds, average='binary', zero_division=0)
        fc_sens, fc_spec = compute_sensitivity_specificity(all_labels, all_fc_preds)
        sc_sens, sc_spec = compute_sensitivity_specificity(all_labels, all_sc_preds)
        fc_auroc = roc_auc_score(all_labels, all_fc_probs)
        sc_auroc = roc_auc_score(all_labels, all_sc_probs)
    else:
        fc_acc = sc_acc = fc_recall = sc_recall = 0.0
        fc_sens = fc_spec = sc_sens = sc_spec = 0.0
        fc_auroc = sc_auroc = 0.0
    
    return (avg_fc_loss, avg_sc_loss, fc_acc, sc_acc, fc_recall, sc_recall, 
            fc_sens, fc_spec, sc_sens, sc_spec, fc_auroc, sc_auroc)


def train_stage2(model, dataloader, criterion, optimizer, device):
    """
    Stage 2: 训练效应学习器
    学习模态间的差异和互补性
    """
    model.train()
    model.set_training_stage(2)
    
    total_loss = 0
    
    for fc, sc, labels in tqdm(dataloader, desc="Stage 2 Training", leave=False):
        fc = fc.to(device)
        sc = sc.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(fc, sc)
        fc_logits = outputs['fc_logits']
        sc_logits = outputs['sc_logits']
        effect_fc = outputs['effect_fc']
        effect_sc = outputs['effect_sc']
        
        # 效应学习的目标：
        # effect_fc 应该预测 (真实标签 - FC预测)
        # effect_sc 应该预测 (真实标签 - SC预测)
        
        # 创建有效样本 mask
        valid_mask = labels != -1
        
        # 将标签转换为 one-hot 用于效应计算
        # 先将 -1 替换为 0（临时值），避免 one_hot 索引错误
        labels_safe = labels.clone()
        labels_safe[labels == -1] = 0
        labels_onehot = F.one_hot(labels_safe, num_classes=2).float()
        # 将无效样本的 one-hot 设为 0
        labels_onehot[~valid_mask] = 0
        
        fc_probs = F.softmax(fc_logits, dim=1)
        sc_probs = F.softmax(sc_logits, dim=1)
        
        # 计算目标效应（真实 - 预测）
        target_effect_fc = labels_onehot - fc_probs.detach()  # detach 防止影响基础预测器
        target_effect_sc = labels_onehot - sc_probs.detach()
        
        # 只对有效样本计算效应学习损失
        if valid_mask.sum() > 0:
            loss_effect_fc = F.mse_loss(effect_fc[valid_mask], target_effect_fc[valid_mask])
            loss_effect_sc = F.mse_loss(effect_sc[valid_mask], target_effect_sc[valid_mask])
        else:
            loss_effect_fc = torch.tensor(0.0, device=device, requires_grad=True)
            loss_effect_sc = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = loss_effect_fc + loss_effect_sc
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_stage3(model, dataloader, criterion, optimizer, device):
    """
    Stage 3: 训练倾向性网络并端到端微调
    学习如何智能融合两个模态的预测
    """
    model.train()
    model.set_training_stage(3)
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_weights = []
    
    for fc, sc, labels in tqdm(dataloader, desc="Stage 3 Training", leave=False):
        fc = fc.to(device)
        sc = sc.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(fc, sc)
        final_logits = outputs['final_logits']
        prop_weight = outputs['prop_weight']
        
        # 计算损失
        loss = criterion(final_logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        
        probs = torch.softmax(final_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        labels_np = labels.cpu().numpy()
        valid_mask = labels_np != -1
        
        all_preds.extend(preds.cpu().numpy()[valid_mask])
        all_labels.extend(labels_np[valid_mask])
        all_probs.extend(probs[:, 1].detach().cpu().numpy()[valid_mask])
        all_weights.extend(prop_weight.detach().cpu().numpy()[valid_mask, 0])
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_probs)
        avg_weight = np.mean(all_weights)
    else:
        acc = precision = recall = f1 = sensitivity = specificity = auroc = avg_weight = 0.0
    
    return avg_loss, acc, precision, recall, f1, sensitivity, specificity, auroc, avg_weight


def evaluate_stage3(model, dataloader, criterion, device, show_report=False):
    """评估 Stage 3 的性能"""
    model.eval()
    model.set_training_stage(3)
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_weights = []
    
    with torch.no_grad():
        for fc, sc, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            fc = fc.to(device)
            sc = sc.to(device)
            labels = labels.to(device)
            
            outputs = model(fc, sc)
            final_logits = outputs['final_logits']
            prop_weight = outputs['prop_weight']
            
            loss = criterion(final_logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(final_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            labels_np = labels.cpu().numpy()
            valid_mask = labels_np != -1
            
            all_preds.extend(preds.cpu().numpy()[valid_mask])
            all_labels.extend(labels_np[valid_mask])
            all_probs.extend(probs[:, 1].cpu().numpy()[valid_mask])
            all_weights.extend(prop_weight.cpu().numpy()[valid_mask, 0])
    
    avg_loss = total_loss / len(dataloader)
    
    if len(all_labels) > 0 and len(set(all_labels)) > 1:
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        sensitivity, specificity = compute_sensitivity_specificity(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_probs)
        avg_weight = np.mean(all_weights)
        
        if show_report:
            print("\n" + "=" * 60)
            print("Classification Report:")
            print("=" * 60)
            print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0))
            cm = confusion_matrix(all_labels, all_preds)
            print("\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"              Neg      Pos")
            print(f"Actual Neg   {cm[0, 0]:4d}    {cm[0, 1]:4d}")
            print(f"       Pos   {cm[1, 0]:4d}    {cm[1, 1]:4d}")
            print(f"\nSensitivity (Recall/TPR): {sensitivity:.4f}")
            print(f"Specificity (TNR):        {specificity:.4f}")
            print(f"Average Propensity Weight: {avg_weight:.4f} (接近1偏向FC, 接近0偏向SC)")
            print("=" * 60 + "\n")
    else:
        acc = precision = recall = f1 = sensitivity = specificity = auroc = avg_weight = 0.0
    
    return avg_loss, acc, precision, recall, f1, sensitivity, specificity, auroc, avg_weight


def train_xlearner(train_data, val_data, test_data, num_nodes, config, device, task_idx='xlearner'):
    """
    X-Learner 完整训练流程
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        num_nodes: 节点数
        config: 配置字典
        device: 设备
        task_idx: 任务标识（用于保存模型）
    """
    print(f"\n{'#' * 80}")
    print(f"# X-Learner 训练流程 - 任务 {task_idx}")
    print(f"{'#' * 80}\n")
    
    print_task_summary(task_idx, train_data, val_data, test_data)
    
    num_classes = get_num_classes(train_data)
    train_labels = [d['label'] for d in train_data]
    train_labels_valid = [l for l in train_labels if l != -1]
    class_counts = dict(Counter(train_labels_valid))
    
    # 准备数据加载器
    train_dataset = BrainDataset(train_data)
    val_dataset = BrainDataset(val_data)
    test_dataset = BrainDataset(test_data) if test_data else None
    
    if config.get('use_balanced_sampler', True):
        minority_ratio = config.get('minority_ratio', 0.3)
        print(f"使用平衡采样器 (batch_size={config['batch_size']}, minority_ratio={minority_ratio})")
        train_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=config['batch_size'],
            minority_ratio=minority_ratio,
            ignore_label=-1
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # 创建模型
    model = XLearnerModel(
        num_nodes=num_nodes,
        d_model=config['d_model'],
        num_classes=num_classes,
        dropout=config.get('dropout', 0.4)
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    print_loss_info(config['loss_type'], gamma=config.get('focal_gamma', 2.0))
    criterion = get_loss_function(
        loss_type=config['loss_type'],
        gamma=config.get('focal_gamma', 2.0),
        class_counts=class_counts,
        ignore_index=-1
    )
    
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    if hasattr(criterion, 'alpha') and isinstance(criterion.alpha, torch.Tensor):
        criterion.alpha = criterion.alpha.to(device)
    
    # ========== Stage 1: 训练基础预测器 ==========
    print("\n" + "="*80)
    print("STAGE 1: 训练基础预测器 (FC-only & SC-only)")
    print("="*80)
    
    model.set_training_stage(1)
    optimizer_stage1 = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('stage1_lr', config['lr']),
        weight_decay=config['weight_decay']
    )
    
    best_stage1_acc = 0
    patience_counter = 0
    stage1_path = f'xlearner_stage1_task_{task_idx}.pth'
    
    for epoch in range(config.get('stage1_epochs', 30)):
        (fc_loss, sc_loss, fc_acc, sc_acc, fc_recall, sc_recall,
         fc_sens, fc_spec, sc_sens, sc_spec, fc_auroc, sc_auroc) = train_stage1(
            model, train_loader, criterion, optimizer_stage1, device
        )
        
        # 验证 (使用 FC 和 SC 的平均性能)
        model.eval()
        model.set_training_stage(1)
        val_fc_preds, val_sc_preds = [], []
        val_labels = []
        
        with torch.no_grad():
            for fc, sc, labels in val_loader:
                fc = fc.to(device)
                sc = sc.to(device)
                labels_np = labels.numpy()
                
                outputs = model(fc, sc)
                fc_preds = torch.argmax(outputs['fc_logits'], dim=1).cpu().numpy()
                sc_preds = torch.argmax(outputs['sc_logits'], dim=1).cpu().numpy()
                
                valid_mask = labels_np != -1
                val_fc_preds.extend(fc_preds[valid_mask])
                val_sc_preds.extend(sc_preds[valid_mask])
                val_labels.extend(labels_np[valid_mask])
        
        val_fc_acc = accuracy_score(val_labels, val_fc_preds)
        val_sc_acc = accuracy_score(val_labels, val_sc_preds)
        val_fc_recall = recall_score(val_labels, val_fc_preds, average='binary', zero_division=0)
        val_sc_recall = recall_score(val_labels, val_sc_preds, average='binary', zero_division=0)
        val_fc_sens, val_fc_spec = compute_sensitivity_specificity(val_labels, val_fc_preds)
        val_sc_sens, val_sc_spec = compute_sensitivity_specificity(val_labels, val_sc_preds)
        val_avg_acc = (val_fc_acc + val_sc_acc) / 2
        
        print(f"Epoch [{epoch + 1}/{config.get('stage1_epochs', 30)}]")
        print(f"  Train - FC: Loss={fc_loss:.4f}, Acc={fc_acc:.4f}, Recall={fc_recall:.4f}, Sens={fc_sens:.4f}, Spec={fc_spec:.4f}, AUROC={fc_auroc:.4f}")
        print(f"          SC: Loss={sc_loss:.4f}, Acc={sc_acc:.4f}, Recall={sc_recall:.4f}, Sens={sc_sens:.4f}, Spec={sc_spec:.4f}, AUROC={sc_auroc:.4f}")
        print(f"  Val   - FC: Acc={val_fc_acc:.4f}, Recall={val_fc_recall:.4f}, Sens={val_fc_sens:.4f}, Spec={val_fc_spec:.4f}")
        print(f"          SC: Acc={val_sc_acc:.4f}, Recall={val_sc_recall:.4f}, Sens={val_sc_sens:.4f}, Spec={val_sc_spec:.4f}, Avg={val_avg_acc:.4f}")
        
        if val_avg_acc > best_stage1_acc:
            best_stage1_acc = val_avg_acc
            patience_counter = 0
            torch.save(model.state_dict(), stage1_path)
            print(f"  ✓ Saved Stage 1 model (Avg Acc: {val_avg_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get('stage1_patience', 10):
                print(f"\nEarly stopping in Stage 1!")
                break
    
    # 加载最佳 Stage 1 模型
    model.load_state_dict(torch.load(stage1_path))
    print(f"\nStage 1 完成! 最佳平均准确率: {best_stage1_acc:.4f}")
    
    # ========== Stage 2: 训练效应学习器 ==========
    print("\n" + "="*80)
    print("STAGE 2: 训练效应学习器")
    print("="*80)
    
    # 冻结基础预测器
    model.freeze_stage(1)
    
    model.set_training_stage(2)
    optimizer_stage2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('stage2_lr', config['lr'] * 0.5),
        weight_decay=config['weight_decay']
    )
    
    best_stage2_loss = float('inf')
    patience_counter = 0
    stage2_path = f'xlearner_stage2_task_{task_idx}.pth'
    
    for epoch in range(config.get('stage2_epochs', 20)):
        train_loss = train_stage2(model, train_loader, criterion, optimizer_stage2, device)
        
        # 验证损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for fc, sc, labels in val_loader:
                fc = fc.to(device)
                sc = sc.to(device)
                labels = labels.to(device)
                
                outputs = model(fc, sc)
                
                # 创建有效样本 mask
                valid_mask = labels != -1
                
                # 安全地转换为 one-hot
                labels_safe = labels.clone()
                labels_safe[labels == -1] = 0
                labels_onehot = F.one_hot(labels_safe, num_classes=2).float()
                labels_onehot[~valid_mask] = 0
                
                fc_probs = F.softmax(outputs['fc_logits'], dim=1)
                sc_probs = F.softmax(outputs['sc_logits'], dim=1)
                
                target_effect_fc = labels_onehot - fc_probs
                target_effect_sc = labels_onehot - sc_probs
                
                # 只对有效样本计算损失
                if valid_mask.sum() > 0:
                    loss_fc = F.mse_loss(outputs['effect_fc'][valid_mask], target_effect_fc[valid_mask])
                    loss_sc = F.mse_loss(outputs['effect_sc'][valid_mask], target_effect_sc[valid_mask])
                    val_loss += (loss_fc + loss_sc).item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch + 1}/{config.get('stage2_epochs', 20)}]")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_stage2_loss:
            best_stage2_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), stage2_path)
            print(f"  ✓ Saved Stage 2 model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get('stage2_patience', 5):
                print(f"\nEarly stopping in Stage 2!")
                break
    
    # 加载最佳 Stage 2 模型
    model.load_state_dict(torch.load(stage2_path))
    print(f"\nStage 2 完成! 最佳验证损失: {best_stage2_loss:.4f}")
    
    # ========== Stage 3: 训练倾向性网络 ==========
    print("\n" + "="*80)
    print("STAGE 3: 训练倾向性网络并端到端微调")
    print("="*80)
    
    # 可选：解冻所有参数进行端到端微调
    if config.get('stage3_finetune_all', False):
        model.unfreeze_all()
        print("解冻所有参数，进行端到端微调")
    else:
        # 仅训练倾向性网络
        print("仅训练倾向性网络，冻结其他部分")
    
    model.set_training_stage(3)
    optimizer_stage3 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('stage3_lr', config['lr'] * 0.1),
        weight_decay=config['weight_decay']
    )
    
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    final_path = f'xlearner_final_task_{task_idx}.pth'
    
    for epoch in range(config.get('stage3_epochs', 30)):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_sens, train_spec, train_auroc, train_weight = train_stage3(
            model, train_loader, criterion, optimizer_stage3, device
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_sens, val_spec, val_auroc, val_weight = evaluate_stage3(
            model, val_loader, criterion, device
        )
        
        print(f"Epoch [{epoch + 1}/{config.get('stage3_epochs', 30)}]")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
              f"Sens: {train_sens:.4f}, Spec: {train_spec:.4f}, AUROC: {train_auroc:.4f}, Weight: {train_weight:.3f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
              f"Sens: {val_sens:.4f}, Spec: {val_spec:.4f}, AUROC: {val_auroc:.4f}, Weight: {val_weight:.3f}")
        
        if val_acc > best_val_acc or val_f1 > best_val_f1:
            best_val_acc = max(val_acc, best_val_acc)
            best_val_f1 = max(val_f1, best_val_f1)
            patience_counter = 0
            torch.save(model.state_dict(), final_path)
            print(f"  ✓ Saved final model (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get('stage3_patience', 10):
                print(f"\nEarly stopping in Stage 3!")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(final_path))
    
    # 最终评估
    print(f"\n{'='*80}")
    print("最终评估 - 验证集:")
    print(f"{'='*80}")
    _, val_acc, val_prec, val_rec, val_f1, val_sens, val_spec, val_auroc, val_weight = evaluate_stage3(
        model, val_loader, criterion, device, show_report=True
    )
    
    test_acc, test_sens, test_spec = 0.0, 0.0, 0.0
    if test_loader is not None:
        print(f"\n{'='*80}")
        print("最终评估 - 测试集:")
        print(f"{'='*80}")
        test_loss, test_acc, test_prec, test_rec, test_f1, test_sens, test_spec, test_auroc, test_weight = evaluate_stage3(
            model, test_loader, criterion, device, show_report=True
        )
        
        print(f"\n{'*' * 100}")
        print(f"测试集性能:")
        print(f"{'*' * 100}")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall: {test_rec:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  Sensitivity: {test_sens:.4f}")
        print(f"  Specificity: {test_spec:.4f}")
        print(f"  AUROC: {test_auroc:.4f}")
        print(f"  Avg Propensity Weight: {test_weight:.4f}")
        print(f"{'*' * 100}\n")
    
    return val_acc, val_sens, val_spec, test_acc, test_sens, test_spec


def load_config_from_yaml(config_path):
    """从 YAML 文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='X-Learner 训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, default='args/abide.yml', help='YAML 配置文件路径')
    
    # 数据
    parser.add_argument('--dataset', type=str, default='ABCD', choices=['abide', 'ABCD'],
                        help='数据集: abide (单任务) 或 ABCD (多任务)')
    parser.add_argument('--data_path', type=str,
                        default=r'W:\Brain Analysis\data\data\data_dict.pkl',
                        help='数据文件路径 (.npy for ABIDE, .pkl for ABCD)')
    parser.add_argument('--use_split_data', action='store_true', default=False,
                        help='使用预分割数据')
    parser.add_argument('--train_data_path', type=str,
                        default=r'W:\Brain Analysis\FSCN\data\abide_train.npy')
    parser.add_argument('--val_data_path', type=str,
                        default=r'W:\Brain Analysis\FSCN\data\abide_val.npy')
    parser.add_argument('--test_data_path', type=str,
                        default=r'W:\Brain Analysis\FSCN\data\abide_test.npy')
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # 模型
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.4)
    
    # 训练 - Stage 1
    parser.add_argument('--stage1_epochs', type=int, default=30)
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage1_patience', type=int, default=10)
    
    # 训练 - Stage 2
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--stage2_lr', type=float, default=5e-5)
    parser.add_argument('--stage2_patience', type=int, default=5)
    
    # 训练 - Stage 3
    parser.add_argument('--stage3_epochs', type=int, default=30)
    parser.add_argument('--stage3_lr', type=float, default=1e-5)
    parser.add_argument('--stage3_patience', type=int, default=10)
    parser.add_argument('--stage3_finetune_all', action='store_true', default=False,
                        help='Stage 3 是否端到端微调所有参数')
    
    # 通用训练参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--minority_ratio', type=float, default=0.3)
    
    # 损失函数
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal', 'weighted_ce', 'weighted_focal'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--tasks', type=int, nargs='+', default=None,
                        help='指定要训练的任务 ID（仅用于 ABCD 数据集，默认: [4,5,7,8]）')
    
    args = parser.parse_args()
    
    # 从 YAML 加载配置并覆盖
    yaml_config = load_config_from_yaml(args.config)
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # 为 ABCD 数据集设置默认任务
    if args.tasks is None and args.dataset == 'ABCD':
        args.tasks = [4, 5, 7, 8]
    
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    config = vars(args)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("X-Learner 训练配置:")
    print("=" * 80)
    for key, value in config.items():
        if key != 'device':
            print(f"  {key}: {value}")
    print(f"  device: {device}")
    print("=" * 80 + "\n")
    
    # 加载数据
    if args.dataset == 'abide':
        # ABIDE 单任务数据集
        if args.use_split_data:
            print(f"加载预分割的 ABIDE 数据集...")
            train_dict, val_dict, test_dict, num_nodes = load_abide_split_data(
                args.train_data_path,
                args.val_data_path,
                args.test_data_path
            )
            train_data, val_data, test_data = prepare_task_data_from_dict(
                train_dict, val_dict, test_dict, task_idx=None
            )
        else:
            print(f"加载 ABIDE 数据集...")
            data_dict, num_nodes = load_abide_data(args.data_path)
            train_data, val_data = prepare_task_data(
                data_dict,
                task_idx=None,
                val_split=args.val_split,
                random_state=args.seed
            )
            test_data = []
        
        # 训练单个任务
        val_acc, val_sens, val_spec, test_acc, test_sens, test_spec = train_xlearner(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            num_nodes=num_nodes,
            config=config,
            device=device,
            task_idx='abide'
        )
        
        print("\n" + "=" * 80)
        print("X-Learner 训练完成!")
        print("=" * 80)
        print(f"\n验证集 (最佳模型):")
        print(f"  Accuracy:    {val_acc:.4f}")
        print(f"  Sensitivity: {val_sens:.4f}")
        print(f"  Specificity: {val_spec:.4f}")
        if args.use_split_data:
            print(f"\n测试集:")
            print(f"  Accuracy:    {test_acc:.4f}")
            print(f"  Sensitivity: {test_sens:.4f}")
            print(f"  Specificity: {test_spec:.4f}")
        print("=" * 80 + "\n")
    
    else:
        # ABCD 多任务数据集
        print(f"加载 ABCD 数据集...")
        data_dict, num_nodes = load_raw_data(args.data_path)
        
        results = {}
        
        print(f"\n将训练以下任务: {args.tasks}\n")
        
        for task_idx in args.tasks:
            print(f"\n{'#' * 80}")
            print(f"# 任务 {task_idx}")
            print(f"{'#' * 80}\n")
            
            # 准备任务数据
            train_data, val_data = prepare_task_data(
                data_dict,
                task_idx=task_idx,
                val_split=args.val_split,
                random_state=args.seed
            )
            test_data = []  # ABCD 没有单独的测试集
            
            # 训练
            val_acc, val_sens, val_spec, test_acc, test_sens, test_spec = train_xlearner(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                num_nodes=num_nodes,
                config=config,
                device=device,
                task_idx=task_idx
            )
            
            results[task_idx] = {
                'val_acc': val_acc,
                'val_sens': val_sens,
                'val_spec': val_spec
            }
        
        # 打印所有任务的总结
        print("\n" + "=" * 80)
        print("所有任务训练完成!")
        print("=" * 80)
        print("\n任务性能总结:")
        print(f"{'任务 ID':<10} {'准确率':<15} {'敏感度':<15} {'特异度':<15} {'状态'}")
        print("-" * 70)
        for task_idx in args.tasks:
            acc = results[task_idx]['val_acc']
            sens = results[task_idx]['val_sens']
            spec = results[task_idx]['val_spec']
            print(f"{task_idx:<10} {acc:<15.4f} {sens:<15.4f} {spec:<15.4f} 完成")
        
        avg_acc = np.mean([r['val_acc'] for r in results.values()])
        avg_sens = np.mean([r['val_sens'] for r in results.values()])
        avg_spec = np.mean([r['val_spec'] for r in results.values()])
        print("-" * 70)
        print(f"{'平均':<10} {avg_acc:<15.4f} {avg_sens:<15.4f} {avg_spec:<15.4f}")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

