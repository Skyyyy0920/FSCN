#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BrainNetworkTrainer:
    """
    脑网络模型训练器
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 log_dir: str = './logs',
                 save_dir: str = './checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
    
    def train_epoch(self, 
                   train_loader, 
                   optimizer: optim.Optimizer, 
                   criterion: nn.Module,
                   epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if hasattr(self.model, 'forward_classification'):
                # GraphDiffusion模型
                output = self.model(batch.x, batch.edge_index, batch.batch, mode='classification')
            elif isinstance(output := self.model(batch.x, batch.edge_index, batch.batch), dict):
                # GraphVAE模型
                output = output['classification']
            else:
                # 其他模型
                output = self.model(batch.x, batch.edge_index, batch.batch)
            
            # 计算损失
            loss = criterion(output, batch.y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * total_correct / total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, 
                     val_loader, 
                     criterion: nn.Module,
                     epoch: int) -> Tuple[float, float, Dict]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(self.device)
                
                # 前向传播
                if hasattr(self.model, 'forward_classification'):
                    # GraphDiffusion模型
                    output = self.model(batch.x, batch.edge_index, batch.batch, mode='classification')
                elif isinstance(output := self.model(batch.x, batch.edge_index, batch.batch), dict):
                    # GraphVAE模型
                    output = output['classification']
                else:
                    # 其他模型
                    output = self.model(batch.x, batch.edge_index, batch.batch)
                
                # 计算损失
                loss = criterion(output, batch.y)
                
                # 统计
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(batch.y).sum().item()
                total_samples += batch.y.size(0)
                
                # 收集预测结果
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * total_correct / total_samples:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        # 计算详细指标
        metrics = self.compute_metrics(all_targets, all_preds, all_probs)
        
        return avg_loss, accuracy, metrics
    
    def compute_metrics(self, targets: List, preds: List, probs: List) -> Dict:
        """计算详细指标"""
        targets = np.array(targets)
        preds = np.array(preds)
        probs = np.array(probs)
        
        # 基础指标
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='weighted')
        
        # AUC (多分类)
        try:
            if len(np.unique(targets)) > 2:
                auc = roc_auc_score(targets, probs, multi_class='ovr')
            else:
                auc = roc_auc_score(targets, probs[:, 1])
        except:
            auc = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(targets, preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
    
    def train(self, 
              train_loader,
              val_loader,
              num_epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 20,
              model_name: str = 'model') -> Dict:
        """训练模型"""
        print(f"开始训练 {model_name}，共 {num_epochs} 个epoch...")
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        
        # 早停
        early_stopping_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # 验证
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader, criterion, epoch)
            
            # 学习率调度
            scheduler.step(val_acc)
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - '
                  f'Time: {epoch_time:.2f}s')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                early_stopping_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_metrics': val_metrics
                }, self.save_dir / f'{model_name}_best.pth')
                
                print(f'✅ 新的最佳模型已保存 (Val Acc: {val_acc:.4f})')
            else:
                early_stopping_counter += 1
            
            # 早停检查
            if early_stopping_counter >= patience:
                print(f'早停触发，在第 {epoch+1} 个epoch停止训练')
                break
        
        total_time = time.time() - start_time
        print(f'训练完成，总用时: {total_time:.2f}s')
        print(f'最佳验证准确率: {self.best_val_acc:.4f} (第 {self.best_epoch+1} 个epoch)')
        
        # 保存训练历史
        self.save_training_history(model_name)
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'final_metrics': val_metrics
        }
    
    def evaluate(self, test_loader, model_path: Optional[str] = None) -> Dict:
        """评估模型"""
        if model_path:
            # 加载最佳模型
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载模型: {model_path}")
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Testing')
            
            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(self.device)
                
                # 前向传播
                if hasattr(self.model, 'forward_classification'):
                    # GraphDiffusion模型
                    output = self.model(batch.x, batch.edge_index, batch.batch, mode='classification')
                elif isinstance(output := self.model(batch.x, batch.edge_index, batch.batch), dict):
                    # GraphVAE模型
                    output = output['classification']
                else:
                    # 其他模型
                    output = self.model(batch.x, batch.edge_index, batch.batch)
                
                # 计算损失
                loss = criterion(output, batch.y)
                
                # 统计
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(batch.y).sum().item()
                total_samples += batch.y.size(0)
                
                # 收集预测结果
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probs.extend(torch.softmax(output, dim=1).cpu().numpy())
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * total_correct / total_samples:.2f}%'
                })
        
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_samples
        
        # 计算详细指标
        metrics = self.compute_metrics(all_targets, all_preds, all_probs)
        
        print(f'测试结果:')
        print(f'损失: {avg_loss:.4f}')
        print(f'准确率: {accuracy:.4f}')
        print(f'精确率: {metrics["precision"]:.4f}')
        print(f'召回率: {metrics["recall"]:.4f}')
        print(f'F1分数: {metrics["f1"]:.4f}')
        print(f'AUC: {metrics["auc"]:.4f}')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'metrics': metrics,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    def save_training_history(self, model_name: str):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        with open(self.save_dir / f'{model_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves(model_name)
    
    def plot_training_curves(self, model_name: str):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.axvline(x=self.best_epoch, color='green', linestyle='--', label=f'Best Epoch ({self.best_epoch+1})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, targets: List, preds: List, model_name: str, class_names: List[str] = None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names or [f'Class {i}' for i in range(len(cm))],
                   yticklabels=class_names or [f'Class {i}' for i in range(len(cm))])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def close(self):
        """关闭训练器"""
        self.writer.close()


class GraphVAETrainer(BrainNetworkTrainer):
    """
    Graph VAE专用训练器
    """
    
    def train_epoch(self, 
                   train_loader, 
                   optimizer: optim.Optimizer, 
                   criterion: nn.Module,
                   epoch: int,
                   lambda_recon: float = 1.0,
                   lambda_kl: float = 1.0) -> Tuple[float, float]:
        """训练一个epoch (VAE版本)"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = self.model(batch.x, batch.edge_index, batch.batch)
            
            # 计算VAE损失 - 使用图级别特征进行重建
            # 先获取图级别的原始特征（通过平均池化）
            from torch_geometric.nn import global_mean_pool
            graph_features = global_mean_pool(batch.x, batch.batch)
            
            loss_dict = self.model.compute_loss(output, batch.y, graph_features, lambda_recon, lambda_kl)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output['classification'].argmax(dim=1)
            total_correct += pred.eq(batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Total Loss': f'{loss.item():.4f}',
                'CE Loss': f'{loss_dict["ce_loss"].item():.4f}',
                'Recon Loss': f'{loss_dict["recon_loss"].item():.4f}',
                'KL Loss': f'{loss_dict["kl_loss"].item():.4f}',
                'Acc': f'{100. * total_correct / total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy


class GraphDiffusionTrainer(BrainNetworkTrainer):
    """
    Graph Diffusion专用训练器
    """
    
    def pretrain_diffusion(self, 
                          train_loader,
                          num_epochs: int = 50,
                          learning_rate: float = 0.001) -> Dict:
        """预训练扩散模型"""
        print("开始预训练扩散模型...")
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}')
            
            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 使用图级别特征进行扩散训练
                # 先通过编码器获取图级别特征
                encoded_features = self.model.encode(batch.x, batch.edge_index, batch.batch)
                
                # 随机时间步
                timestep = torch.randint(0, self.model.num_timesteps, (batch.batch_size,), device=self.device)
                
                # 计算扩散损失
                loss = self.model.compute_diffusion_loss(encoded_features, timestep, batch.edge_index, batch.batch)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Pretrain Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}')
        
        print("扩散模型预训练完成")
        return {'pretrain_loss': avg_loss}


if __name__ == "__main__":
    # 测试训练器
    print("测试训练器...")
    
    # 创建简单的测试模型
    from models import GCNClassifier
    
    model = GCNClassifier(input_dim=8, num_classes=2)
    trainer = BrainNetworkTrainer(model)
    
    print("✅ 训练器测试成功")

