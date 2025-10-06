import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 test_loader,
                 task_type='classification',
                 num_classes=2,
                 device='cuda',
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 save_dir='checkpoints'):
        """
        Args:
            model: 神经网络模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            task_type: 'classification' 或 'regression'
            num_classes: 分类任务的类别数
            device: 'cuda' 或 'cpu'
            learning_rate: 学习率
            weight_decay: L2 正则化系数
            save_dir: 模型保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 损失函数
        if task_type == 'classification':
            if num_classes == 2:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self):
        """
        训练一个 epoch
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 计算损失
            if self.task_type == 'classification':
                if self.num_classes == 2:
                    # 二分类
                    loss = self.criterion(output.squeeze(), data.y.float())
                    preds = (torch.sigmoid(output.squeeze()) > 0.5).long()
                else:
                    # 多分类
                    loss = self.criterion(output, data.y)
                    preds = output.argmax(dim=1)
            else:
                # 回归
                loss = self.criterion(output.squeeze(), data.y.float())
                preds = output.squeeze()
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        metric = self._calculate_metric(all_labels, all_preds)
        
        return avg_loss, metric
    
    def validate(self, loader):
        """
        验证/测试
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 计算损失
                if self.task_type == 'classification':
                    if self.num_classes == 2:
                        loss = self.criterion(output.squeeze(), data.y.float())
                        probs = torch.sigmoid(output.squeeze())
                        preds = (probs > 0.5).long()
                        all_probs.extend(probs.cpu().numpy())
                    else:
                        loss = self.criterion(output, data.y)
                        probs = torch.softmax(output, dim=1)
                        preds = output.argmax(dim=1)
                        all_probs.extend(probs.cpu().numpy())
                else:
                    loss = self.criterion(output.squeeze(), data.y.float())
                    preds = output.squeeze()
                
                total_loss += loss.item() * data.num_graphs
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        metric = self._calculate_metric(all_labels, all_preds)
        
        # 计算额外指标
        extra_metrics = {}
        if self.task_type == 'classification' and len(all_probs) > 0:
            all_labels_np = np.array(all_labels)
            if self.num_classes == 2:
                try:
                    extra_metrics['auc'] = roc_auc_score(all_labels_np, all_probs)
                except:
                    extra_metrics['auc'] = 0.0
            all_preds_np = np.array(all_preds)
            extra_metrics['f1'] = f1_score(all_labels_np, all_preds_np, average='weighted')
        
        return avg_loss, metric, extra_metrics
    
    def _calculate_metric(self, labels, preds):
        """
        计算评估指标
        """
        if self.task_type == 'classification':
            return accuracy_score(labels, preds)
        else:
            return r2_score(labels, preds)
    
    def train(self, num_epochs=100, early_stopping_patience=20):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        print(f"开始训练 ({self.task_type})...")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print("-" * 80)
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss, train_metric = self.train_epoch()
            
            # 验证
            val_loss, val_metric, val_extra = self.validate(self.val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            self.history['learning_rate'].append(current_lr)
            
            # 打印进度
            epoch_time = time.time() - start_time
            metric_name = 'Acc' if self.task_type == 'classification' else 'R2'
            
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f} | Train {metric_name}: {train_metric:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Val   {metric_name}: {val_metric:.4f}", end="")
            
            if val_extra:
                for key, value in val_extra.items():
                    print(f" | {key.upper()}: {value:.4f}", end="")
            print()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发 (耐心值: {early_stopping_patience})")
                break
            
            print("-" * 80)
        
        print(f"\n训练完成! 最佳 epoch: {self.best_epoch}")
        
        # 加载最佳模型并在测试集上评估
        self.load_checkpoint('best_model.pth')
        test_loss, test_metric, test_extra = self.validate(self.test_loader)
        
        print(f"\n{'='*80}")
        print(f"测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  {metric_name}: {test_metric:.4f}")
        if test_extra:
            for key, value in test_extra.items():
                print(f"  {key.upper()}: {value:.4f}")
        print(f"{'='*80}\n")
        
        # 保存训练历史和测试结果
        results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'test_loss': test_loss,
            f'test_{metric_name.lower()}': test_metric,
            **{f'test_{k}': v for k, v in test_extra.items()},
            'history': self.history
        }
        
        # 保存融合参数（如果有）
        fusion_params = self.model.get_fusion_params()
        if fusion_params:
            results['fusion_params'] = fusion_params
            print(f"融合参数: {fusion_params}")
        
        self.save_results(results)
        self.plot_curves()
        
        return results
    
    def save_checkpoint(self, filename):
        """
        保存模型检查点
        """
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, filepath)
    
    def load_checkpoint(self, filename):
        """
        加载模型检查点
        """
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
    
    def save_results(self, results):
        """
        保存训练结果为 JSON 文件
        """
        filepath = os.path.join(self.save_dir, 'results.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"结果已保存到: {filepath}")
    
    def plot_curves(self):
        """
        绘制训练曲线
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].axvline(x=self.best_epoch-1, color='r', linestyle='--', label='Best Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 指标曲线
        metric_name = 'Accuracy' if self.task_type == 'classification' else 'R² Score'
        axes[1].plot(self.history['train_metric'], label=f'Train {metric_name}')
        axes[1].plot(self.history['val_metric'], label=f'Val {metric_name}')
        axes[1].axvline(x=self.best_epoch-1, color='r', linestyle='--', label='Best Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Training and Validation {metric_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        filepath = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {filepath}")
        plt.close()
