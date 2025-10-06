#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_loader import BrainNetworkDataset, create_data_loaders
from models import get_model
from trainer import BrainNetworkTrainer, GraphVAETrainer, GraphDiffusionTrainer


class BrainNetworkExperiment:
    """
    脑网络数据预测实验主类
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 output_dir: str = './experiments'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        self.set_seed(config.get('seed', 42))
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 结果存储
        self.results = {}
        
    def set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 创建数据集
        self.dataset = BrainNetworkDataset(
            root=self.config['data']['root'],
            name=self.config['data']['name'],
            threshold=self.config['data']['threshold']
        )
        
        print(f"数据集大小: {len(self.dataset)}")
        if len(self.dataset) > 0:
            print(f"节点特征维度: {self.dataset[0].x.shape[1]}")
            # 根据数据集类型显示不同信息
            if self.dataset.name in ['HCPFI', 'HCPWM']:
                print(f"任务类型: 回归")
                print(f"输出维度: 1")
            else:
                all_labels = torch.cat([data.y for data in self.dataset])
                print(f"任务类型: 分类")
                print(f"类别数: {len(torch.unique(all_labels))}")
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader, self.data_info = create_data_loaders(
            self.dataset,
            batch_size=self.config['training']['batch_size'],
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size'],
            random_state=self.config['seed']
        )
        
        print(f"训练集大小: {self.data_info['train_size']}")
        print(f"验证集大小: {self.data_info['val_size']}")
        print(f"测试集大小: {self.data_info['test_size']}")
        
    def run_single_experiment(self, model_name: str) -> Dict[str, Any]:
        """运行单个模型实验"""
        print(f"\n{'='*60}")
        print(f"开始实验: {model_name.upper()}")
        print(f"{'='*60}")
        
        # 创建模型
        model = get_model(
            model_name=model_name,
            input_dim=self.data_info['num_features'],
            num_classes=self.data_info['num_classes'],
            task_type=self.data_info['task_type'],
            **self.config['models'][model_name]
        )
        
        # 创建训练器
        if model_name == 'graphvae':
            trainer = GraphVAETrainer(
                model=model,
                device=self.device,
                log_dir=self.output_dir / f'{model_name}_logs',
                save_dir=self.output_dir / f'{model_name}_checkpoints'
            )
        elif model_name == 'graphdiffusion':
            trainer = GraphDiffusionTrainer(
                model=model,
                device=self.device,
                log_dir=self.output_dir / f'{model_name}_logs',
                save_dir=self.output_dir / f'{model_name}_checkpoints'
            )
        else:
            trainer = BrainNetworkTrainer(
                model=model,
                device=self.device,
                log_dir=self.output_dir / f'{model_name}_logs',
                save_dir=self.output_dir / f'{model_name}_checkpoints'
            )
        
        # 训练模型
        start_time = time.time()
        
        if model_name == 'graphdiffusion':
            # 扩散模型需要先预训练
            pretrain_results = trainer.pretrain_diffusion(
                self.train_loader,
                num_epochs=self.config['training']['pretrain_epochs'],
                learning_rate=self.config['training']['learning_rate']
            )
        
        # 正常训练
        train_results = trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=self.config['training']['num_epochs'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            patience=self.config['training']['patience'],
            model_name=model_name
        )
        
        # 测试模型
        test_results = trainer.evaluate(
            test_loader=self.test_loader,
            model_path=self.output_dir / f'{model_name}_checkpoints' / f'{model_name}_best.pth'
        )
        
        total_time = time.time() - start_time
        
        # 收集结果
        experiment_results = {
            'model_name': model_name,
            'train_results': train_results,
            'test_results': test_results,
            'total_time': total_time,
            'data_info': self.data_info
        }
        
        if model_name == 'graphdiffusion':
            experiment_results['pretrain_results'] = pretrain_results
        
        # 保存结果
        self.save_experiment_results(model_name, experiment_results)
        
        # 关闭训练器
        trainer.close()
        
        print(f"\n{model_name.upper()} 实验完成!")
        print(f"最佳验证准确率: {train_results['best_val_acc']:.4f}")
        print(f"测试准确率: {test_results['accuracy']:.4f}")
        print(f"总用时: {total_time:.2f}s")
        
        return experiment_results
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有模型实验"""
        print("开始运行所有模型实验...")
        
        models_to_run = self.config['experiment']['models']
        
        for model_name in models_to_run:
            try:
                results = self.run_single_experiment(model_name)
                self.results[model_name] = results
            except Exception as e:
                print(f"❌ {model_name.upper()} 实验失败: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # 生成比较报告
        self.generate_comparison_report()
        
        return self.results
    
    def save_experiment_results(self, model_name: str, results: Dict[str, Any]):
        """保存单个实验结果"""
        results_file = self.output_dir / f'{model_name}_results.json'
        
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def generate_comparison_report(self):
        """生成比较报告"""
        print("\n" + "="*80)
        print("实验结果比较报告")
        print("="*80)
        
        # 创建比较表格
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Status': 'Failed',
                    'Error': results['error'],
                    'Val_Acc': 'N/A',
                    'Test_Acc': 'N/A',
                    'Test_F1': 'N/A',
                    'Test_AUC': 'N/A',
                    'Time(s)': 'N/A'
                })
            else:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Status': 'Success',
                    'Error': 'N/A',
                    'Val_Acc': f"{results['train_results']['best_val_acc']:.4f}",
                    'Test_Acc': f"{results['test_results']['accuracy']:.4f}",
                    'Test_F1': f"{results['test_results']['metrics']['f1']:.4f}",
                    'Test_AUC': f"{results['test_results']['metrics']['auc']:.4f}",
                    'Time(s)': f"{results['total_time']:.2f}"
                })
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_data)
        
        # 保存为CSV
        csv_file = self.output_dir / 'experiment_comparison.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 打印表格
        print(df.to_string(index=False))
        
        # 保存为JSON
        json_file = self.output_dir / 'experiment_comparison.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        # 找出最佳模型
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if successful_results:
            best_model = max(successful_results.items(), 
                           key=lambda x: x[1]['test_results']['accuracy'])
            print(f"\n🏆 最佳模型: {best_model[0].upper()}")
            print(f"   测试准确率: {best_model[1]['test_results']['accuracy']:.4f}")
            print(f"   测试F1分数: {best_model[1]['test_results']['metrics']['f1']:.4f}")
            print(f"   测试AUC: {best_model[1]['test_results']['metrics']['auc']:.4f}")
        
        print(f"\n结果已保存到: {self.output_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    config = {
        "seed": 42,
        "data": {
            "root": "./data",
            "name": "HCPGender",
            "threshold": 0.1,
            "test_size": 0.2,
            "val_size": 0.1
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 100,
            "pretrain_epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "patience": 20
        },
        "models": {
            "gcn": {
                "hidden_dims": [64, 32],
                "dropout": 0.5,
                "pool_type": "mean"
            },
            "graphsage": {
                "hidden_dims": [64, 32],
                "dropout": 0.5,
                "pool_type": "mean"
            },
            "gat": {
                "hidden_dims": [64, 32],
                "dropout": 0.5,
                "pool_type": "mean",
                "heads": [4, 4]
            },
            "graphvae": {
                "hidden_dims": [64, 32],
                "latent_dim": 128,
                "dropout": 0.5,
                "pool_type": "mean"
            },
            "graphdiffusion": {
                "hidden_dims": [64, 32],
                "dropout": 0.5,
                "pool_type": "mean",
                "num_timesteps": 1000
            }
        },
        "experiment": {
            "models": ["gcn", "graphsage", "gat", "graphvae", "graphdiffusion"]
        }
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='脑网络数据预测实验')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--dataset', type=str, default='HCPGender', 
                       choices=['HCPGender', 'HCPTask', 'HCPAge', 'HCPFI', 'HCPWM'],
                       help='要使用的NeuroGraphDataset名称')
    parser.add_argument('--models', nargs='+', default=['gcn', 'graphsage', 'gat', 'graphvae', 'graphdiffusion'],
                       help='要运行的模型列表')
    parser.add_argument('--output_dir', type=str, default='./experiments', help='输出目录')
    parser.add_argument('--create_config', action='store_true', help='创建默认配置文件')
    
    args = parser.parse_args()
    
    # 创建默认配置文件
    if args.create_config:
        config = create_default_config()
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("默认配置文件已创建: config.json")
        return
    
    # 加载配置
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"配置文件 {args.config} 不存在，使用默认配置")
        config = create_default_config()
    
    # 更新配置
    config['experiment']['models'] = args.models
    config['data']['name'] = args.dataset
    
    # 创建实验
    experiment = BrainNetworkExperiment(
        config=config,
        output_dir=args.output_dir
    )
    
    # 加载数据
    experiment.load_data()
    
    # 运行实验
    if len(args.models) == 1:
        # 运行单个模型
        results = experiment.run_single_experiment(args.models[0])
        experiment.results[args.models[0]] = results
    else:
        # 运行所有模型
        results = experiment.run_all_experiments()
    
    print("\n🎉 所有实验完成!")


if __name__ == "__main__":
    main()

