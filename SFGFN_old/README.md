# 脑网络多模态融合框架 (SFGFN)

## 项目简介

本项目实现了一个基于 PyTorch Geometric 的脑网络多模态融合框架，用于整合 fMRI 和 DTI 数据进行预测任务。

## 三种融合方法

### 方法 A: 并行双图编码器 + 门控融合
- 在结构连接 (SC) 和功能连接 (FC) 上分别应用 GNN
- 使用门控机制自适应融合两种表示
- 公式: `H = g ⊙ H_sc + (1-g) ⊙ H_fc`

### 方法 B: 多重图神经网络
- 在每个 GNN 层同时在 SC 和 FC 上传播消息
- 通过求和的方式融合多层次信息
- 公式: `H^(l+1) = GNN_sc(H^(l), A_sc) + GNN_fc(H^(l), A_fc)`

### 方法 C: SC 作为骨干 + FC 调制
- 构建混合邻接矩阵，学习最优融合权重
- SC 提供稳定的结构基础，FC 提供动态调制
- 公式: `A_mix = α·A_sc + (1-α)·A_fc`, 其中 α 是可学习参数

## 项目结构

```
SFGFN/
├── models.py           # 模型定义（三种融合方法 + 共享组件）
├── data_loader.py      # 数据加载和预处理
├── trainer.py          # 训练器（训练、验证、测试逻辑）
├── main.py            # 主实验脚本
├── requirements.txt   # Python 依赖包
└── README.md         # 项目文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: PyTorch Geometric 的安装可能需要根据您的 PyTorch 和 CUDA 版本进行调整。请参考 [PyG 官方文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

## 数据格式

数据应保存为 pickle 文件 (`data_dict.pkl`)，格式如下：

```python
data_dict[idx] = {
    "FC": fc,                  # [N, N] 功能连接矩阵
    "SC": sc,                  # [N, N] 结构连接矩阵
    "label": [label],          # 标签（整数列表）
    "name": subject_id,        # 受试者 ID
}
```

## 使用方法

### 1. 运行单个融合方法

```bash
# 方法 A
python main.py --fusion_method A --num_epochs 100

# 方法 B
python main.py --fusion_method B --num_epochs 100

# 方法 C
python main.py --fusion_method C --num_epochs 100
```

### 2. 运行所有方法并比较

```bash
python main.py --fusion_method all --num_epochs 100
```

### 3. 自定义参数

```bash
python main.py \
    --fusion_method A \
    --data_path "你的数据路径.pkl" \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_gnn_layers 3 \
    --learning_rate 0.0005 \
    --num_epochs 200 \
    --device cuda
```

## 主要参数说明

### 数据参数
- `--data_path`: 数据文件路径（默认: `W:\Brain Analysis\data\data\data_dict.pkl`）
- `--threshold`: 边权重阈值，过滤弱连接（默认: 0.1）

### 模型参数
- `--fusion_method`: 融合方法 [A, B, C, all]（默认: A）
- `--node_feature_dim`: 节点特征维度（默认: 64）
- `--hidden_dim`: 隐藏层维度（默认: 128）
- `--num_gnn_layers`: GNN 层数（默认: 2）
- `--pooling_type`: 图池化方式 [mean, sum, attention]（默认: mean）
- `--task_type`: 任务类型 [classification, regression]（默认: classification）

### 训练参数
- `--batch_size`: 批次大小（默认: 16）
- `--num_epochs`: 训练轮数（默认: 100）
- `--learning_rate`: 学习率（默认: 0.001）
- `--weight_decay`: L2 正则化系数（默认: 1e-4）
- `--early_stopping_patience`: 早停耐心值（默认: 20）

### 其他参数
- `--device`: 计算设备 [cuda, cpu]（默认: cuda）
- `--seed`: 随机种子（默认: 42）
- `--save_dir`: 结果保存目录（默认: results）

## 输出结果

运行后会在指定的保存目录中生成以下文件：

```
results/
├── config.json                    # 实验配置
├── comparison.json                # 方法比较结果（运行 all 时）
├── method_A/
│   ├── best_model.pth            # 最佳模型权重
│   ├── results.json              # 详细结果
│   └── training_curves.png       # 训练曲线图
├── method_B/
│   └── ...
└── method_C/
    └── ...
```

### 结果文件说明

- **config.json**: 记录所有实验参数
- **best_model.pth**: 在验证集上表现最好的模型检查点
- **results.json**: 包含最佳 epoch、测试集性能、训练历史等
- **training_curves.png**: 训练和验证的损失/指标曲线
- **comparison.json**: 所有方法的性能对比（仅在运行 `--fusion_method all` 时生成）

## 模型架构

```
输入: FC 矩阵, SC 矩阵
  ↓
[时间编码器] → 节点嵌入 (H_f)
  ↓
[融合模块] → 融合后的节点嵌入 (H)
  ├─ 方法 A: 并行 GNN + 门控
  ├─ 方法 B: 多重图 GNN
  └─ 方法 C: 混合邻接矩阵
  ↓
[图池化] → 图级别嵌入 (z)
  ↓
[预测器 MLP] → 输出
```

## 评估指标

### 分类任务
- Accuracy (准确率)
- F1-Score (F1 分数)
- AUC (ROC 曲线下面积，仅二分类)

### 回归任务
- MSE (均方误差)
- R² Score (决定系数)

## 注意事项

1. **数据路径**: 确保 `--data_path` 指向正确的数据文件
2. **GPU 内存**: 如果遇到 OOM 错误，尝试减小 `--batch_size` 或 `--hidden_dim`
3. **训练时间**: 完整训练可能需要较长时间，建议使用 GPU
4. **可重复性**: 使用 `--seed` 参数确保实验可重复

## 扩展和定制

### 添加新的融合方法
在 `models.py` 中创建新的融合模块类，继承 `nn.Module`：

```python
class FusionMethodD(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 定义您的层
    
    def forward(self, x, edge_index_sc, edge_weight_sc, 
                edge_index_fc, edge_weight_fc):
        # 实现您的融合逻辑
        return h_fused
```

然后在 `BrainNetworkFusionModel` 中添加选项。

### 自定义损失函数
在 `trainer.py` 的 `Trainer.__init__()` 中修改 `self.criterion`。

### 使用不同的 GNN 层
在融合方法中，将 `GCNConv` 替换为其他 PyG 层（如 `GATConv`, `SAGEConv` 等）。

## 引用

如果您使用了本项目的代码，请考虑引用相关论文。

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
