import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
import networkx as nx

# 准备测试数据
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1])

# 两种不同的边
edge_index_sc = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # 节点0-1之间
edge_weight_sc = torch.tensor([0.5, 0.5])

edge_index_fc = torch.tensor([[2, 3], [3, 2]], dtype=torch.long)  # 节点2-3之间
edge_weight_fc = torch.tensor([0.8, 0.8])

print("=" * 60)
print("场景1: 使用自定义属性名（你的原始方法）")
print("=" * 60)

# 你的方法：使用自定义属性名
data_custom = Data(
    x=x,
    edge_index_sc=edge_index_sc,
    edge_weight_sc=edge_weight_sc,
    edge_index_fc=edge_index_fc,
    edge_weight_fc=edge_weight_fc,
    y=y,
    num_nodes=4
)

print("\n✓ 数据创建成功")
print(f"data_custom.edge_index_sc: {data_custom.edge_index_sc.shape}")
print(f"data_custom.edge_index_fc: {data_custom.edge_index_fc.shape}")

# 问题1: to_networkx() 只处理标准的 edge_index
print("\n--- 测试 to_networkx() ---")
try:
    G_custom = to_networkx(data_custom, to_undirected=True)
    print(f"⚠️  转换成功，但只有 {G_custom.number_of_edges()} 条边")
    print(f"   原因：没有标准的 edge_index，所以图是空的或只有节点")
    print(f"   节点数: {G_custom.number_of_nodes()}, 边数: {G_custom.number_of_edges()}")
except Exception as e:
    print(f"❌ 转换失败: {e}")

# 问题2: subgraph() 只处理标准的 edge_index
print("\n--- 测试 subgraph() ---")
subset = torch.tensor([0, 1])  # 只取前两个节点
try:
    edge_idx, edge_attr = subgraph(subset, data_custom.edge_index_sc)
    print(f"✓ 需要手动指定 edge_index_sc")
    print(f"   但这意味着你要为每个自定义边都写一遍代码")
except Exception as e:
    print(f"❌ 失败: {e}")


print("\n" + "=" * 60)
print("场景2: 使用标准属性名 + 边类型标识（推荐方法）")
print("=" * 60)

# 推荐方法：合并边并添加类型标识
edge_index_combined = torch.cat([edge_index_sc, edge_index_fc], dim=1)
edge_weight_combined = torch.cat([edge_weight_sc, edge_weight_fc])
edge_type = torch.cat([
    torch.zeros(edge_index_sc.size(1), dtype=torch.long),  # 0 代表 sc
    torch.ones(edge_index_fc.size(1), dtype=torch.long)     # 1 代表 fc
])

data_standard = Data(
    x=x,
    edge_index=edge_index_combined,
    edge_weight=edge_weight_combined,
    edge_type=edge_type,
    y=y,
    num_nodes=4
)

print("\n✓ 数据创建成功")
print(f"总边数: {data_standard.edge_index.shape[1]}")
print(f"SC类型边数: {(edge_type == 0).sum().item()}")
print(f"FC类型边数: {(edge_type == 1).sum().item()}")

# 优势1: to_networkx() 自动工作
print("\n--- 测试 to_networkx() ---")
G_standard = to_networkx(data_standard, to_undirected=True,
                         edge_attrs=['edge_weight', 'edge_type'])
print(f"✓ 转换成功!")
print(f"   节点数: {G_standard.number_of_nodes()}")
print(f"   边数: {G_standard.number_of_edges()}")
print(f"   边属性被保留:")
for u, v, attr in G_standard.edges(data=True):
    print(f"     边 ({u}, {v}): weight={attr.get('edge_weight', 'N/A')}, "
          f"type={attr.get('edge_type', 'N/A')}")

# 优势2: subgraph() 自动工作
print("\n--- 测试 subgraph() ---")
subset = torch.tensor([0, 1])
sub_edge_index, sub_edge_attr = subgraph(
    subset,
    data_standard.edge_index,
    edge_attr=data_standard.edge_weight,
    relabel_nodes=True
)
print(f"✓ 子图提取成功!")
print(f"   子图边索引: {sub_edge_index}")
print(f"   子图边权重: {sub_edge_attr}")


print("\n" + "=" * 60)
print("场景3: 如果你需要分别处理两种边")
print("=" * 60)

# 从标准格式中分离出不同类型的边
sc_mask = data_standard.edge_type == 0
fc_mask = data_standard.edge_type == 1

edge_index_sc_extracted = data_standard.edge_index[:, sc_mask]
edge_weight_sc_extracted = data_standard.edge_weight[sc_mask]

edge_index_fc_extracted = data_standard.edge_index[:, fc_mask]
edge_weight_fc_extracted = data_standard.edge_weight[fc_mask]

print(f"✓ SC边: {edge_index_sc_extracted.shape[1]} 条")
print(f"  {edge_index_sc_extracted}")
print(f"✓ FC边: {edge_index_fc_extracted.shape[1]} 条")
print(f"  {edge_index_fc_extracted}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("自定义属性名的问题:")
print("  ❌ PyG内置函数无法自动识别")
print("  ❌ 需要手动处理每个自定义属性")
print("  ❌ 代码可维护性差")
print("\n标准属性名 + 边类型的优势:")
print("  ✓ 所有PyG内置函数自动工作")
print("  ✓ 可以用 edge_type 轻松筛选")
print("  ✓ 代码简洁易维护")
print("  ✓ 与PyG生态完美兼容")