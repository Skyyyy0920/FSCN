"""
快速测试：验证 TransformerEncoder 维度修复

运行此脚本以验证修复是否成功
"""

import torch
from model_xlearner import TransformerEncoder

print("="*60)
print("测试 TransformerEncoder 维度修复")
print("="*60)

# 测试不同的节点数（包括不能被4整除的）
test_cases = [
    100,   # 不能被4整除
    200,   # 能被4整除
    268,   # ABCD 可能的节点数
    300,   # 不能被4整除
]

success_count = 0
failed_count = 0

for num_nodes in test_cases:
    print(f"\n测试 num_nodes={num_nodes}:")
    
    try:
        # 创建编码器
        encoder = TransformerEncoder(
            input_dim=num_nodes,
            d_model=128,  # 能被 4 整除
            nhead=4,
            num_layers=2
        )
        
        # 测试前向传播
        batch_size = 2
        x = torch.randn(batch_size, num_nodes, num_nodes)
        out = encoder(x)
        
        # 验证输出形状
        expected_shape = (batch_size, num_nodes, 128)
        assert out.shape == expected_shape, f"形状不匹配: {out.shape} vs {expected_shape}"
        
        print(f"  ✓ 成功! 输入: [{batch_size}, {num_nodes}, {num_nodes}]")
        print(f"  ✓ 输出: {out.shape}")
        success_count += 1
        
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        failed_count += 1
    except Exception as e:
        print(f"  ✗ 错误: {type(e).__name__}: {e}")
        failed_count += 1

print("\n" + "="*60)
print("测试总结")
print("="*60)
print(f"总计: {len(test_cases)} 个测试")
print(f"成功: {success_count} ✓")
print(f"失败: {failed_count} ✗")

if failed_count == 0:
    print("\n🎉 所有测试通过！TransformerEncoder 修复成功！")
else:
    print(f"\n⚠️  {failed_count} 个测试失败，需要进一步检查")

print("="*60)

