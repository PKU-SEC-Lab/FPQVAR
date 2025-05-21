import sys
sys.path.append('/home/wrj/Q-VAR')

import torch
from torch import nn
import numpy as np
import torch
import typing
import transformers
import tqdm, math
import quant_utils
from rotate_utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
import pdb

# 原函数改造为分块版本
def block_random_hadamard_matrix(
    total_size=1920,
    block_size=128,
    device='cuda',
    seed=42,
    force_identity=False
):
    """
    生成分块对角随机Hadamard矩阵
    参数：
    total_size: 总矩阵尺寸（需能被block_size整除）
    block_size: 每个子块的大小
    seed: 随机种子（每个块使用seed+i作为独立种子）
    force_identity: 调试用，强制使用单位矩阵
    """
    # 尺寸校验
    assert total_size % block_size == 0, "尺寸不匹配"
    n_blocks = total_size // block_size
    
    # 生成分块矩阵
    blocks = []
    for i in range(n_blocks):
        # 每个块独立设置seed
        torch.manual_seed(seed + i)
        
        if force_identity:
            # 调试用单位矩阵
            block = torch.eye(block_size, dtype=torch.float64)
        else:
            # 生成随机对角矩阵
            block = random_hadamard_matrix(block_size, device, seed)
            
        blocks.append(block.to(device))
    
    # 构建分块对角矩阵
    return block_diag(blocks)

# 辅助函数：生成分块对角矩阵
def block_diag(blocks):
    """
    将多个块组合成分块对角矩阵
    输入：blocks - 张量列表，每个形状为 [b, b]
    输出：分块对角矩阵 [n, n]，n = sum(b)
    """
    matrix = torch.zeros(
        (len(blocks)*blocks[0].shape[0], 
         len(blocks)*blocks[0].shape[1]),
        device=blocks[0].device,
        dtype=blocks[0].dtype
    )
    
    for i, blk in enumerate(blocks):
        row = i * blk.shape[0]
        col = i * blk.shape[1]
        matrix[row:row+blk.shape[0], col:col+blk.shape[1]] = blk
    
    return matrix

# 验证示例
if __name__ == "__main__":
    # 参数设置
    total_size = 1920
    block_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成分块矩阵
    H_block = block_random_hadamard_matrix(
        total_size=total_size,
        block_size=block_size,
        device=device,
        seed=42
    )
    
    # 验证正交性（分块正交矩阵的直和仍正交）
    identity = torch.matmul(H_block, H_block.T)
    error = torch.norm(identity - torch.eye(total_size, device=device))
    print(f"正交性误差: {error.item():.2e}")  # 应接近0

    # 计算效率测试
    x = torch.randn(100, 1920, device=device)
    
    # pdb.set_trace()

    # 完整矩阵乘法（对比用）
    # H_full = random_hadamard_matrix(1920, device, seed=42)
    # y_full = x @ H_full
    
    # 分块乘法（等效实现）
    x_blocks = x.chunk(15, dim=1)
    y_blocks = [x_blk @ H_block[i*128:(i+1)*128, i*128:(i+1)*128] 
                for i, x_blk in enumerate(x_blocks)]
    y_block = torch.cat(y_blocks, dim=1)
    
    # 验证计算结果等效性
    # print("最大差值:", torch.max(torch.abs(y_full - y_block)).item())