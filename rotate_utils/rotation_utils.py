import torch
import typing
import utils
import transformers
import tqdm, math
import quant_utils
from rotate_utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
# from fast_hadamard_transform import hadamard_transform


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        print(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
        

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode='hadamard', device=None, seed=42):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device, seed)
    else:
        raise ValueError(f'Unknown mode {mode}')
    


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
        # torch.manual_seed(seed + i)
        
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


def rotate_mat_qkv(layer, Q):
    print("rotate mat qkv......")
    W_qkv = layer.attn.mat_qkv.weight.data
    dtype = W_qkv.dtype
    W_qkv = W_qkv.to(torch.float64)

    _, C = W_qkv.shape
    W_q = W_qkv[:C, :]
    W_k = W_qkv[C : 2*C, :]
    W_v = W_qkv[2*C : 3*C, :]

    W_q_rot = torch.matmul(W_q, Q)
    W_k_rot = torch.matmul(W_k, Q)
    W_v_rot = torch.matmul(W_v, Q)

    layer.attn.mat_qkv.weight.data = torch.cat([W_q_rot, W_k_rot, W_v_rot], dim=0).to(dtype)


def rotate_fc1(layer, Q):
    print("rotate fc1......")
    W_fc1 = layer.ffn.fc1.weight.data
    dtype = W_fc1.dtype
    W_fc1 = W_fc1.to(torch.float64)

    W_fc1_rot = torch.matmul(W_fc1, Q)
    layer.ffn.fc1.weight.data = W_fc1_rot.to(dtype)


def rotate_fc2(layer, Q):
    print("rotate fc2......")
    W_fc2 = layer.ffn.fc2.weight.data
    dtype = W_fc2.dtype
    W_fc2 = W_fc2.to(torch.float64)

    W_fc2_rot = torch.matmul(W_fc2, Q)
    layer.ffn.fc2.weight.data = W_fc2_rot.to(dtype)


def rotate_ada_lin(layer, Q):
    print("rotate ada_lin......")
    W = layer.ada_lin[1].weight.data
    B = layer.ada_lin[1].bias.data

    dtype = W.dtype
    W = W.to(torch.float64)
    B = B.to(torch.float64)

    _, C = W.shape
    # rotate weight
    W_gamma1 = W[:C, :]
    W_gamma2 = W[C:2*C, :]
    W_scale1 = W[2*C:3*C, :]
    W_scale2 = W[3*C:4*C, :]
    W_shift1 = W[4*C:5*C, :]
    W_shift2 = W[5*C:6*C, :]

    W_scale1_rot = torch.matmul(Q.T, W_scale1)
    W_scale2_rot = torch.matmul(Q.T, W_scale2)
    W_shift1_rot = torch.matmul(Q.T, W_shift1)
    W_shift2_rot = torch.matmul(Q.T, W_shift2)

    layer.ada_lin[1].weight.data = torch.cat([W_gamma1, W_gamma2, W_scale1,
                                              W_scale2, W_shift1_rot, W_shift2_rot], dim=0).to(dtype)
    
    # rotate bias
    bias_gamma1 = B[:C]
    bias_gamma2 = B[C:2*C]
    bias_scale1 = B[2*C:3*C]
    bias_scale2 = B[3*C:4*C]
    bias_shift1 = B[4*C:5*C]
    bias_shift2 = B[5*C:6*C]

    bias_scale1_rot = torch.matmul(bias_scale1, Q)
    bias_scale2_rot = torch.matmul(bias_scale2, Q)
    bias_shift1_rot = torch.matmul(bias_shift1, Q)
    bias_shift2_rot = torch.matmul(bias_shift2 , Q)

    layer.ada_lin[1].bias.data = torch.cat([bias_gamma1, bias_gamma2, bias_scale1,
                                            bias_scale2, bias_shift1_rot, bias_shift2_rot], dim=0).to(dtype)



def rotate_model(model, device, block_rotate):
    if block_rotate == False:
        '''rotate'''
        Q = get_orthogonal_matrix(model.C, mode='hadamard', device=device)
        # Q1 = get_orthogonal_matrix(int(model.mlp_ratio * model.C), mode='hadamard', device=device)

        layers = model.blocks
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")): 
            rotate_mat_qkv(layer, Q)
            rotate_fc1(layer, Q)
            # rotate_fc2(layer, Q1)
            # rotate_ada_lin(layer, Q)

    else:
        print("Block rotating...")
        '''block rotate'''
        total_size = model.C
        block_size = 128

        Q_block = block_random_hadamard_matrix(
            total_size=total_size,
            block_size=block_size,
            device=device,
            seed=42
        )

        layers = model.blocks
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")): 
            rotate_mat_qkv(layer, Q_block)
            rotate_fc1(layer, Q_block)




