import torch
import numpy as np
import pdb
from functools import partial

@torch.no_grad()
def quantize_activation_per_tensor_sym(t, n_bits):
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.div_(q_max).clamp_(min=1e-5)
    t_hat = torch.clamp(torch.round(t / scales), q_min, q_max) * scales
    return t_hat, scales


@torch.no_grad()
def quantize_activation_per_token_sym(t, n_bits):
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.div_(q_max).clamp_(min=1e-5)
    t_hat = torch.clamp(torch.round(t / scales), q_min, q_max) * scales
    return t_hat

# per group linear quantize
@torch.no_grad()
def quantize_activation_per_group_sym(t, n_bits, group_size):
    t_shape = t.shape
    t_flat = t.view(-1, group_size)
    scales = t_flat.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.div_(q_max).clamp_(min=1e-5)
    # pdb.set_trace()
    t_flat = torch.clamp(torch.round(t_flat / scales), q_min, q_max) * scales
    t_hat = t_flat.view(t_shape)
    return t_hat


def log2_quant_per_token_sym(x, n_bits):
    zero_mask = (x == 0)
    sign = torch.sign(x)  # 避免零值符号丢失
    abs_x = torch.abs(x)
    log2_x = torch.log2(torch.where(zero_mask, 1.0, abs_x)) 
    
    # pdb.set_trace()

    log2_x_absmax = log2_x.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)

    scales = log2_x_absmax.div_(q_max).clamp(min=1e-5)
    log2_dequant = torch.clamp(torch.round(log2_x / scales), q_min, q_max).mul_(scales)

    x_dequant = torch.pow(2, log2_dequant) * sign
    x_dequant[zero_mask] = 0.0  

    return x_dequant


def log2_quant_per_group_sym(x, n_bits, group_size):
    x_shape = x.shape
    x = x.view(-1, group_size)

    zero_mask = (x == 0)
    sign = torch.sign(x)  # 避免零值符号丢失
    abs_x = torch.abs(x)
    log2_x = torch.log2(torch.where(zero_mask, 1.0, abs_x)) 
    
    # pdb.set_trace()

    log2_x_absmax = log2_x.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)

    scales = log2_x_absmax.div_(q_max).clamp(min=1e-5)
    log2_dequant = torch.clamp(torch.round(log2_x / scales), q_min, q_max).mul_(scales)

    x_dequant = torch.pow(2, log2_dequant) * sign
    x_dequant[zero_mask] = 0.0  
    x_dequant = x_dequant.view(x_shape)
    return x_dequant



def log2_quant_per_token_asym(x, n_bits):
    zero_mask = (x == 0)
    sign = torch.sign(x)  # 避免零值符号丢失
    abs_x = torch.abs(x)
    log2_x = torch.log2(torch.where(zero_mask, 1.0, abs_x)) 
    
    # pdb.set_trace()

    log2_x_max = log2_x.max(dim=-1, keepdim=True)[0]
    log2_x_min = log2_x.min(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)

    scales = (log2_x_max - log2_x_min).div_(q_max - q_min).clamp(min=1e-5)
    zero_point = torch.round(q_min - log2_x_min / scales)
    log2_dequant = torch.clamp(torch.round(log2_x / scales) + zero_point, q_min, q_max).sub_(zero_point).mul_(scales)

    x_dequant = torch.pow(2, log2_dequant) * sign
    x_dequant[zero_mask] = 0.0  

    return x_dequant


def log2_quant_per_group_asym(x, n_bits, group_size):
    x_shape = x.shape
    x = x.view(-1, group_size)

    zero_mask = (x == 0)
    sign = torch.sign(x)  # 避免零值符号丢失
    abs_x = torch.abs(x)
    log2_x = torch.log2(torch.where(zero_mask, 1.0, abs_x)) 
    
    # pdb.set_trace()

    log2_x_max = log2_x.max(dim=-1, keepdim=True)[0]
    log2_x_min = log2_x.min(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)

    scales = (log2_x_max - log2_x_min).div_(q_max - q_min).clamp(min=1e-5)
    zero_point = torch.round(q_min - log2_x_min / scales)
    log2_dequant = torch.clamp(torch.round(log2_x / scales) + zero_point, q_min, q_max).sub_(zero_point).mul_(scales)

    x_dequant = torch.pow(2, log2_dequant) * sign
    x_dequant[zero_mask] = 0.0  

    x_dequant = x_dequant.view(x_shape)

    return x_dequant


def du_quantizer_per_token(x, n_bits, c=1.67, m=5, K=3.0):
    # pdb.set_trace()
    std = torch.std(x, dim=-1, keepdim=True)
    x = x / std

    s_1 = c / m
    n = 2 ** (n_bits - 1) - 1 - m
    s_2 = (K - c) / n

    x = torch.clamp(x, -K, K)
    abs_x = torch.abs(x)
    
    # 内部量化区域
    inner_mask = abs_x <= c
    quantized_inner = torch.clamp(torch.round(x[inner_mask] / s_1), -m, m) * s_1
    
    # 外部量化区域
    outer_mask = ~inner_mask
    x_outer = x[outer_mask]
    sign = torch.sign(x_outer)
    abs_outer = torch.abs(x_outer)
    quantized_outer = sign * (c + torch.clamp(torch.round((abs_outer - c) / s_2), 0, n) * s_2)
    
    # 合并结果
    output = torch.zeros_like(x)
    output[inner_mask] = quantized_inner
    output[outer_mask] = quantized_outer

    output = output * std

    return output


def du_quantizer_per_group(x, n_bits=4, group_size=128, c=1.61, m=5, K=3.0):
    # pdb.set_trace()

    x_shape = x.shape
    x = x.view(-1, group_size)
    
    # v1
    # std = torch.std(x, dim=-1, keepdim=True)
    # x = x / std
    
    # v2
    scale = x.abs().max(dim=-1, keepdim=True)[0] / 3.0
    x = x / scale

    s_1 = c / m
    n = 2 ** (n_bits - 1) - 1 - m
    s_2 = (K - c) / n

    x = torch.clamp(x, -K, K)
    abs_x = torch.abs(x)
    
    # 内部量化区域
    inner_mask = abs_x <= c
    quantized_inner = torch.clamp(torch.round(x[inner_mask] / s_1), -m, m) * s_1
    
    # 外部量化区域
    outer_mask = ~inner_mask
    x_outer = x[outer_mask]
    sign = torch.sign(x_outer)
    abs_outer = torch.abs(x_outer)
    quantized_outer = sign * (c + torch.clamp(torch.round((abs_outer - c) / s_2), 0, n) * s_2)
    
    # 合并结果
    output = torch.zeros_like(x)
    output[inner_mask] = quantized_inner
    output[outer_mask] = quantized_outer

    # output = output * std
    output = output * scale
    
    output = output.view(x_shape)
    return output



def compute_quant_error(x_fp, x_quant):
    quant_error = torch.mean((x_fp - x_quant) ** 2).item()
    # quant_error = torch.mean((x_fp - x_quant) ** 2 / x_fp ** 2).item()
    return quant_error


def quantize_to_nearest_grid(x: torch.Tensor, quant_grid: torch.Tensor):
    """
    将输入张量 `x` 中的每个元素映射到 `quant_grid` 中的最近邻值。
    
    Args:
        x (torch.Tensor): 输入张量。
        quant_grid (torch.Tensor): 已排序的量化值网格。
    
    Returns:
        torch.Tensor: 量化后的张量。
    """
    # 确保 quant_grid 是升序排列
    assert torch.all(torch.eq(quant_grid, torch.sort(quant_grid).values)), "quant_grid must be sorted."
    
    # 计算所有绝对距离
    distances = torch.abs(x.unsqueeze(-1) - quant_grid)  # 形状: (..., len(quant_grid))
    
    # 找到每个元素的最小距离索引
    min_indices = torch.argmin(distances, dim=-1)         # 形状: (...)
    
    # 返回对应的 quant_grid 值
    return quant_grid[min_indices]


def flint_quant(x):
    '''4-bit 量化'''
    quant_grid = torch.tensor([-10.0000,  -5.0000,  -3.7500,  -2.5000,  -1.8750,  -1.2500,  -0.6250,
        0.0000,   0.0000,   0.6250,   1.2500,   1.8750,   2.5000,   3.7500,
        5.0000,  10.0000]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output


def fp_quant_e1(x):
    '''4-bit 量化'''
    quant_grid = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output


def fp_quant_e2(x):
    '''4-bit 量化'''
    quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output


def fp_quant_e2_per_tensor(x):
    '''4-bit 量化'''
    quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max() / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output, scale


def fp_quant_e3(x):
    '''4-bit 量化''' 
    quant_grid = torch.tensor([-64, -32, -16, -8, -4, -2, -1, 0.0, 1, 2, 4, 8, 16, 32, 64]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    # pdb.set_trace()
    output = quantized_x * scale

    return output


def fp_quant_e1_per_group(x, group_size=128):
    # pdb.set_trace()
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    '''4-bit 量化'''
    quant_grid = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale
    output = output.reshape(x_shape)

    return output


def fp_quant_e2_per_group(x, group_size=128):
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    '''4-bit 量化'''
    quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
    
    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale
    output = output.reshape(x_shape)
    return output


def fp_quant_e3_per_group(x, group_size=128):
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    '''4-bit 量化'''
    # quant_grid = torch.tensor([-64, -32, -16, -8, -4, -2, -1, 0.0, 1, 2, 4, 8, 16, 32, 64]).cuda()
    quant_grid = torch.tensor([-16, -8, -4, -2, -1, -0.5, -0.25, 0.0, 0.25, 0.5, 1, 2, 4, 8, 16]).cuda()

    # x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    # pdb.set_trace()
    output = quantized_x * scale
    output = output.reshape(x_shape)
    return output


class FPQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits=4, group_size=128, format=None, clipping_strength=1.0):
        assert n_bits == 4
        '''4-bit 量化'''
        if format == "e1m2":
            quant_grid = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]).cuda()
        elif format == "e2m1":
            quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
        elif format == "e3m0":
            quant_grid = torch.tensor([-16, -8, -4, -2, -1, -0.5, -0.25, 0.0, 0.25, 0.5, 1, 2, 4, 8, 16]).cuda()
        else:
            raise ValueError("Unsupported format type")

        clip_value = clipping_strength * x.abs().max()
        x = torch.clamp(x, -clip_value, clip_value)
        x_shape = x.shape
        x = x.reshape(-1, group_size)
        scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
        x  = x / scale
        quantized_x = quantize_to_nearest_grid(x, quant_grid)
        output = quantized_x * scale
        output = output.reshape(x_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        grad_clip = None
        return grad_input, grad_clip, None, None,None,None
    

def compute_quant_error(x_fp, x_quant):

    quant_error = torch.mean((x_fp - x_quant)**2)

    return quant_error



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    block_idx = 9
    w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True).cuda()
    

    # INT4 quantization
    w_int_quant, int_scale = quantize_activation_per_tensor_sym(w_fp, 4)
    int4_grid = [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    int4_quant_levels = [(x * int_scale).cpu() for x in int4_grid]

    # FP quantization
    w_quant, fp_scale = fp_quant_e2_per_tensor(w_fp)
    fp4_e2m1_grid = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    fp_quant_levels =  [(x * fp_scale).cpu() for x in fp4_e2m1_grid]


    # plot
    w_fp_np = w_fp.cpu().detach().numpy().flatten()  # 转换为 numpy 并展平

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})

    # 绘制直方图（分布）
    # plt.figure(figsize=(12, 6))
    plt.hist(w_fp_np, bins=100, alpha=0.7, color='blue')

    # 添加 INT4 量化级别的竖线（需指定 label）
    for level in int4_quant_levels:
        plt.axvline(x=level, color='red', linestyle='--', linewidth=1.0, alpha=1.0, label='INT4 Quant' if level == int4_quant_levels[0] else "")

    # 添加 FP4 量化级别的竖线（需指定 label）
    for level in fp_quant_levels:
        plt.axvline(x=level, color='green', linestyle=':', linewidth=1.5, alpha=1.0, label='FP4-E2M1 Quant' if level == fp_quant_levels[0] else "")

    # 添加图例和标题
    plt.legend()
    plt.xlabel('Weight Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # # 调整坐标轴范围（可选）
    # plt.xlim(min(w_fp_np.min(), min(int4_quant_levels + fp_quant_levels)) - 0.5,
    #         max(w_fp_np.max(), max(int4_quant_levels + fp_quant_levels)) + 0.5)

    plt.tight_layout()
    plt.savefig("weight_distribution_for_motivation.png")
    pdb.set_trace()