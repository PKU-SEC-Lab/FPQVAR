import sys
sys.path.append("/home/rjwei/Q-VAR")
import torch
import numpy as np
import pdb
from functools import partial
from rotate_utils import rotation_utils
import argparse
import quant_cuda

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
    

class FPQuant_e1m2_neg_e2m1_pos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits=4, group_size=128, clipping_strength=1.0):
        assert n_bits == 4
        '''4-bit 量化'''
        quant_grid_e1m2_neg = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0]).cuda()
        quant_grid_e2m1_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()

        clip_value = clipping_strength * x.abs().max()
        x = torch.clamp(x, -clip_value, clip_value)
        x_shape = x.shape
        x = x.reshape(-1, group_size)
        
        # 分离正负部分
        x_neg = torch.where(x <= 0, x, torch.zeros_like(x))
        x_pos = torch.where(x > 0, x, torch.zeros_like(x))

        # 分别计算正负部分的scale
        scale_neg = x_neg.abs().max(dim=-1, keepdim=True)[0] / quant_grid_e1m2_neg.abs().max()
        scale_pos = x_pos.abs().max(dim=-1, keepdim=True)[0] / quant_grid_e2m1_pos.abs().max()

        # 分别归一化
        x_neg_normalized = x_neg / scale_neg
        x_pos_normalized = x_pos / scale_pos

        # 分别量化
        quantized_neg = quantize_to_nearest_grid(x_neg_normalized, quant_grid_e1m2_neg)
        quantized_pos = quantize_to_nearest_grid(x_pos_normalized, quant_grid_e2m1_pos)

        # 合并结果
        quantized_x = quantized_neg + quantized_pos
        output = quantized_x * torch.where(x <= 0, scale_neg, scale_pos)
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
    

class FPQuant_e1m2_neg_e3m0_pos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits=4, group_size=128, clipping_strength=1.0):
        assert n_bits == 4
        '''4-bit 量化'''
        quant_grid_e1m2_neg = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0]).cuda()
        quant_grid_e3m0_pos = torch.tensor([0.0, 0.25, 0.5, 1, 2, 4, 8, 16]).cuda()

        clip_value = clipping_strength * x.abs().max()
        x = torch.clamp(x, -clip_value, clip_value)
        x_shape = x.shape
        x = x.reshape(-1, group_size)
        
        # 分离正负部分
        x_neg = torch.where(x <= 0, x, torch.zeros_like(x))
        x_pos = torch.where(x > 0, x, torch.zeros_like(x))

        # 分别计算正负部分的scale
        scale_neg = x_neg.abs().max(dim=-1, keepdim=True)[0] / quant_grid_e1m2_neg.abs().max()
        scale_pos = x_pos.abs().max(dim=-1, keepdim=True)[0] / quant_grid_e3m0_pos.abs().max()

        # 分别归一化
        x_neg_normalized = x_neg / scale_neg
        x_pos_normalized = x_pos / scale_pos

        # 分别量化
        quantized_neg = quantize_to_nearest_grid(x_neg_normalized, quant_grid_e1m2_neg)
        quantized_pos = quantize_to_nearest_grid(x_pos_normalized, quant_grid_e3m0_pos)

        # 合并结果
        quantized_x = quantized_neg + quantized_pos
        output = quantized_x * torch.where(x <= 0, scale_neg, scale_pos)
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
    

# 下面定义FP6量化
fp6_e2m3_grid = torch.tensor([
                -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0,
                -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0,
                -1.875, -1.75, -1.625, -1.5, -1.375, -1.25, -1.125, -1.0,
                -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0,
                0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 
                1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
                4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5
                ])

fp6_e3m2_grid = torch.tensor([
                -28.0, -24.0, -20.0, -16.0, 
                -14.0, -12.0, -10.0, -8.0,
                -7.0, -6.0, -5.0, -4.0, 
                -3.5, -3.0, -2.5, -2.0,
                -1.75, -1.5, -1.25, -1.0, 
                -0.875, -0.75, -0.625, -0.5,
                -0.4375, -0.375, -0.3125, -0.25, 
                -0.1875, -0.125, -0.0625, 0,
                0, 0.0625, 0.125, 0.1875,
                0.25, 0.3125, 0.375, 0.4375,
                0.5, 0.625, 0.75, 0.875,
                1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5,
                4.0, 5.0, 6.0, 7.0,
                8.0, 10.0, 12.0, 14.0,
                16.0, 20.0, 24.0, 28.0
                ])

int_neg_grid = torch.tensor([
                -32.0, -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0,
                -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, 
                -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, 
                -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,
                ])

e2m3_pos_grid = torch.tensor([
                0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 
                1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
                2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
                4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5
                ])


e3m2_pos_grid = torch.tensor([
                0, 0.0625, 0.125, 0.1875,
                0.25, 0.3125, 0.375, 0.4375,
                0.5, 0.625, 0.75, 0.875,
                1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5,
                4.0, 5.0, 6.0, 7.0,
                8.0, 10.0, 12.0, 14.0,
                16.0, 20.0, 24.0, 28.0
                ])



def fp6_quant_e2m3_per_token_cuda(x, n_bits):
    assert n_bits == 6
    quant_grid = fp6_e2m3_grid.to(x.device)
    # x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quant_array = x.view(-1)
    quant_array  = quant_array.to(torch.float32)
    # round to nearest fp
    quant_array, _ = quant_cuda.quant(quant_array, quant_grid)
    quant_array = quant_array.view(x_shape)
    output = quant_array * scale
    output = output.to(torch.float16)
    return output


def fp6_quant_e3m2_per_token_cuda(x, n_bits):
    assert n_bits == 6
    quant_grid = fp6_e3m2_grid.to(x.device)
    # x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quant_array = x.view(-1)
    quant_array  = quant_array.to(torch.float32)
    # round to nearest fp
    quant_array, _ = quant_cuda.quant(quant_array, quant_grid)
    quant_array = quant_array.view(x_shape)
    output = quant_array * scale
    output = output.to(torch.float16)
    return output


def fp6_quant_e2m3_per_group_cuda(x, n_bits, group_size=128):
    assert n_bits == 6
    quant_grid = fp6_e2m3_grid.to(x.device)
    # x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    x_shape_1 = x.shape
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quant_array = x.view(-1)
    quant_array  = quant_array.to(torch.float32)
    # round to nearest fp
    quant_array, _ = quant_cuda.quant(quant_array, quant_grid)
    quant_array = quant_array.view(x_shape_1)
    output = quant_array * scale
    output = output.view(x_shape)
    output = output.to(torch.float16)
    return output


def fp6_quant_e3m2_per_group_cuda(x, n_bits, group_size=128):
    assert n_bits == 6
    quant_grid = fp6_e3m2_grid.to(x.device)
    # x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    x_shape_1 = x.shape
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quant_array = x.view(-1)
    quant_array  = quant_array.to(torch.float32)
    # round to nearest fp
    quant_array, _ = quant_cuda.quant(quant_array, quant_grid)
    quant_array = quant_array.view(x_shape_1)
    output = quant_array * scale
    output = output.view(x_shape)
    output = output.to(torch.float16)
    return output


def fp6_quant_int_neg_e2m3_pos_per_group_cuda(x, n_bits, group_size=128):
    assert n_bits == 6
    quant_grid_int_neg = int_neg_grid.to(x.device)
    quant_grid_e2m3_pos = e2m3_pos_grid.to(x.device)

    x_shape = x.shape
    x = x.reshape(-1, group_size)
    x_shape_1 = x.shape
    
    # 分离正负部分
    x_neg = torch.where(x <= 0, x, torch.zeros_like(x))
    x_pos = torch.where(x > 0, x, torch.zeros_like(x))

    # 分别计算正负部分的scale
    scale_neg = x_neg.abs().max(dim=-1, keepdim=True)[0] / quant_grid_int_neg.abs().max()
    scale_pos = x_pos.abs().max(dim=-1, keepdim=True)[0] / quant_grid_e2m3_pos.abs().max()

    # 分别归一化
    x_neg_normalized = x_neg / scale_neg
    x_pos_normalized = x_pos / scale_pos

    x_neg_normalized = x_neg_normalized.view(-1).to(torch.float32)
    x_pos_normalized = x_pos_normalized.view(-1).to(torch.float32)

    # 分别量化
    quantized_neg, _ = quant_cuda.quant(x_neg_normalized, quant_grid_int_neg)
    quantized_pos, _ = quant_cuda.quant(x_pos_normalized, quant_grid_e2m3_pos)

    quantized_neg = quantized_neg.view(x_shape_1)
    quantized_pos = quantized_pos.view(x_shape_1)

    # 合并结果
    output = quantized_neg * scale_neg + quantized_pos * scale_pos
    output = output.view(x_shape).to(x.dtype)
    return output


def fp6_quant_int_neg_e3m2_pos_per_group_cuda(x, n_bits, group_size=128):
    assert n_bits == 6
    assert n_bits == 6
    quant_grid_int_neg = int_neg_grid.to(x.device)
    quant_grid_e3m2_pos = e3m2_pos_grid.to(x.device)

    x_shape = x.shape
    x = x.reshape(-1, group_size)
    x_shape_1 = x.shape
    
    # 分离正负部分
    x_neg = torch.where(x <= 0, x, torch.zeros_like(x))
    x_pos = torch.where(x > 0, x, torch.zeros_like(x))

    # 分别计算正负部分的scale
    scale_neg = x_neg.abs().max(dim=-1, keepdim=True)[0] / quant_grid_int_neg.abs().max()
    scale_pos = x_pos.abs().max(dim=-1, keepdim=True)[0] / quant_grid_e3m2_pos.abs().max()

    # 分别归一化
    x_neg_normalized = x_neg / scale_neg
    x_pos_normalized = x_pos / scale_pos

    x_neg_normalized = x_neg_normalized.view(-1).to(torch.float32)
    x_pos_normalized = x_pos_normalized.view(-1).to(torch.float32)

    # 分别量化
    quantized_neg, _ = quant_cuda.quant(x_neg_normalized, quant_grid_int_neg)
    quantized_pos, _ = quant_cuda.quant(x_pos_normalized, quant_grid_e3m2_pos)

    quantized_neg = quantized_neg.view(x_shape_1)
    quantized_pos = quantized_pos.view(x_shape_1)

    # 合并结果
    output = quantized_neg * scale_neg + quantized_pos * scale_pos
    output = output.view(x_shape).to(x.dtype)
    return output


def compute_quant_error(x_fp, x_quant):

    quant_error = torch.mean((x_fp - x_quant)**2)

    return quant_error


def compute_quant_error_act_e2m3(x, w, Q):
    # pdb.set_trace()
    fp_result = torch.matmul(x, w.T)

    x_2 = torch.matmul(x, Q)
    x_2_quant = fp6_quant_e2m3_per_group_cuda(x_2, 6)

    w_2 = torch.matmul(w, Q)
    w_2_quant = fp6_quant_e2m3_per_token_cuda(w_2, 6)

    quant_result = torch.matmul(x_2_quant, w_2_quant.T)
    quant_error = torch.mean(((fp_result - quant_result)**2))
    pdb.set_trace()
    return quant_error


def compute_quant_error_act_e3m2(x, w, Q):
    # pdb.set_trace()
    fp_result = torch.matmul(x, w.T)

    x_2 = torch.matmul(x, Q)
    x_2_quant = fp6_quant_e3m2_per_group_cuda(x_2, 6)

    w_2 = torch.matmul(w, Q)
    w_2_quant = fp6_quant_e2m3_per_token_cuda(w_2, 6)

    quant_result = torch.matmul(x_2_quant, w_2_quant.T)
    quant_error = torch.mean(((fp_result - quant_result)**2))

    return quant_error


def compute_quant_error_act_int(x, w, Q):
    # pdb.set_trace()
    fp_result = torch.matmul(x, w.T)

    x_2 = torch.matmul(x, Q)
    x_2_quant = quantize_activation_per_group_sym(x_2, 6, 128)

    w_2 = torch.matmul(w, Q)
    w_2_quant = quantize_activation_per_token_sym(w_2, 6)

    quant_result = torch.matmul(x_2_quant, w_2_quant.T)
    quant_error = torch.mean(((fp_result - quant_result)**2))

    return quant_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    # parser.add_argument("--block_rotate", action="store_true", default=False)
    args = parser.parse_args()

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    # # fc2
    # print("**********fc2**************")
    # mse = {
    #     'int': [],      # 存储 int mse 的列表
    #     'e2m3': [],     # 存储 e1m2 mse 的列表
    #     'e3m2': [],     # 存储 e2m1 mse 的列表
    #     'int_neg_e2m3_pos': [],
    #     'int_neg_e3m2_pos': [],
    # }


    # for block_idx in range(30):
    #     print(block_idx)
    #     activation = []
    #     for label_idx in range(100):
    #         for step_idx in range(10):
    #             file_name = f"/home/rjwei/Data_raid/Q-VAR/cali_data/fc2_input/label{label_idx}_block{block_idx}_step{step_idx}_fc2_input.pt"
    #             x_fp = torch.load(file_name)    
    #             activation.append(x_fp)

    #     # pdb.set_trace()

    # # 使用两种数据格式对GeLU的输出进行量化
    # loss_int = 0.0
    # loss_e2m3 = 0.0
    # loss_e3m2 = 0.0
    # loss_int_neg_e2m3_pos = 0.0
    # loss_int_neg_e3m2_pos = 0.0

    # for j in range(len(activation)):
    #     x_fp = activation[j]
    #     x_quant = quantize_activation_per_group_sym(x_fp, 6, 128)
    #     loss_int += compute_quant_error(x_fp, x_quant) 

    #     x_quant_e2m3 = fp6_quant_e2m3_per_group_cuda(x_fp, 6)
    #     loss_e2m3 += compute_quant_error(x_fp, x_quant_e2m3) 

    #     x_quant_e3m2 = fp6_quant_e3m2_per_group_cuda(x_fp, 6)
    #     loss_e3m2 += compute_quant_error(x_fp, x_quant_e3m2) 

    #     x_quant_1 = fp6_quant_int_neg_e2m3_pos_per_group_cuda(x_fp, 6)
    #     loss_int_neg_e2m3_pos += compute_quant_error(x_fp, x_quant_1) 

    #     x_quant_2 = fp6_quant_int_neg_e3m2_pos_per_group_cuda(x_fp, 6)
    #     loss_int_neg_e3m2_pos += compute_quant_error(x_fp, x_quant_2) 


    # loss_int = loss_int / len(activation)
    # print(f"int, Loss: {loss_int:.6f}")
    # mse['int'].append(loss_int.item())


    # loss_e2m3 = loss_e2m3 / len(activation)
    # print(f"e2m3, Loss: {loss_e2m3:.6f}")
    # mse['e2m3'].append(loss_e2m3.item())


    # loss_e3m2 = loss_e3m2 / len(activation)
    # print(f"e3m2, Loss: {loss_e3m2:.6f}")
    # mse['e3m2'].append(loss_e3m2.item())


    # loss_int_neg_e2m3_pos = loss_int_neg_e2m3_pos / len(activation)
    # print(f"int_neg_e2m3_pos, Loss: {loss_int_neg_e2m3_pos:.6f}")
    # mse['int_neg_e2m3_pos'].append(loss_int_neg_e2m3_pos.item())


    # loss_int_neg_e3m2_pos = loss_int_neg_e3m2_pos / len(activation)
    # print(f"int_neg_e3m2_pos, Loss: {loss_int_neg_e3m2_pos:.6f}")
    # mse['int_neg_e3m2_pos'].append(loss_int_neg_e3m2_pos.item())

    # torch.save(mse, "activation_mse_fp6.fc2.pt")
    # pdb.set_trace()

    # kv 
    activation = []
    block_idx = 9

    for step_idx in range(10):
        file_name = f"/home/rjwei/Data_raid/Q-VAR/activation_distribution_class_980/k/block{block_idx}_step{step_idx}_k.npy"
        x_fp = np.load(file_name)
        x_fp = torch.tensor(x_fp, device=device)    
        activation.append(x_fp)


    # for step_idx in range(10):
    #     file_name = f"/home/rjwei/Data_raid/Q-VAR/activation_distribution_class_980/v/block{block_idx}_step{step_idx}_v.npy"
    #     x_fp = np.load(file_name)
    #     x_fp = torch.tensor(x_fp, device=device)    
    #     activation.append(x_fp)



    loss_e2m3 ,  loss_e3m2 = 0.0, 0.0
    for j in range(len(activation)):

        x_quant_e2m3 = fp6_quant_e2m3_per_token_cuda(x_fp, 6)
        loss_e2m3 += compute_quant_error(x_fp, x_quant_e2m3) 

        x_quant_e3m2 = fp6_quant_e3m2_per_token_cuda(x_fp, 6)
        loss_e3m2 += compute_quant_error(x_fp, x_quant_e3m2) 


    # pdb.set_trace()

    loss_e2m3 = loss_e2m3 / len(activation)
    print(f"e2m3, Loss: {loss_e2m3:.8f}")

    loss_e3m2 = loss_e3m2 / len(activation)
    print(f"e3m2, Loss: {loss_e3m2:.8f}")
        

    pdb.set_trace()
