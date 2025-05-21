import torch
from torch import nn
from functools import partial

import pdb
from models_fp_quant_rotate.basic_var import FFN, SelfAttention, AdaLNSelfAttn
import quant_cuda

@torch.no_grad()
def quantize_weight_per_channel_sym(w, n_bits):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().clamp_(q_min, q_max).mul_(scales) # 实际上是[-127, 127]
    return w


@torch.no_grad()
def quantize_weight_per_tensor_sym(w, n_bits):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().clamp_(q_min, q_max).mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_group_sym(w, n_bits, group_size):
    # w: (out_features, in_features)
    w_shape = w.shape
    w_flat = w.view(-1, group_size)
    scales = w_flat.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.clamp_(min=1e-5).div_(q_max)
    w_flat.div_(scales).round_().clamp_(q_min, q_max).mul_(scales)
    w_hat = w_flat.view(w_shape)
    return w_hat


@torch.no_grad()
def quantize_activation_per_token_sym(t, n_bits):
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(q_min, q_max).mul_(scales)
    return t


# per token量化好像不需要reshape
@torch.no_grad()
def quantize_activation_per_token_asymmetric(t, n_bits):
    # Compute the max and min values for non-symmetric quantization
    t_min = t.min(dim=-1, keepdim=True)[0]
    t_max = t.max(dim=-1, keepdim=True)[0]

    # Compute scale and zero point for non-symmetric quantization
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales = (t_max - t_min).clamp(min=1e-5).div_(q_max - q_min)
    zero_point = torch.round(q_min - t_min / scales)

    # Apply scaling and zero-point shift for non-symmetric quantization
    t_hat = torch.clamp(torch.round(t / scales) + zero_point, q_min, q_max)
    t_hat.sub_(zero_point).mul_(scales) 

    # 显式释放中间变量（可选，适用于极端内存限制场景）
    del t, t_min, t_max, scales, zero_point
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return t_hat


@torch.no_grad()
def quantize_activation_per_group_sym(t, n_bits, group_size):
    t_shape = t.shape
    t_flat = t.view(-1, group_size)
    scales = t_flat.abs().max(dim=-1, keepdim=True)[0]
    q_min = - 2 ** (n_bits - 1)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t_flat.div_(scales).round_().clamp_(q_min, q_max).mul_(scales)
    t_hat = t_flat.view(t_shape)
    return t_hat


@torch.no_grad()
def quantize_activation_per_group_asymmetric(t, n_bits, group_size):
    t_shape = t.shape
    t_flat = t.view(-1, group_size)

    # Compute the max and min values for non-symmetric quantization
    t_min = t_flat.min(dim=-1, keepdim=True)[0]
    t_max = t_flat.max(dim=-1, keepdim=True)[0]

    # Compute scale and zero point for non-symmetric quantization
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales = (t_max - t_min).clamp(min=1e-5).div_(q_max - q_min)
    zero_point = torch.round(q_min - t_min / scales)

    # Apply scaling and zero-point shift for non-symmetric quantization
    t_hat = torch.clamp(torch.round(t_flat / scales) + zero_point, q_min, q_max)
    t_hat.sub_(zero_point).mul_(scales) 
    t_hat = t_hat.view(t_shape)

    # 显式释放中间变量（可选，适用于极端内存限制场景）
    del t_flat, t_min, t_max, scales, zero_point
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return t_hat



@torch.no_grad()
def quantize_activation_per_tensor_sym(t, n_bits):
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().clamp_(q_min, q_max).mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_asymmetric(t, n_bits):
    # Compute the max and min values for non-symmetric quantization
    t_min = t.min()
    t_max = t.max()

    # Compute scale and zero point for non-symmetric quantization
    q_max = 2 ** (n_bits - 1) - 1
    q_min = - 2 ** (n_bits - 1)
    scales = (t_max - t_min).clamp(min=1e-5).div_(q_max - q_min)
    zero_point = torch.round(q_min - t_min / scales)

    # Apply scaling and zero-point shift for non-symmetric quantization
    t_hat = torch.clamp(torch.round(t / scales) + zero_point, q_min, q_max)
    t_hat.sub_(zero_point).mul_(scales) 

    # 显式释放中间变量（可选，适用于极端内存限制场景）
    del t, t_min, t_max, scales, zero_point
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return t_hat


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
    # assert torch.all(torch.eq(quant_grid, torch.sort(quant_grid).values)), "quant_grid must be sorted."
    
    # 计算所有绝对距离
    distances = torch.abs(x.unsqueeze(-1) - quant_grid)  # 形状: (..., len(quant_grid))
    
    # 找到每个元素的最小距离索引
    min_indices = torch.argmin(distances, dim=-1)         # 形状: (...)

    # 返回对应的 quant_grid 值
    return quant_grid[min_indices]


def fp_quant_e3_per_token(x, n_bits):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-16, -8, -4, -2, -1, -0.5, -0.25, 0.0, 0.25, 0.5, 1, 2, 4, 8, 16]).cuda()
    x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output


def fp_quant_e3_per_group(x, n_bits, group_size=128):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-16, -8, -4, -2, -1, -0.5, -0.25, 0.0, 0.25, 0.5, 1, 2, 4, 8, 16]).cuda()
    x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale
    output = output.reshape(x_shape)
    return output


def fp_quant_e2_per_token(x, n_bits):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
    x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output


def fp_quant_e2_per_group(x, n_bits, group_size=128):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
    # x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    x = x.view(-1, group_size)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x.div_(scale)
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale
    output = output.view(x_shape)
    return output


def fp_quant_e2_per_group_cuda(x, n_bits, group_size=128):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).cuda()
    # x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    x_shape_1 = x.shape
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quant_array = x.view(-1)
    quant_grid = quant_grid.type_as(quant_array)
    # round to nearest fp
    quant_array, _ = quant_cuda.quant(quant_array, quant_grid)
    quant_array = quant_array.view(x_shape_1)
    output = quant_array * scale
    output = output.view(x_shape)
    return output


def fp_quant_e1_per_token(x, n_bits):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]).cuda()
    x = torch.clamp(x, -3, 3)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale

    return output


def fp_quant_e1_per_group(x, n_bits, group_size=128):
    assert n_bits == 4
    '''4-bit 量化'''
    quant_grid = torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]).cuda()
    x = torch.clamp(x, -3, 3)
    x_shape = x.shape
    x = x.reshape(-1, group_size)
    scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
    x  = x / scale
    quantized_x = quantize_to_nearest_grid(x, quant_grid)
    output = quantized_x * scale
    output = output.reshape(x_shape)
    return output


def fp_quant_e1m2_neg_e2m1_pos_per_group(x, n_bits, group_size=128, clipping_strength=1.0):
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


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant=None,
        quantize_output=False,
        w_bit=8,
        a_bit=8,
        act_quant_sym=True,
        fc2_act_log2_quant=False,
        activation_fp_quant=False,
        weight_fp_quant=False,
        act_fp_type=False,
        weight_fp_type=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act_quant_sym = act_quant_sym
        self.fc2_act_log2_quant = fc2_act_log2_quant
        self.activation_fp_quant = activation_fp_quant
        self.weight_fp_quant = weight_fp_quant
        self.act_fp_type = act_fp_type

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)
        
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            if self.activation_fp_quant == True: # fp quant
                if act_fp_type == "fp_e1":
                    self.act_quant = partial(fp_quant_e1_per_token, n_bits=a_bit)
                elif act_fp_type == "fp_e2":
                    self.act_quant = partial(fp_quant_e2_per_token, n_bits=a_bit)
                elif act_fp_type == "fp_e3":
                    self.act_quant = partial(fp_quant_e3_per_token, n_bits=a_bit)
                elif act_fp_type == "fp6_e2m3":
                    self.act_quant = partial(fp6_quant_e2m3_per_token_cuda, n_bits=a_bit)
                elif act_fp_type == "fp6_e3m2":
                    self.act_quant = partial(fp6_quant_e3m2_per_token_cuda, n_bits=a_bit)
                else:
                    raise ValueError(f"Unsupported fp_type.")
            elif self.act_quant_sym == True: # linear quant
                self.act_quant = partial(quantize_activation_per_token_sym, n_bits=a_bit)
            else:
                self.act_quant = partial(quantize_activation_per_token_asymmetric, n_bits=a_bit)

        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            if self.act_quant_sym == True:
                self.act_quant = partial(quantize_activation_per_tensor_sym, n_bits=a_bit)
            else:
                self.act_quant = partial(quantize_activation_per_tensor_asymmetric, n_bits=a_bit)

        elif act_quant == "per_group":
            self.act_quant_name = "per_group"
            if self.activation_fp_quant == True: # fp quant
                if act_fp_type == "fp_e1":
                    self.act_quant = partial(fp_quant_e1_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp_e2":
                    self.act_quant = partial(fp_quant_e2_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp_e3":
                    self.act_quant = partial(fp_quant_e3_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp6_e2m3":
                    self.act_quant = partial(fp6_quant_e2m3_per_group_cuda, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp6_e3m2":
                    self.act_quant = partial(fp6_quant_e3m2_per_group_cuda, n_bits=a_bit, group_size=128)
                else:
                    raise ValueError(f"Unsupported fp_type.")
            elif self.fc2_act_log2_quant == True: # log2 quant
                self.act_quant = partial(log2_quant_per_group_asym, n_bits=a_bit, group_size=128)
            elif self.act_quant_sym == True: # linear sym quant
                self.act_quant = partial(quantize_activation_per_group_sym, n_bits=a_bit, group_size=128)
            else: # linear asym quant
                self.act_quant = partial(quantize_activation_per_group_asymmetric, n_bits=a_bit, group_size=128)
            
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        # pdb.set_trace()
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False,
        w_bit=8, a_bit=8, act_quant_sym=None, fc2_act_log2_quant=False, 
        activation_fp_quant=False, weight_fp_quant=False,
        act_fp_type=None, weight_fp_type=None
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = QuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            w_bit=w_bit,
            a_bit=a_bit,
            act_quant_sym=act_quant_sym,
            fc2_act_log2_quant=fc2_act_log2_quant,
            activation_fp_quant=activation_fp_quant,
            weight_fp_quant=weight_fp_quant,
            act_fp_type=act_fp_type,
            weight_fp_type=weight_fp_type,
        )
        if weight_quant == "per_channel":
            if weight_fp_quant == True:
                if weight_fp_type == "fp_e1":
                    new_module.weight = fp_quant_e1_per_token(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp_e2":
                    new_module.weight = fp_quant_e2_per_token(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp_e3":
                    new_module.weight = fp_quant_e3_per_token(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp6_e2m3":
                    new_module.weight = fp6_quant_e2m3_per_token_cuda(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp6_e3m2":
                    new_module.weight = fp6_quant_e3m2_per_token_cuda(
                        module.weight, n_bits=w_bit
                    )
                else: 
                    raise ValueError(f"Unsupported fp_type.")
            else:
                new_module.weight = quantize_weight_per_channel_sym(
                    module.weight, n_bits=w_bit
                )

        if weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_sym(
                module.weight, n_bits=w_bit
            )
        
        if weight_quant == "per_group":
            if weight_fp_quant == True:
                if weight_fp_type == "fp_e1":
                    new_module.weight = fp_quant_e1_per_group(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp_e2":
                    new_module.weight = fp_quant_e2_per_group(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp_e3":
                    new_module.weight = fp_quant_e3_per_group(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp6_e2m3":
                    new_module.weight = fp6_quant_e2m3_per_group_cuda(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp6_e3m2":
                    new_module.weight = fp6_quant_e3m2_per_group_cuda(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                else: 
                    raise ValueError(f"Unsupported fp_type.")
            else:
                new_module.weight = quantize_weight_per_group_sym(
                    module.weight, n_bits=w_bit, group_size=128
                )

        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QuantizedLinear{self.in_features}, {self.out_features}, bias={self.bias is not None}, " \
            f"weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, " \
            f"w_bit={self.w_bit}, a_bit={self.a_bit}, act_quant_sym={self.act_quant_sym}, act_log2_quant={self.fc2_act_log2_quant}," \
            f"activation_fp_quant={self.activation_fp_quant}, weight_fp_quant={self.weight_fp_quant}, " \
            f"activation_quant_type={self.act_fp_type}"


class QuantizedLinear_fc2(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        w_bit=8,
        a_bit=8,
        act_quant_sym=True,
        fc2_act_log2_quant=False,
        activation_fp_quant=False,
        weight_fp_quant=False,
        act_fp_type=False,
        weight_fp_type=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act_quant_sym = act_quant_sym
        self.fc2_act_log2_quant = fc2_act_log2_quant
        self.activation_fp_quant = activation_fp_quant
        self.weight_fp_quant = weight_fp_quant


        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)
        
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            if self.activation_fp_quant == True: # fp quant
                if act_fp_type == "fp_e1":
                    self.act_quant = partial(fp_quant_e1_per_token, n_bits=a_bit)
                elif act_fp_type == "fp_e2":
                    self.act_quant = partial(fp_quant_e2_per_token, n_bits=a_bit)
                elif act_fp_type == "fp_e3":
                    self.act_quant = partial(fp_quant_e3_per_token, n_bits=a_bit)
                elif act_fp_type == "fp6_e2m3":
                    self.act_quant = partial(fp6_quant_e2m3_per_token_cuda, n_bits=a_bit)
                elif act_fp_type == "fp6_e3m2":
                    self.act_quant = partial(fp6_quant_e3m2_per_token_cuda, n_bits=a_bit)
                else:
                    raise ValueError(f"Unsupported fp_type.")
            elif self.act_quant_sym == True: # linear quant
                self.act_quant = partial(quantize_activation_per_token_sym, n_bits=a_bit)
            else:
                self.act_quant = partial(quantize_activation_per_token_asymmetric, n_bits=a_bit)

        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            if self.act_quant_sym == True:
                self.act_quant = partial(quantize_activation_per_tensor_sym, n_bits=a_bit)
            else:
                self.act_quant = partial(quantize_activation_per_tensor_asymmetric, n_bits=a_bit)

        elif act_quant == "per_group":
            self.act_quant_name = "per_group"
            if self.activation_fp_quant == True: # fp quant
                if act_fp_type == "fp_e1":
                    self.act_quant = partial(fp_quant_e1_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp_e2":
                    self.act_quant = partial(fp_quant_e2_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp_e3":
                    self.act_quant = partial(fp_quant_e3_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp_e1m2_neg_e2m1_pos":
                    self.act_quant = partial(fp_quant_e1m2_neg_e2m1_pos_per_group, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp6_e2m3":
                    self.act_quant = partial(fp6_quant_e2m3_per_group_cuda, n_bits=a_bit, group_size=128)
                elif act_fp_type == "fp6_e3m2":
                    self.act_quant = partial(fp6_quant_e3m2_per_group_cuda, n_bits=a_bit, group_size=128)
                else:
                    raise ValueError(f"Unsupported fp_type.")
            elif self.fc2_act_log2_quant == True: # log2 quant
                self.act_quant = partial(log2_quant_per_group_asym, n_bits=a_bit, group_size=128)
            elif self.act_quant_sym == True: # linear sym quant
                self.act_quant = partial(quantize_activation_per_group_sym, n_bits=a_bit, group_size=128)
            else: # linear asym quant
                self.act_quant = partial(quantize_activation_per_group_asymmetric, n_bits=a_bit, group_size=128)
            
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        # pdb.set_trace()
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False,
        w_bit=8, a_bit=8, act_quant_sym=None, fc2_act_log2_quant=False, 
        activation_fp_quant=False, weight_fp_quant=False, 
        act_fp_type=None, weight_fp_type=None,
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = QuantizedLinear_fc2(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            w_bit=w_bit,
            a_bit=a_bit,
            act_quant_sym=act_quant_sym,
            fc2_act_log2_quant=fc2_act_log2_quant,
            activation_fp_quant=activation_fp_quant,
            weight_fp_quant=weight_fp_quant,
            act_fp_type = act_fp_type,
            weight_fp_type = weight_fp_type,
        )
        if weight_quant == "per_channel":
            if weight_fp_quant == True:
                if weight_fp_type == "fp_e1":
                    new_module.weight = fp_quant_e1_per_token(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp_e2":
                    new_module.weight = fp_quant_e2_per_token(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp_e3":
                    new_module.weight = fp_quant_e3_per_token(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp6_e2m3":
                    new_module.weight = fp6_quant_e2m3_per_token_cuda(
                        module.weight, n_bits=w_bit
                    )
                elif weight_fp_type == "fp6_e3m2":
                    new_module.weight = fp6_quant_e3m2_per_token_cuda(
                        module.weight, n_bits=w_bit
                    )
                else: 
                    raise ValueError(f"Unsupported fp_type.")
            else:
                new_module.weight = quantize_weight_per_channel_sym(
                    module.weight, n_bits=w_bit
                )

        if weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_sym(
                module.weight, n_bits=w_bit
            )
        
        if weight_quant == "per_group":
            if weight_fp_quant == True:
                if weight_fp_type == "fp_e1":
                    new_module.weight = fp_quant_e1_per_group(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp_e2":
                    new_module.weight = fp_quant_e2_per_group(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp_e3":
                    new_module.weight = fp_quant_e3_per_group(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp6_e2m3":
                    new_module.weight = fp6_quant_e2m3_per_group_cuda(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                elif weight_fp_type == "fp6_e3m2":
                    new_module.weight = fp6_quant_e3m2_per_group_cuda(
                        module.weight, n_bits=w_bit, group_size=128
                    )
                else: 
                    raise ValueError(f"Unsupported fp_type.")
            else:
                new_module.weight = quantize_weight_per_group_sym(
                    module.weight, n_bits=w_bit, group_size=128
                )
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QuantizedLinear_fc2{self.in_features}, {self.out_features}, bias={self.bias is not None}, " \
            f"weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name}, " \
            f"w_bit={self.w_bit}, a_bit={self.a_bit}, act_quant_sym={self.act_quant_sym}, act_log2_quant={self.fc2_act_log2_quant}," \
            f"activation_fp_quant={self.activation_fp_quant}, weight_fp_quant={self.weight_fp_quant}"
    

def quantize_VAR(
    model, 
    weight_quant=None, act_quant=None, quantize_bmm_input=False,
    w_bit=8, a_bit=8, kv_bit=8,
    act_quant_sym=None, fc2_act_log2_quant=None, quant_kv=None,
    activation_fp_quant=False, weight_fp_quant=False, 
    act_fp_type=None,
    weight_fp_type=None,
    fc2_fp_type=None,
):

    for name, m in model.named_modules():
        if isinstance(m, FFN):
            # pdb.set_trace()
            m.fc1 = QuantizedLinear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant, 
                act_fp_type=act_fp_type,
                weight_fp_type=weight_fp_type
            )

            m.fc2 = QuantizedLinear_fc2.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=False,
                fc2_act_log2_quant=fc2_act_log2_quant,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant, 
                act_fp_type=fc2_fp_type,
                weight_fp_type=weight_fp_type
            )

        elif isinstance(m, SelfAttention):
            m.mat_qkv = QuantizedLinear.from_float(
                m.mat_qkv, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant,
                act_fp_type=act_fp_type,
                weight_fp_type=weight_fp_type,
            )

            m.proj = QuantizedLinear.from_float(
                m.proj, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant,
                act_fp_type=act_fp_type,
                weight_fp_type=weight_fp_type,
            )

        elif isinstance(m, AdaLNSelfAttn):
            m.ada_lin[1] = QuantizedLinear.from_float(
                m.ada_lin[1], weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant,
                act_fp_type=act_fp_type,
                weight_fp_type=weight_fp_type,
            )

    return model



def quantize_VAR_use_different_datatype(
    model, 
    weight_quant=None, act_quant=None, quantize_bmm_input=False,
    w_bit=8, a_bit=8, kv_bit=8,
    act_quant_sym=None, fc2_act_log2_quant=None, quant_kv=None,
    activation_fp_quant=False, weight_fp_quant=False, 
    act_fp_type=None,
    weight_fp_type=None,
    fc2_fp_type=None,
):

    for name, m in model.named_modules():
        if isinstance(m, FFN):
            # pdb.set_trace()
            block_num = int(name.split('.')[1])
            if block_num in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                m.fc1 = QuantizedLinear.from_float(
                    m.fc1, weight_quant=weight_quant, act_quant=act_quant,
                    w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                    activation_fp_quant=activation_fp_quant, 
                    weight_fp_quant=weight_fp_quant, 
                    act_fp_type="fp_e2",
                    weight_fp_type="fp_e2"
                )
            else:
                m.fc1 = QuantizedLinear.from_float(
                    m.fc1, weight_quant=weight_quant, act_quant=act_quant,
                    w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                    activation_fp_quant=activation_fp_quant, 
                    weight_fp_quant=weight_fp_quant, 
                    act_fp_type="fp_e3",
                    weight_fp_type="fp_e2"
                )

            m.fc2 = QuantizedLinear_fc2.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=False,
                fc2_act_log2_quant=fc2_act_log2_quant,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant, 
                act_fp_type = fc2_fp_type,
                weight_fp_type = weight_fp_type
            )

        elif isinstance(m, SelfAttention):
            # pdb.set_trace()
            block_num = int(name.split('.')[1])
            if block_num in [24, 25]:
                m.mat_qkv = QuantizedLinear.from_float(
                    m.mat_qkv, weight_quant=weight_quant, act_quant=act_quant,
                    w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                    activation_fp_quant=activation_fp_quant, 
                    weight_fp_quant=weight_fp_quant, 
                    act_fp_type = "fp_e2",
                    weight_fp_type = "fp_e2"
                )
            else:
                m.mat_qkv = QuantizedLinear.from_float(
                    m.mat_qkv, weight_quant=weight_quant, act_quant=act_quant,
                    w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                    activation_fp_quant=activation_fp_quant, 
                    weight_fp_quant=weight_fp_quant, 
                    act_fp_type = "fp_e3",
                    weight_fp_type = "fp_e2"
                )

            m.proj = QuantizedLinear.from_float(
                m.proj, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant, 
                act_fp_type = act_fp_type,
                weight_fp_type = weight_fp_type,
            )

        elif isinstance(m, AdaLNSelfAttn):
            m.ada_lin[1] = QuantizedLinear.from_float(
                m.ada_lin[1], weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym,
                activation_fp_quant=activation_fp_quant, 
                weight_fp_quant=weight_fp_quant,
                act_fp_type=act_fp_type,
                weight_fp_type=weight_fp_type,
            )

    return model