import torch
import numpy as np
import pdb
from functools import partial
import argparse
import torch.nn as nn

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
    

class FPQuantizer(nn.Module):
    def __init__(self, n_bits=4, group_size=128, format=None, device=None):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.format = format
        self.device = device
        self.clipping_strength = nn.Parameter(torch.tensor(1.0))  # 可训练参数 alpha

        # 初始化量化网格
        self.register_buffer("quant_grid", self._get_quant_grid(format))

    def _get_quant_grid(self, format):
        if format == "e1m2":
            return torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 
                                0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], device=self.device)
        elif format == "e2m1":
            return torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 
                                0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=self.device)
        elif format == "e3m0":
            return torch.tensor([-16, -8, -4, -2, -1, -0.5, -0.25, 0.0, 
                                0.25, 0.5, 1, 2, 4, 8, 16], device=self.device)
        else:
            raise ValueError("Unsupported format")

    def forward(self, x):
        clip_value = self.clipping_strength * x.abs().max()
        x = torch.clamp(x, -clip_value, clip_value)
        
        x_shape = x.shape
        x = x.reshape(-1, self.group_size)
        scale = x.abs().max(dim=-1, keepdim=True)[0] / self.quant_grid.abs().max()
        x = x / scale
        x_quant = self.quantize_to_nearest_grid(x, self.quant_grid)
        output = x + (x_quant * scale - x).detach() # STE
        output = output.reshape(x_shape)
        return output

    def quantize_to_nearest_grid(self, x, grid):
        # 确保 quant_grid 是升序排列
        # assert torch.all(torch.eq(grid, torch.sort(grid).values)), "quant_grid must be sorted."
        
        # 计算所有绝对距离
        distances = torch.abs(x.unsqueeze(-1) - grid)
        
        # 找到每个元素的最小距离索引
        min_indices = torch.argmin(distances, dim=-1) 

        # 返回对应的 quant_grid 值
        return grid[min_indices]


class FPQuantizer_sigmoid(nn.Module):
    def __init__(self, n_bits=4, group_size=128, format=None, device=None):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.format = format
        self.device = device
        init_value = 5
        self.clipping_strength_logit = nn.Parameter(torch.tensor(1.0)*init_value)  # 可训练参数 alpha
        # 初始化量化网格
        self.register_buffer("quant_grid", self._get_quant_grid(format))

    def _get_quant_grid(self, format):
        if format == "e1m2":
            return torch.tensor([-1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 
                                0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], device=self.device)
        elif format == "e2m1":
            return torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 
                                0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=self.device)
        elif format == "e3m0":
            return torch.tensor([-16, -8, -4, -2, -1, -0.5, -0.25, 0.0, 
                                0.25, 0.5, 1, 2, 4, 8, 16], device=self.device)
        else:
            raise ValueError("Unsupported format")

    def forward(self, x):
        clip_value = torch.sigmoid(self.clipping_strength_logit) * x.abs().max()
        x = torch.clamp(x, -clip_value, clip_value)
        
        x_shape = x.shape
        x = x.reshape(-1, self.group_size)
        scale = x.abs().max(dim=-1, keepdim=True)[0] / self.quant_grid.abs().max()
        x = x / scale
        x_quant = self.quantize_to_nearest_grid(x, self.quant_grid)
        output = x + (x_quant * scale - x).detach()
        output = output.reshape(x_shape)
        return output

    def quantize_to_nearest_grid(self, x, grid):
        # 确保 quant_grid 是升序排列
        # assert torch.all(torch.eq(grid, torch.sort(grid).values)), "quant_grid must be sorted."
        
        # 计算所有绝对距离
        distances = torch.abs(x.unsqueeze(-1) - grid)
        
        # 找到每个元素的最小距离索引
        min_indices = torch.argmin(distances, dim=-1) 

        # 返回对应的 quant_grid 值
        return grid[min_indices]


def compute_quant_error(x_fp, x_quant):
    # quant_error = torch.mean((x_fp - x_quant)**2)
    quant_error = torch.sum((x_fp - x_quant)**2)
    return quant_error



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    '''搜索量化格式，并且搜索clipping '''

    # 方法1：对alpha加penalty
    # lr = 0.01
    # epochs = 100
    # optim_formats_for_each_block = {}

    # for block_idx in range(30):
    #     print(f"\n\n====== Processing block {block_idx} ======")
    #     w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ada.pt", weights_only=True)
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.proj.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc1.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc2.pt", weights_only=True).cuda()

    #     formats = ['e1m2', 'e2m1', 'e3m0']
    #     best_results = {'loss': float('inf'), 'format': None, 'alpha': None}

    #     for format in formats:
    #         # 初始化可训练参数
    #         alpha = nn.Parameter(torch.tensor(1.0, device=device))
    #         optimizer = torch.optim.AdamW([alpha], lr=lr)
            
    #         # 跟踪当前格式的最佳参数
    #         current_best = {'loss': float('inf'), 'alpha': alpha.detach().clone()}
            
    #         for epoch in range(epochs):
    #             # 量化过程
    #             w_quant = FPQuant.apply(w_fp, 4, 128, format, alpha)
                
    #             # 计算损失
    #             loss = compute_quant_error(w_fp, w_quant)
                
    #             penalty = torch.nn.functional.relu(alpha - 1.0).pow(2) * 1e1

    #             loss = loss + penalty

    #             # 反向传播
    #             loss.backward()
    #             # debug
    #             print(f"Gradient: {alpha.grad.item():.4f}")
    #             optimizer.step()
    #             optimizer.zero_grad()
                
    #             # 更新当前格式的最佳参数
    #             if loss.item() < current_best['loss']:
    #                 current_best['loss'] = loss.item()
    #                 current_best['alpha'] = alpha.detach().clone()
                    
    #             # 打印训练信息
    #             if epoch % 10 == 0:
    #                 print(f"Block {block_idx} | {format} | Epoch {epoch:03d} | "
    #                     f"Loss {loss.item():.6f} | Alpha {alpha.item():.4f}")

    #         # 更新全局最佳结果
    #         if current_best['loss'] < best_results['loss']:
    #             best_results = {
    #                 'loss': current_best['loss'],
    #                 'format': format,
    #                 'alpha': current_best['alpha']
    #             }

    #     # 保存当前block的最佳结果
    #     optim_formats_for_each_block[f"blocks.{block_idx}.ada"] = {
    #         'format': best_results['format'],
    #         'alpha': best_results['alpha'].cpu().item()  # 转换为Python float
    #     }

    #     print(f"Optimal format: {best_results['format']} with alpha {best_results['alpha'].item():.4f}, loss: {best_results['loss']:.6f}")


    # # 保存最终结果
    # torch.save(optim_formats_for_each_block, "optim_quant_formats_with_alpha_penalty.pth")

    # 方法1, 对alpha加penalty，使用quantizer class
    lr = 0.01
    epochs = 100
    optim_formats_for_each_block = {}

    for block_idx in range(30):
        print(f"\n\n====== Processing block {block_idx} ======")
        w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ada.pt", weights_only=True).to(device)
        # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True).cuda()
        # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.proj.pt", weights_only=True).cuda()
        # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc1.pt", weights_only=True).cuda()
        # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc2.pt", weights_only=True).cuda()

        formats = ['e1m2', 'e2m1', 'e3m0']
        best_results = {'loss': float('inf'), 'format': None, 'alpha': None}

        for format in formats:
            quantizer = FPQuantizer(format=format, device=device)

            # 初始化可训练参数
            optimizer = torch.optim.AdamW([quantizer.clipping_strength], lr=lr)
            
            # 跟踪当前格式的最佳参数
            current_best = {'loss': float('inf'), 'alpha': quantizer.clipping_strength.detach().clone()}
            
            for epoch in range(epochs):
                # 量化过程
                w_quant = quantizer(w_fp)
                
                # 计算损失
                loss = compute_quant_error(w_fp, w_quant)
                
                penalty = torch.nn.functional.relu(quantizer.clipping_strength - 1.0).pow(2) * 1e2

                loss = loss + penalty

                # 反向传播
                loss.backward()
                # debug
                # print(f"Gradient: {quantizer.clipping_strength.grad.item():.4f}")
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新当前格式的最佳参数
                if loss.item() < current_best['loss']:
                    current_best['loss'] = loss.item()
                    current_best['alpha'] = quantizer.clipping_strength.detach().clone()
                    
                # 打印训练信息
                if epoch % 10 == 0:
                    print(f"Block {block_idx} | {format} | Epoch {epoch:03d} | "
                        f"Loss {loss.item():.6f} | Alpha {quantizer.clipping_strength.item():.4f}")

            # 更新全局最佳结果
            if current_best['loss'] < best_results['loss']:
                best_results = {
                    'loss': current_best['loss'],
                    'format': format,
                    'alpha': current_best['alpha']
                }

        # 保存当前block的最佳结果
        optim_formats_for_each_block[f"blocks.{block_idx}.ada"] = {
            'format': best_results['format'],
            'alpha': best_results['alpha'].cpu().item()  # 转换为Python float
        }

        print(f"Optimal format: {best_results['format']} with alpha {best_results['alpha'].item():.4f}, loss: {best_results['loss']:.6f}")


    # 保存最终结果
    torch.save(optim_formats_for_each_block, "optim_quant_formats_with_alpha_penalty.pth")
    pdb.set_trace()


    # # 方法2：使用sigmoid函数
    # optim_formats_for_each_block = {}
    # lr = 0.01
    # epochs = 100

    # for block_idx in range(30):
    #     print(f"\n\n====== Processing block {block_idx} ======")
    #     w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ada.pt", weights_only=True)
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.proj.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc1.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc2.pt", weights_only=True).cuda()

    #     formats = ['e1m2', 'e2m1', 'e3m0']
    #     best_results = {'loss': float('inf'), 'format': None, 'alpha': None}

    #     for format in formats:
    #         # 初始化可训练参数
    #         logit_alpha = nn.Parameter(torch.tensor(0.0, device=device))
    #         optimizer = torch.optim.AdamW([logit_alpha], lr=lr)
            
    #         alpha = torch.sigmoid(logit_alpha)
    #         # 跟踪当前格式的最佳参数
    #         current_best = {'loss': float('inf'), 'alpha': alpha.detach().clone()}
            
    #         for epoch in range(epochs):
    #             # 量化过程
    #             alpha = torch.sigmoid(logit_alpha)
    #             w_quant = FPQuant.apply(w_fp, 4, 128, format, alpha)
                
    #             # 计算损失
    #             loss = compute_quant_error(w_fp, w_quant)

    #             # 反向传播
    #             loss.backward()
    #             print(f"Gradient: {logit_alpha.grad}")
    #             optimizer.step()
    #             optimizer.zero_grad()
                
    #             # 更新当前格式的最佳参数
    #             if loss.item() < current_best['loss']:
    #                 current_best['loss'] = loss.item()
    #                 current_best['alpha'] = alpha.detach().clone()
                    
    #             # 打印训练信息
    #             if epoch % 10 == 0:
    #                 print(f"Block {block_idx} | {format} | Epoch {epoch:03d} | "
    #                     f"Loss {loss.item():.6f} | Alpha {alpha.item():.4f}")

    #         # 更新全局最佳结果
    #         if current_best['loss'] < best_results['loss']:
    #             best_results = {
    #                 'loss': current_best['loss'],
    #                 'format': format,
    #                 'alpha': current_best['alpha']
    #             }

    #     # 保存当前block的最佳结果
    #     optim_formats_for_each_block[f"blocks.{block_idx}.ada"] = {
    #         'format': best_results['format'],
    #         'alpha': best_results['alpha'].cpu().item()  # 转换为Python float
    #     }

    #     print(f"Optimal format: {best_results['format']} with alpha {best_results['alpha'].item():.4f}, loss: {best_results['loss']:.6f}")


    # # 方法2, 对alpha使用sigmoid函数，使用quantizer class
    # lr = 0.01
    # epochs = 100
    # optim_formats_for_each_block = {}

    # for block_idx in range(30):
    #     print(f"\n\n====== Processing block {block_idx} ======")
    #     w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ada.pt", weights_only=True).to(device)
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.proj.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc1.pt", weights_only=True).cuda()
    #     # w_fp = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.ffn.fc2.pt", weights_only=True).cuda()

    #     formats = ['e1m2', 'e2m1', 'e3m0']
    #     best_results = {'loss': float('inf'), 'format': None, 'alpha': None}

    #     for format in formats:
    #         quantizer = FPQuantizer_sigmoid(format=format, device=device)

    #         # 初始化可训练参数
    #         optimizer = torch.optim.AdamW([quantizer.clipping_strength_logit], lr=lr)
            
    #         # 跟踪当前格式的最佳参数
    #         current_best = {'loss': float('inf'), 
    #                         'alpha': torch.sigmoid(quantizer.clipping_strength_logit.detach().clone())}
            
    #         for epoch in range(epochs):
    #             # 量化过程
    #             w_quant = quantizer(w_fp)
                
    #             # 计算损失
    #             loss = compute_quant_error(w_fp, w_quant)
            
    #             # 反向传播
    #             loss.backward()
    #             # debug
    #             # print(f"Gradient: {quantizer.clipping_strength_logit.grad.item():.4f}")
    #             optimizer.step()
    #             optimizer.zero_grad()
                
    #             # 更新当前格式的最佳参数
    #             if loss.item() < current_best['loss']:
    #                 current_best['loss'] = loss.item()
    #                 current_best['alpha'] = torch.sigmoid(quantizer.clipping_strength_logit.detach().clone())
                    
    #             # 打印训练信息
    #             if epoch % 10 == 0:
    #                 print(f"Block {block_idx} | {format} | Epoch {epoch:03d} | "
    #                     f"Loss {loss.item():.6f} | Alpha {torch.sigmoid(quantizer.clipping_strength_logit.detach().clone()):.4f}")

    #         # 更新全局最佳结果
    #         if current_best['loss'] < best_results['loss']:
    #             best_results = {
    #                 'loss': current_best['loss'],
    #                 'format': format,
    #                 'alpha': current_best['alpha']
    #             }

    #     # 保存当前block的最佳结果
    #     optim_formats_for_each_block[f"blocks.{block_idx}.ada"] = {
    #         'format': best_results['format'],
    #         'alpha': best_results['alpha'].cpu().item()  # 转换为Python float
    #     }

    #     print(f"Optimal format: {best_results['format']} with alpha {best_results['alpha'].item():.4f}, loss: {best_results['loss']:.6f}")

    # # 保存最终结果
    # torch.save(optim_formats_for_each_block, "optim_quant_formats_with_alpha_sigmoid.pth")

    # pdb.set_trace()



    # pdb.set_trace()
    block_idx = 9
    activation = []
    for step_idx in range(10):
        file_name = f"/home/rjwei/Data_raid/Q-VAR/activation_distribution_class_980/condition/block{block_idx}_step{step_idx}_condition.pt"
        x_fp = torch.load(file_name)    
        activation.append(x_fp)





    y_fp = torch.matmul(x_fp, w_fp.T)
    pdb.set_trace()



    x_quant = quantize_activation_per_token_sym(x_fp, 4)
    w_quant = quantize_activation_per_token_sym(w_fp, 4)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate linear quant per token error:", quant_error)

    # pdb.set_trace()

    x_quant = quantize_activation_per_group_sym(x_fp, 4, 128)
    w_quant = quantize_activation_per_group_sym(w_fp, 4, 128)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate linear quant per group error:", quant_error)


    x_quant = fp_quant_e1(x_fp)
    w_quant = fp_quant_e1(w_fp)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate fp_quant_e1 quant per token error:", quant_error)


    x_quant = fp_quant_e2(x_fp)
    w_quant = fp_quant_e2(w_fp)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate fp_quant_e2 quant per token error:", quant_error)


    x_quant = fp_quant_e3(x_fp)
    w_quant = fp_quant_e3(w_fp)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate fp_quant_e3 quant per token error:", quant_error)


    x_quant = fp_quant_e1_per_group(x_fp)
    w_quant = fp_quant_e1_per_group(w_fp)
    y_quant = torch.matmul(x_quant, w_quant.T)
    # pdb.set_trace()
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate fp_quant_e1 quant per group error:", quant_error)


    x_quant = fp_quant_e2_per_group(x_fp)
    w_quant = fp_quant_e2_per_group(w_fp)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate fp_quant_e2 quant per group error:", quant_error)


    x_quant = fp_quant_e3_per_group(x_fp)
    w_quant = fp_quant_e3_per_group(w_fp)
    y_quant = torch.matmul(x_quant, w_quant.T)
    quant_error = compute_quant_error(y_fp, y_quant)
    print("rotate fp_quant_e3 quant per group error:", quant_error)