import sys
sys.path.append("/home/rjwei/Q-VAR")
import torch
import numpy as np
from rotate_utils import rotation_utils
import argparse
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import time

# # per group linear quantize
# @torch.no_grad()
# def quantize_activation_per_group_sym(t, n_bits=4, group_size=128):
#     t_shape = t.shape
#     t_flat = t.view(-1, group_size)
#     abs_max = t_flat.abs().max(dim=-1, keepdim=True)[0]
#     q_max = 2 ** (n_bits - 1) - 1
#     q_min = - 2 ** (n_bits - 1)
#     scales = abs_max.div_(q_max).clamp_(min=1e-5)
#     t_flat_quant = torch.clamp(torch.round(t_flat / scales), q_min, q_max).mul_(scales)
#     t_hat = t_flat_quant.view(t_shape)
#     return t_hat


class SymQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_bits=4, group_size=128):
        t_shape = input.shape
        t_flat = input.view(-1, group_size)
        abs_max = t_flat.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        q_min = - 2 ** (n_bits - 1)
        scales = abs_max.div_(q_max).clamp_(min=1e-5)
        t_flat_quant = torch.clamp(torch.round(t_flat / scales), q_min, q_max).mul_(scales)
        t_hat = t_flat_quant.view(t_shape)
        return t_hat

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


class FPQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits=4, group_size=128):
        assert n_bits == 4
        '''4-bit 量化'''
        quant_grid = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]).to(x.device)
        # x = torch.clamp(x, -3, 3)
        x_shape = x.shape
        x = x.view(-1, group_size)
        scale = x.abs().max(dim=-1, keepdim=True)[0] / quant_grid.abs().max()
        x  = x / scale
        quantized_x = quantize_to_nearest_grid(x, quant_grid)
        output = quantized_x * scale
        output = output.view(x_shape)
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
    

def compute_quant_error(x, w, learnable_s, Q):
    # pdb.set_trace()
    fp_result = torch.matmul(x, w.T)

    x_1 = x * learnable_s
    x_2 = torch.matmul(x_1, Q)
    x_2_quant = FPQuant.apply(x_2)

    w_1 = w / learnable_s
    w_2 = torch.matmul(w_1, Q)
    w_2_quant = FPQuant.apply(w_2)

    quant_result = torch.matmul(x_2_quant, w_2_quant.T)

    quant_error = torch.mean((fp_result - quant_result)**2)

    return quant_error


def compute_quant_error_v1(x, w, learnable_s, Q):
    # pdb.set_trace()
    fp_result = torch.matmul(x, w.T)

    x_1 = x * learnable_s
    x_2 = torch.matmul(x_1, Q)
    x_2_quant = FPQuant.apply(x_2)

    w_1 = w / learnable_s
    w_2 = torch.matmul(w_1, Q)
    w_2_quant = FPQuant.apply(w_2)

    quant_result = torch.matmul(x_2_quant, w_2_quant.T)
    # pdb.set_trace()
    quant_error = torch.mean(((fp_result - quant_result)**2))

    return quant_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    
    # # # preprocess: 对于每个step, 把100个样本拼成一个大tensor
    # for block_idx in range(30):
    #     print("block_idx:", block_idx)
    #     for step_idx in range(10):

    #         activation = []
    #         for label_idx in range(100):
    #             file_name = f"/home/rjwei/Data_raid/Q-VAR/cali_data/mat_qkv_input/label{label_idx}_block{block_idx}_step{step_idx}_mat_qkv_input.pt"
    #             x_fp = torch.load(file_name).to(device)
    #             activation.append(x_fp)

    #         activation_tensor = torch.stack(activation)
    #         torch.save(activation_tensor, f"/home/rjwei/Data_raid/Q-VAR/cali_data/learnable_s/mat_qkv_block{block_idx}_step{step_idx}.pt")

    # pdb.set_trace()

    # block rotation matrix 
    # total_size = 1920
    # block_size = 128
    # Q = rotation_utils.block_random_hadamard_matrix(
    #     total_size=total_size,
    #     block_size=block_size,
    #     device=device,
    #     seed=42
    # ).to(torch.float32)
    
    Q = rotation_utils.get_orthogonal_matrix(1920, "hadamard", device).to(torch.float32)


    learnable_s_mat_qkv = []

    # # load activation和weight
    # for block_idx in range(30):
    #     print("block_idx:", block_idx)
    #     activation = []
    #     for label_idx in range(100):
    #         for step_idx in range(10):
    #             file_name = f"/home/rjwei/Data_raid/Q-VAR/cali_data/mat_qkv_input/label{label_idx}_block{block_idx}_step{step_idx}_mat_qkv_input.pt"
    #             x_fp = torch.load(file_name).to(device)
    #             activation.append(x_fp)

    #     weight = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True)
    #     weight = weight.to(device)

    #     learnable_s = nn.Parameter(torch.ones(1920, device=device))
    #     lr = 0.01
    #     epochs = 100
    #     optimizer = torch.optim.AdamW([learnable_s], lr=lr)

    #     best_loss = float('inf')

    #     for i in range(epochs):
    #         total_loss = torch.tensor(0.0, device=device)
        
    #         for j in range(len(activation)):
    #             total_loss += compute_quant_error(activation[j], weight, learnable_s, Q)
            
    #         total_loss /= len(activation)  # 平均损失
    #         total_loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         if total_loss < best_loss:
    #             best_loss = total_loss
    #             best_s = learnable_s
            
    #         # 打印训练进度
    #         print(f"Epoch {i}: Loss={total_loss.item():.6f}")
        
    #     print("best_loss:", best_loss)
    #     print("best_s:", best_s)
    #     learnable_s_mat_qkv.append(best_s)


    # load activation和weight v1
    # for block_idx in range(30):
    #     print("block_idx:", block_idx)
    #     activation = []
    #     for step_idx in range(10):
    #         x = torch.load(f"/home/rjwei/Data_raid/Q-VAR/cali_data/learnable_s/mat_qkv_block{block_idx}_step{step_idx}.pt").to(device)
    #         # pdb.set_trace()
    #         activation.append(x)
    #         weight = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True)
    #         weight = weight.to(device)

    #     learnable_s = nn.Parameter(torch.ones(1920, device=device))
    #     lr = 0.01
    #     epochs = args.epochs
    #     optimizer = torch.optim.AdamW([learnable_s], lr=lr)

    #     best_loss = float('inf')

    #     for i in range(epochs):
    #         total_loss = torch.tensor(0.0, device=device)

    #         for j in range(10):
    #             total_loss += compute_quant_error_v1(activation[j], weight, learnable_s, Q)
            
    #         total_loss /= 10
    #         total_loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         if total_loss < best_loss:
    #             best_loss = total_loss
    #             best_s = learnable_s
            
    #         # 打印训练进度
    #         print(f"Epoch {i}: Loss={total_loss.item():.6f}")
        
    #     print("best_loss:", best_loss)
    #     print("best_s:", best_s)
    #     learnable_s_mat_qkv.append(best_s)
    
    # torch.save(learnable_s_mat_qkv, f"/home/rjwei/Data_raid/Q-VAR/best_s/mat_qkv_best_s.pt")

    start = time.time()
    # load activation和weight v2
    for block_idx in range(30):
        print("block_idx:", block_idx)
        activation = []
        for step_idx in range(10):
            x = torch.load(f"/home/rjwei/Data_raid/Q-VAR/cali_data/learnable_s/mat_qkv_block{block_idx}_step{step_idx}.pt", map_location=device)
            activation.append(x)
            weight = torch.load(f"/home/rjwei/Data_raid/Q-VAR/var30_weights/blocks.{block_idx}.attn.mat_qkv.pt", weights_only=True, map_location=device)
        learnable_s = nn.Parameter(torch.ones(1920, device=device))
        lr = 0.01
        # lr = 5e-3
        epochs = args.epochs
        optimizer = torch.optim.AdamW([learnable_s], lr=lr)

        best_loss = float('inf')

        for i in range(epochs):
            epoch_loss = 0.0

            for j in range(10):
                loss = compute_quant_error_v1(activation[j], weight, learnable_s, Q)
            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / 10
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_s = learnable_s
            
            # 打印训练进度
            print(f"Epoch {i}: Loss={avg_epoch_loss:.6f}")
        
        print("best_loss:", best_loss)
        print("best_s:", best_s)
        learnable_s_mat_qkv.append(best_s)

    torch.cuda.synchronize()
    latency = (time.time() - start)
    print(f"Total Time: {latency:.2f}")
    
    pdb.set_trace()
    torch.save(learnable_s_mat_qkv, f"/home/rjwei/Data_raid/Q-VAR/best_s_512x512/mat_qkv_best_s_fp4.pt")