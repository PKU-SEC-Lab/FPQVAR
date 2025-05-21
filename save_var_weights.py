################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models_smoothquant import VQVAE, build_vae_var
import pdb
from quant_utils_smoothquant import quantize_VAR_smoothquant
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--act_sym', type=bool, default=False)
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}

    # download checkpoint
    vae_ckpt = '/opt/pretrained_models/var/vae_ch160v4096z32.pth'
    var_ckpt = f'/opt/pretrained_models/var/var_d{MODEL_DEPTH}.pth'

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    print("saving weights.......")

    for name, m in var.named_modules():
        if isinstance(m, torch.nn.Linear) and ('fc1' in name or 'mat_qkv' in name):
            print(name)
            torch.save(m.weight, f"var30_weights/{name}.pt")








