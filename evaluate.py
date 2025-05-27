################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
import pdb
import torch
from thop import profile

MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
vae_ckpt = '/home/rjwei/Data_raid/huggingface/models/var/vae_ch160v4096z32.pth'
var_ckpt = f'/home/rjwei/Data_raid/huggingface/models/var/var_d{MODEL_DEPTH}.pth'

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)

# class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}

more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

num_img_per_class = 1

cali_data_size = 100

for i in range(cali_data_size):
    print(i)
    class_labels = [i for _ in range(num_img_per_class)]
    B = len(class_labels)

    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=1.5, top_k=900, top_p=0.96, g_seed=seed, more_smooth=False)

    # pdb.set_trace()

    for j in range(num_img_per_class):
        print(j)
        # img_tensor = recon_B3HW[j].permute(1, 2, 0).mul(255).cpu().numpy()
        # chw = PImage.fromarray(img_tensor.astype(np.uint8))
        # chw.save(f'evaluate_figs_FP/class{i}_img{j}.png')

