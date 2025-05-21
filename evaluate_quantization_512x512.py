################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models_quant import VQVAE, build_vae_var
import pdb
from models_quant.quant_utils import quantize_VAR
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--act_sym', action="store_true", default=False)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--weight_quant', type=str, default="per_channel")
    parser.add_argument('--act_quant', type=str, default="per_token")

    args = parser.parse_args()

    MODEL_DEPTH = 36    # TODO: =====> please specify MODEL_DEPTH <=====
    # assert MODEL_DEPTH in {16, 20, 24, 30}


    # download checkpoint
    vae_ckpt = '/home/rjwei/Data_raid/huggingface/models/var/vae_ch160v4096z32.pth'
    var_ckpt = f'/home/rjwei/Data_raid/huggingface/models/var/var_d{MODEL_DEPTH}.pth'

    # build vae, var
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32) 

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=True,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # quantize model
    w_bit = args.w_bit
    a_bit = args.a_bit
    act_quant_sym = args.act_sym

    var = quantize_VAR(
            var,
            weight_quant=args.weight_quant,
            act_quant=args.act_quant,
            quantize_bmm_input=False,
            w_bit=w_bit,
            a_bit=a_bit,
            act_quant_sym=act_quant_sym
        ).to(device)   

    var = var.half()
    with open(f'log/w{w_bit}_{args.weight_quant}_a{a_bit}_{args.act_quant}_fc2_asym.txt', 'w') as f:
        print(var, file=f)
    
    # pdb.set_trace()

    ############################# 2. Sample with classifier-free guidance

    # set args
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    more_smooth = False # True for more smooth output

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    save_file_path = f'/home/rjwei/Data_raid/512x512_img/evaluate_figs_LiteVAR_w{w_bit}_{args.weight_quant}_a{a_bit}_{args.act_quant}' 

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    else:
        pass

    num_img_per_class = 10
    num_class = 1000
    num_iter = int(50 / num_img_per_class)

    for i in range(num_class):
        print(i)
        for k in range(num_iter):
            # 需要控制每次iter, random seed都不一样，否则会生成一样的图片
            seed = k + 10
            class_labels = [i for _ in range(num_img_per_class)]
            B = len(class_labels)

            label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                    recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=1.5, top_k=900, top_p=0.96, g_seed=seed, more_smooth=False)

            # pdb.set_trace()

            for j in range(num_img_per_class):

                img_tensor = recon_B3HW[j].permute(1, 2, 0).mul(255).cpu().numpy()
                chw = PImage.fromarray(img_tensor.astype(np.uint8))

                j = j + k * num_img_per_class
                print(j)

                chw.save(f'{save_file_path}/class{i}_img{j}.png')



