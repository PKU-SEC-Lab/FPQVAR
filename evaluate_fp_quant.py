################## 1. Download checkpoints and build models
import os
import os.path as osp
import sys

# 获取当前文件夹的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前文件夹添加到 sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)


import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models_fp_quant import VQVAE, build_vae_var
from models_fp_quant.quant_utils import quantize_VAR, quantize_VAR_mixed_fp4_datatype, quantize_VAR_mixed_fp6_datatype
import pdb
import argparse
from datetime import datetime
from rotate_utils import rotation_utils
from learnable_transformation import transform_model_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_bit', type=int, default=32)
    parser.add_argument('--a_bit', type=int, default=32)
    parser.add_argument('--kv_bit', type=int, default=32)
    parser.add_argument('--groupsize', type=int, default=128)

    parser.add_argument('--act_sym', action="store_true", default=False)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--weight_quant', type=str, default="per_channel")
    parser.add_argument('--act_quant', type=str, default="per_token")
    parser.add_argument("--quant", action="store_true", default=False)
    parser.add_argument("--fc2_act_log2_quant", action="store_true", default=False)
    parser.add_argument("--quant_kv", action="store_true", default=False)

    parser.add_argument("--activation_fp_quant", action="store_true", default=False)
    parser.add_argument("--weight_fp_quant", action="store_true", default=False)
    parser.add_argument('--act_fp_type', type=str, default="fp_e2")
    parser.add_argument('--weight_fp_type', type=str, default="fp_e2")
    parser.add_argument('--fc2_fp_type', type=str, default="fp_e1m2_neg_e2m1_pos")

    args = parser.parse_args()

    MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}


    # download checkpoint
    vae_ckpt = '/home/rjwei/Data_raid/huggingface/models/var/vae_ch160v4096z32.pth'
    var_ckpt = f'/home/rjwei/Data_raid/huggingface/models/var/var_d{MODEL_DEPTH}.pth'

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
    vae.eval().to(device) 
    var.eval().to(device) 
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # torch.save(var.blocks[0].attn.mat_qkv.weight, "debug/mat_qkv_after_rotate.pt")
    # pdb.set_trace()

    # quantize model

    if args.quant == True:
        var = quantize_VAR(
                var,
                weight_quant=args.weight_quant,
                act_quant=args.act_quant,
                quantize_bmm_input=False,
                w_bit=args.w_bit,
                a_bit=args.a_bit,
                act_quant_sym=args.act_sym,
                fc2_act_log2_quant=args.fc2_act_log2_quant,
                quant_kv=args.quant_kv,
                kv_bit=args.kv_bit,
                activation_fp_quant=args.activation_fp_quant,
                weight_fp_quant=args.weight_fp_quant,
                act_fp_type = args.act_fp_type,
                weight_fp_type = args.weight_fp_type,
                fc2_fp_type = args.fc2_fp_type
            )  


        # var = quantize_VAR_mixed_fp4_datatype(
        #         var,
        #         weight_quant=args.weight_quant,
        #         act_quant=args.act_quant,
        #         quantize_bmm_input=False,
        #         w_bit=args.w_bit,
        #         a_bit=args.a_bit,
        #         act_quant_sym=args.act_sym,
        #         fc2_act_log2_quant=args.fc2_act_log2_quant,
        #         quant_kv=args.quant_kv,
        #         kv_bit=args.kv_bit,
        #         activation_fp_quant=args.activation_fp_quant,
        #         weight_fp_quant=args.weight_fp_quant,
        #         act_fp_type = args.act_fp_type,
        #         weight_fp_type = args.weight_fp_type,
        #         fc2_fp_type = args.fc2_fp_type
        #     )  


        # var = quantize_VAR_mixed_fp6_datatype(
        #         var,
        #         weight_quant=args.weight_quant,
        #         act_quant=args.act_quant,
        #         quantize_bmm_input=False,
        #         w_bit=args.w_bit,
        #         a_bit=args.a_bit,
        #         act_quant_sym=args.act_sym,
        #         fc2_act_log2_quant=args.fc2_act_log2_quant,
        #         quant_kv=args.quant_kv,
        #         kv_bit=args.kv_bit,
        #         activation_fp_quant=args.activation_fp_quant,
        #         weight_fp_quant=args.weight_fp_quant,
        #         act_fp_type = args.act_fp_type,
        #         weight_fp_type = args.weight_fp_type,
        #         fc2_fp_type = args.fc2_fp_type
        #     )  
        

        var = var.half()

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'log/{current_time}_fp4_quant_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}_fc2_asym.txt'
    with open(file_name, 'w') as f:
        print(var, file=f)
    
    # pdb.set_trace()

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

    # # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')


    num_img_per_class = 50

    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp16' 

    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_e3m0_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_e2m1_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_debug' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_w_e2m1_a_e3m0_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 

    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_e2m1_fc2_special_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_e2m1_fc2_special_method2_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_e2m1_fc2_fp4_afpq_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    
    # mixed format
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_different_type_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_different_type_fc2_special_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp_quant_mixed_fp6_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 

    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_rotate_fp_quant_e2m1_fc2_special_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_block_rotate_fp_quant_e2m1_fc2_special_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 

    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp6_quant_e2m3_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp6_quant_e3m2_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp6_quant_w_e2m3_a_e3m2_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 
    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp6_quant_w_e3m2_a_e2m3_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 

    # save_file_path = f'/home/rjwei/Data_raid/Q-VAR/evaluate_figs_fp6_quant_fc2_special_w_e3m2_a_e2m3_w{args.w_bit}_{args.weight_quant}_a{args.a_bit}_{args.act_quant}_kv{args.kv_bit}' 

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    else:
        pass

    for i in range(1000):
        print(i)
        class_labels = [i for _ in range(num_img_per_class)]
        B = len(class_labels)

        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=1.5, top_k=900, top_p=0.96, 
                                                          g_seed=seed, more_smooth=False,
                                                          quant_KV=args.quant_kv, kv_bit=args.kv_bit)

        # pdb.set_trace()

        for j in range(num_img_per_class):
            print(j)
            img_tensor = recon_B3HW[j].permute(1, 2, 0).mul(255).cpu().numpy()
            chw = PImage.fromarray(img_tensor.astype(np.uint8))

            chw.save(f'{save_file_path}/class{i}_img{j}.png')



