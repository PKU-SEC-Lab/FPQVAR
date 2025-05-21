import torch
import typing
import utils
import transformers
import tqdm, math
import quant_utils

def transform_mat_qkv(layer, mat_qkv_best_s):
    print("transform mat qkv......")
    W_qkv = layer.attn.mat_qkv.weight.data
    dtype = W_qkv.dtype
    W_qkv_new = W_qkv / mat_qkv_best_s
    layer.attn.mat_qkv.weight.data = W_qkv_new.to(dtype)


def transform_fc1(layer, fc1_best_s):
    print("transform fc1......")
    W_fc1 = layer.ffn.fc1.weight.data
    dtype = W_fc1.dtype
    W_fc1_new = W_fc1 / fc1_best_s
    layer.ffn.fc1.weight.data = W_fc1_new.to(dtype)


def transform_model(model, mat_qkv_best_s, fc1_best_s):
    layers = model.blocks
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Transforming")): 
        transform_mat_qkv(layer, mat_qkv_best_s[idx])
        transform_fc1(layer, fc1_best_s[idx])




