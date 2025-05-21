import torch
from torch import nn
from functools import partial
from models_quant.basic_var import FFN, SelfAttention, AdaLNSelfAttn
import pdb
from models_quant.var import SharedAdaLin

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_group_absmax(w, n_bits, group_size):
    # w: (out_features, in_features)
    w_shape = w.shape
    w_flat = w.view(-1, group_size)
    scales = w_flat.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w_flat.div_(scales).round_().mul_(scales)
    w_hat = w_flat.view(w_shape)
    return w_hat


@torch.no_grad()
def quantize_activation_per_token_sym(t, n_bits):
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
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
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t_flat.div_(scales).round_().mul_(scales)
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
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
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


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        w_bit=8,
        a_bit=8,
        act_quant_sym=True

    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act_quant_sym = act_quant_sym

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
            if self.act_quant_sym == True:
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
            if self.act_quant_sym == True:
                self.act_quant = partial(quantize_activation_per_group_sym, n_bits=a_bit, group_size=128)
            else:
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
        w_bit=8, a_bit=8, act_quant_sym=None
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
            act_quant_sym=act_quant_sym
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=w_bit
            )
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=w_bit
            )
        elif weight_quant == "per_group":
            new_module.weight = quantize_weight_per_group_absmax(
                module.weight, n_bits=w_bit, group_size=128
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"QuantizedLinear({self.in_features}, {self.out_features}, bias={self.bias is not None},\
                weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name},\
                w_bit={self.w_bit}, a_bit={self.a_bit}, act_quant_sym={self.act_quant_sym})"


def quantize_VAR(
    model, 
    weight_quant=None, act_quant=None, quantize_bmm_input=False,
    w_bit=8, a_bit=8, act_quant_sym=None
):    
    for name, m in model.named_modules():
        if isinstance(m, FFN):
            m.fc1 = QuantizedLinear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym
            )
            # m.fc2 = QuantizedLinear.from_float(
            #     m.fc2, weight_quant=weight_quant, act_quant=act_quant,
            #     w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym
            # )
        elif isinstance(m, SelfAttention):
            m.mat_qkv = QuantizedLinear.from_float(
                m.mat_qkv, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym
            )
            m.proj = QuantizedLinear.from_float(
                m.proj, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym
            )
        # elif isinstance(m, AdaLNSelfAttn):
        #     m.ada_lin[1] = QuantizedLinear.from_float(
        #         m.ada_lin[1], weight_quant=weight_quant, act_quant=act_quant,
        #         w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym
        #     )
        elif isinstance(m, SharedAdaLin): # 512x512
            m = QuantizedLinear.from_float(
                m, weight_quant=weight_quant, act_quant=act_quant,
                w_bit=w_bit, a_bit=a_bit, act_quant_sym=act_quant_sym
            )

    return model
