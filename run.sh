# Image 256x256

# FP4 + KV Cache quantization
python evaluate_fp_quant_transform_rotate.py --quant --w_bit 4 --a_bit 4 --weight_quant per_group --act_quant per_group --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp_e2 --weight_fp_type fp_e2 --fc2_fp_type fp_e1m2_neg_e2m1_pos --rotate --block_rotate --transform --quant_kv --kv_bit 6

# FP6
python evaluate_fp_quant_transform_rotate.py --quant --w_bit 6 --a_bit 6 --weight_quant per_channel --act_quant per_token --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp6_e2m3 --weight_fp_type fp6_e2m3 --fc2_fp_type fp6_int_neg_e2m3_pos --rotate --block_rotate

# FP6 + KV Cache quantization
python evaluate_fp_quant_transform_rotate.py --quant --w_bit 6 --a_bit 6 --weight_quant per_channel --act_quant per_token --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp6_e2m3 --weight_fp_type fp6_e2m3 --fc2_fp_type fp6_int_neg_e2m3_pos --rotate --block_rotate --quant_kv --kv_bit 6


# Image 512x512

# FP4
python evaluate_fp_quant_transform_rotate_512x512.py --quant --w_bit 4 --a_bit 4 --weight_quant per_group --act_quant per_group --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp_e2 --weight_fp_type fp_e2 --fc2_fp_type fp_e1m2_neg_e2m1_pos --rotate --block_rotate --transform

# FP4 + KV Cache quantization
python evaluate_fp_quant_transform_rotate_512x512.py --quant --w_bit 4 --a_bit 4 --weight_quant per_group --act_quant per_group --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp_e2 --weight_fp_type fp_e2 --fc2_fp_type fp_e1m2_neg_e2m1_pos --rotate --block_rotate --transform --quant_kv --kv_bit 6

# FP6
python evaluate_fp_quant_transform_rotate_512x512.py --quant --w_bit 6 --a_bit 6 --weight_quant per_channel --act_quant per_token --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp6_e2m3 --weight_fp_type fp6_e2m3 --fc2_fp_type fp6_int_neg_e2m3_pos --rotate --block_rotate

# FP6 + KV Cachce quantization
python evaluate_fp_quant_transform_rotate_512x512.py --quant --w_bit 6 --a_bit 6 --weight_quant per_channel --act_quant per_token --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp6_e2m3 --weight_fp_type fp6_e2m3 --fc2_fp_type fp6_int_neg_e2m3_pos --rotate --block_rotate --quant_kv --kv_bit 6