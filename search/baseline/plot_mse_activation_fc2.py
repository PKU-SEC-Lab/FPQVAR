import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np

# filename = "activation_mse.fc2_new"
filename = "activation_mse.fc2_new_v1"



mse =  torch.load(f"{filename}.pt")
int_mse = mse['int']
e1m2_mse = mse['e1m2']
e2m1_mse = mse['e2m1']
e3m0_mse = mse['e3m0']
e1m2_neg_e2m1_pos_mse = mse['e1m2_neg_e2m1_pos']
e1m2_neg_e3m0_pos_mse = mse['e1m2_neg_e3m0_pos']
afpq_mse = mse['afpq_e2m1_neg_e2m1_pos']
neg_reverse_quant_mse = mse['neg_reverse_quant']


e2m1_mse_mean = np.mean(e2m1_mse)
e3m0_mse_mean = np.mean(e3m0_mse)
e1m2_neg_e2m1_pos_mse_mean = np.mean(e1m2_neg_e2m1_pos_mse)
e1m2_neg_e3m0_pos_mse = np.mean(e1m2_neg_e3m0_pos_mse)
afpq_mse_mean = np.mean(afpq_mse)
neg_reverse_quant_mse = np.mean(neg_reverse_quant_mse)

print("e2m1_mse_mean", e2m1_mse_mean)
print("e3m0_mse_mean", e3m0_mse_mean)
print("e1m2_neg_e2m1_pos_mse_mean", e1m2_neg_e2m1_pos_mse_mean)
print("afpq_mse_mean", afpq_mse_mean)
print("neg_reverse_quant_mse", neg_reverse_quant_mse)

ratio = afpq_mse_mean / e1m2_neg_e2m1_pos_mse_mean
print(ratio)
# pdb.set_trace()

# Create figure with professional styling
# plt.figure(figsize=(12, 6), dpi=100)

# Custom color palette
colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot each MSE curve
plt.plot(range(len(e1m2_mse)), e1m2_mse, 
         label='E1M2 / INT4', color=colors[0], linewidth=2.5, marker='o', markersize=5)
plt.plot(range(len(e2m1_mse)), e2m1_mse, 
         label='E2M1', color=colors[1], linewidth=2.5, marker='^', markersize=5)
plt.plot(range(len(e3m0_mse)), e3m0_mse, 
         label='E3M0', color=colors[2], linewidth=2.5, marker='D', markersize=5)
plt.plot(range(len(afpq_mse)), afpq_mse, 
         label='AFPQ', color=colors[3], linewidth=2.5, marker='s', markersize=5)
plt.plot(range(len(e3m0_mse)), e1m2_neg_e2m1_pos_mse, 
         label='DFQ', color=colors[4], linewidth=2.5, marker='*', markersize=5)

# Add labels and title
plt.rcParams.update({'font.size': 14})
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.xlabel('Block Index', fontsize=14)
plt.ylabel('Quantization Error', fontsize=14)
plt.legend(fontsize=14, framealpha=1, shadow=True, loc='best')

# Save and show
plt.tight_layout()
plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)