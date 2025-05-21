import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np

# filename = "activation_mse.mat_qkv"
# filename = "activation_mse.proj"
# filename = "activation_mse.fc1"

# filename = "activation_mse.rotate_mat_qkv"
filename = "activation_mse.rotate_fc1"


mse =  torch.load(f"{filename}.pt")
int_mse = mse['int']
e1m2_mse = mse['e1m2']
e2m1_mse = mse['e2m1']
e3m0_mse = mse['e3m0']

e2m1_mse_mean = np.mean(e2m1_mse)
e3m0_mse_mean = np.mean(e3m0_mse)

print("e2m1_mse_mean", e2m1_mse_mean)
print("e3m0_mse_mean", e3m0_mse_mean)

e2m1_indices = torch.where(torch.tensor(e2m1_mse) <= torch.tensor(e3m0_mse))[0].tolist()
e3m0_indices = torch.where(torch.tensor(e3m0_mse) < torch.tensor(e2m1_mse))[0].tolist()

# pdb.set_trace()

# Create figure with professional styling
# plt.figure(figsize=(12, 6), dpi=100)

# Custom color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot each MSE curve
# plt.plot(range(len(int_mse)), int_mse, 
#          label='INT4', color=colors[0], linewidth=2.5, marker='s', markersize=5)
plt.plot(range(len(e1m2_mse)), e1m2_mse, 
         label='E1M2 / INT4', color=colors[1], linewidth=2.5, marker='o', markersize=5)
plt.plot(range(len(e2m1_mse)), e2m1_mse, 
         label='E2M1', color=colors[2], linewidth=2.5, marker='^', markersize=5)
plt.plot(range(len(e3m0_mse)), e3m0_mse, 
         label='E3M0', color=colors[3], linewidth=2.5, marker='D', markersize=5)

# Add labels and title
plt.xlabel('Block Index', fontsize=12)
plt.ylabel('Quantization Error', fontsize=12)
plt.legend(fontsize=12, framealpha=1, shadow=True, loc='best')

# Save and show
plt.tight_layout()
plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)