import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_weights(self, weights, title):

    print("Weight matrix shape:", weights.shape)

    # Collapse diagnostics
    row_norms = torch.norm(weights, dim=1)
    matrix_norm = torch.norm(weights)
    rank = torch.linalg.matrix_rank(weights)
    print("====== Weight Matrix Diagnostics ======")
    print(f"Matrix Norm         : {matrix_norm:.4f}")
    print(f"Matrix Rank         : {rank}/{weights.shape[0]}")
    print(f"Mean Row Norm       : {row_norms.mean():.4f}")
    print(f"Std Dev of Row Norm : {row_norms.std():.4f}")

    # --- Heatmap ---
    plt.figure(figsize=(8, 4))
    #, vmin=-1, vmax=1
    sns.heatmap(weights.numpy(), cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Input Feature Index")
    plt.ylabel("Output Step")
    plt.tight_layout()
    plt.show()

    # --- Row-wise Norm Plot ---
    plt.figure(figsize=(8, 3))
    plt.plot(row_norms.numpy(), marker='o')
    plt.title("Row-wise Weight Norms (Output Specialization)")
    plt.xlabel("Output Step")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.subtract_last = configs['subtract_last']
        self.Linear = nn.Linear(self.seq_len, self.pred_len, bias = True)
        #self.activation = nn.GELU()
        #self.layer_norm = nn.LayerNorm(self.pred_len)
        c_in = configs['enc_in']
        self.target_idx = configs['target_idx']
        context_window = configs['seq_len']
        target_window = configs['pred_len']

    def forward(self, indices_x, indices_y, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False, count=-1):


        x = x_enc

        if self.subtract_last:
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        forecasts = x
        if self.subtract_last:
            x = x + seq_last

        return x