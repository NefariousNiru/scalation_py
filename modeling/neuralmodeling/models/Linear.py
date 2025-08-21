import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modeling.neuralmodeling.layers.auxiliary import TATLinear
from modeling.neuralmodeling.layers.RevIN import RevIN


def visualize_weights(self, weights, title):

    print("Weight matrix shape:", weights.shape)
    row_norms = torch.norm(weights, dim=1)
    matrix_norm = torch.norm(weights)
    rank = torch.linalg.matrix_rank(weights)
    '''print("====== Weight Matrix Diagnostics ======")
    print(f"Matrix Norm         : {matrix_norm:.4f}")
    print(f"Matrix Rank         : {rank}/{weights.shape[0]}")
    print(f"Mean Row Norm       : {row_norms.mean():.4f}")
    print(f"Std Dev of Row Norm : {row_norms.std():.4f}")'''

    plt.figure(figsize=(8, 4))
    sns.heatmap(weights.numpy(), cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Input Feature Index")
    plt.ylabel("Output Step")
    plt.tight_layout()
    plt.show()

    '''plt.figure(figsize=(8, 3))
    plt.plot(row_norms.numpy(), marker='o')
    plt.title("Row-wise Weight Norms (Output Specialization)")
    plt.xlabel("Output Step")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''
    
    '''
    row_norms = torch.norm(weights, dim=1, keepdim=True) 
    col_norms = torch.norm(weights, dim=0, keepdim=True) 

    plt.figure(figsize=(6, 4))
    sns.heatmap(row_norms.numpy(), cmap="viridis", cbar=True)
    plt.title(f"{title} - Row-wise Norms")
    plt.xlabel("Norm Value")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 2))
    sns.heatmap(col_norms.numpy(), cmap="magma", cbar=True)
    plt.title(f"{title} - Column-wise Norms")
    plt.ylabel("Norm Value")
    plt.xlabel("Column Index")
    plt.tight_layout()
    plt.show()'''

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin = configs.revin
        self.subtract_last = configs.subtract_last
        if self.revin: self.revin_layer = RevIN(7, affine=False, subtract_last=False)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.target_idx = configs.target_idx

    def forward(self, indices_x, indices_y, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False):
        if self.revin: 
            x_enc = self.revin_layer(x_enc, 'norm')
        elif self.subtract_last:
            seq_last = x_enc[:, -1:, :].detach()
            x_enc = x_enc - seq_last
        x = x_enc
        x = x.permute(0,2,1)
        x = self.Linear(x)
        x = x.permute(0,2,1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        elif self.subtract_last:
            x = x + seq_last
            
        if mode == 'test':
            weights = self.Linear.weight.detach().cpu()
            result = sum(a * b for a, b in zip(weights[20], x_enc[0,:,self.target_idx]))
            print('intputs')
            print(x_enc[0,:,self.target_idx])
            print('')
            print('weights')
            print(weights[20])
            print('')
            print(result)
            visualize_weights(self, weights, 'Linear Weights')
            
        return x