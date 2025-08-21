__all__ = ['PatchTST_backbone']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from modeling.neuralmodeling.layers.PatchTST_layers import *
from modeling.neuralmodeling.layers.RevIN import RevIN
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def visualize_weights(self, weights, title):

    print("Weight matrix shape:", weights.shape)

    # Collapse diagnostics
    row_norms = torch.norm(weights, dim=1)
    matrix_norm = torch.norm(weights)
    rank = torch.linalg.matrix_rank(weights)
    '''print("====== Weight Matrix Diagnostics ======")
    print(f"Matrix Norm         : {matrix_norm:.4f}")
    print(f"Matrix Rank         : {rank}/{weights.shape[0]}")
    print(f"Mean Row Norm       : {row_norms.mean():.4f}")
    print(f"Std Dev of Row Norm : {row_norms.std():.4f}")'''

    # --- Heatmap ---
    plt.figure(figsize=(8, 4))
    sns.heatmap(weights.numpy(), cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel("Input Feature Index")
    plt.ylabel("Output Step")
    plt.tight_layout()
    plt.show()

    # --- Row-wise Norm Plot ---
    '''plt.figure(figsize=(8, 3))
    plt.plot(row_norms.numpy(), marker='o')
    plt.title("Row-wise Weight Norms (Output Specialization)")
    plt.xlabel("Output Step")
    plt.ylabel("L2 Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

def plot_vector_norms(x_before, x_after, title):
    norm_before = torch.norm(x_before, dim=-1).detach().cpu()  # [B, T]
    norm_after = torch.norm(x_after, dim=-1).detach().cpu()    # [B, T]

    mean_before = norm_before.mean(dim=0)
    std_before = norm_before.std(dim=0)
    mean_after = norm_after.mean(dim=0)
    std_after = norm_after.std(dim=0)

    plt.figure(figsize=(10, 5))
    plt.plot(mean_before, label="Mean Before", color="blue")
    plt.fill_between(range(mean_before.shape[0]), mean_before - std_before, mean_before + std_before, color="blue", alpha=0.2)

    plt.plot(mean_after, label="Mean After", color="orange")
    plt.fill_between(range(mean_after.shape[0]), mean_after - std_after, mean_after + std_after, color="orange", alpha=0.2)

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Vector Norm")
    plt.legend()
    plt.grid(True)
    plt.show()    
# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z, mode):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z, mode)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z, mode)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            if mode == 'testt':
                before = z
            z = self.revin_layer(z, 'denorm')
            if mode == 'testt':
                after = z
                plt.plot(before[0,:,-1].cpu().numpy(), color='black', linewidth=0.5, marker='o', markersize=1, label='z before')
                plt.plot(after[0,:,-1].cpu().numpy(), color='red', linewidth=0.5, marker='o', markersize=1, label='z after')
                plt.legend()
                plt.grid(True)
                plt.show()

            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x, mode):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        
        '''if mode == 'test':
            weights = self.linear.weight.detach().cpu()  # [out_features, input_features]
            visualize_weights(self, weights, 'Linear Weights')'''
        return x
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x, mode) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u, mode)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, mode, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, mode, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, mode, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

'''
class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))  # shape: [C]
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x):
        # x shape: [B, C, L]
        if x.dim() != 3 or x.size(1) != self.num_features:
            raise ValueError(f"Expected input of shape [B, {self.num_features}, L], but got {x.shape}")

        if self.training:
            mean = x.mean(dim=(0, 2))  # over batch and time
            var = x.var(dim=(0, 2), unbiased=True)

            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean[None, :, None])

        if self.affine:
            x_norm = self.gamma[None, :, None] * x_norm + self.beta[None, :, None]

        return x_norm
'''

class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        #norm = 'LayerNrom'
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, mode, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        before = src
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, mode, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, mode, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            before = src
            src = self.norm_attn(src)
            after = src
            if mode == 'testt':
                plot_vector_norms(before, after, 'attn')

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            before = src
            src = self.norm_ffn(src)
            after = src
            if mode == 'testt':
                plot_vector_norms(before, after, 'ffn')
        after = src
        
        if mode == 'testt':
            # Define group settings
            group_size = src.shape[0]//7
            group_index = 4  # 5th group (0-based index)
            start = group_index * group_size
            end = start + group_size

            # Extract and flatten the 5th group
            before_group = before[start:end].cpu().numpy().flatten()
            after_group = after[start:end].cpu().numpy().flatten()

            # Plot distributions
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(before_group, bins=50, alpha=0.7, color='skyblue', range=(-10, 10))
            plt.title("Before Distribution (Group 5)")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            plt.subplot(1, 2, 2)
            plt.hist(after_group, bins=50, alpha=0.7, color='salmon', range=(-10, 10))
            plt.title("After Distribution (Group 5)")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()

            # Overlayed histogram
            plt.figure(figsize=(8, 5))
            plt.hist(before_group, bins=50, alpha=0.5, label='Before', range=(-10, 10))
            plt.hist(after_group, bins=50, alpha=0.5, label='After', range=(-10, 10))
            plt.title("Overlayed Group 5 Distributions")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Statistical analysis
            from scipy.stats import skew, kurtosis, ks_2samp

            print(f"Mean Before: {before_group.mean():.4f}, After: {after_group.mean():.4f}")
            print(f"Std Before: {before_group.std():.4f}, After: {after_group.std():.4f}")
            print(f"Skewness Before: {skew(before_group):.4f}, After: {skew(after_group):.4f}")
            print(f"Kurtosis Before: {kurtosis(before_group):.4f}, After: {kurtosis(after_group):.4f}")

            '''stat, p = ks_2samp(before_group, after_group)
            print(f"KS Test → Stat: {stat:.4f}, p-value: {p:.4f}")'''


        
        '''
        if mode == 'test':
            # Flatten and move to CPU for plotting
            before_flat = before.flatten().cpu()
            after_flat = after.flatten().cpu()

            # Plot side by side
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(before_flat.numpy(), bins=100, alpha=0.7, color='skyblue', range=(-10, 10))
            plt.title("Before Distribution")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            plt.subplot(1, 2, 2)
            plt.hist(after_flat.numpy(), bins=100, alpha=0.7, color='salmon', range=(-10, 10))
            plt.title("After Distribution")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()

            # --- Additional Analysis ---
            import numpy as np
            from scipy.stats import entropy, wasserstein_distance, skew, kurtosis, ks_2samp

            before_np = before_flat.numpy()
            after_np = after_flat.numpy()

            # Histograms to probability distributions
            hist_before, bins = np.histogram(before_np, bins=100, range=(-10,10), density=True)
            hist_after, _ = np.histogram(after_np, bins=100, range=(-10,10), density=True)

            hist_before += 1e-8  # avoid log(0)
            hist_after += 1e-8

            # KL Divergence
            kl_div = entropy(hist_before, hist_after)

            # JS Divergence
            js_div = 0.5 * (entropy(hist_before, 0.5*(hist_before+hist_after)) + entropy(hist_after, 0.5*(hist_before+hist_after)))

            # Wasserstein Distance
            wass_dist = wasserstein_distance(before_np, after_np)

            # Statistical moments
            mean_before, mean_after = before_np.mean(), after_np.mean()
            std_before, std_after = before_np.std(), after_np.std()
            skew_before, skew_after = skew(before_np), skew(after_np)
            kurt_before, kurt_after = kurtosis(before_np), kurtosis(after_np)

            # KS Test
            ks_stat, ks_pvalue = ks_2samp(before_np, after_np)

            # Print analysis
            print("\n--- Distribution Analysis ---")
            print(f"KL Divergence: {kl_div:.4f}")
            print(f"JS Divergence: {js_div:.4f}")
            print(f"Wasserstein Distance: {wass_dist:.4f}\n")

            print(f"Mean Before: {mean_before:.4f}, After: {mean_after:.4f}")
            print(f"Std Before: {std_before:.4f}, After: {std_after:.4f}")
            print(f"Skewness Before: {skew_before:.4f}, After: {skew_after:.4f}")
            print(f"Kurtosis Before: {kurt_before:.4f}, After: {kurt_after:.4f}\n")

            print(f"KS Statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
            if ks_pvalue < 0.05:
                print("=> Distributions are significantly different! (p < 0.05)")
            else:
                print("=> No significant difference detected (p >= 0.05)")

            # Overlay histogram plot
            plt.figure(figsize=(8, 5))
            plt.hist(before_np, bins=100, alpha=0.5, label='Before', range=(-10,10))
            plt.hist(after_np, bins=100, alpha=0.5, label='After', range=(-10,10))
            plt.legend()
            plt.title("Overlayed Distributions")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
            '''
        
        
        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, mode=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        
        '''if mode == 'test':
            weights = self.W_Q.weight.detach().cpu()
            visualize_weights(self, weights, 'W_Q')
            
            weights = self.W_K.weight.detach().cpu()
            visualize_weights(self, weights, 'W_K')
            
            weights = self.W_V.weight.detach().cpu()
            visualize_weights(self, weights, 'W_V')'''
        
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights