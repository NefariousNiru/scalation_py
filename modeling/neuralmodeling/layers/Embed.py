import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm
import math
import ruptures as rpt
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import numpy as np
import pyscamp
class DigitEmbedding(nn.Module):
    def __init__(self, max_digits=32, embedding_dim=16):
        """
        Digit-based embedding module to handle signs, digits, and decimal points.
        
        Args:
            max_digits (int): Maximum number of digits to represent a number.
            embedding_dim (int): Dimension for each digit's embedding.
        """
        super(DigitEmbedding, self).__init__()
        self.max_digits = max_digits
        self.embedding_dim = embedding_dim
        
        # Embedding layers for digits, sign, and decimal point
        self.digit_embedding = nn.Embedding(10, embedding_dim)  # Embedding for digits 0-9
        self.sign_embedding = nn.Embedding(3, embedding_dim)  # Embedding for signs: {0: positive, 1: negative}
        self.decimal_embedding = nn.Embedding(1, embedding_dim)  # Embedding for decimal point
        self.positional_encoding = nn.Embedding(max_digits + 2, embedding_dim)  # Positional encoding

    def forward(self, x):
        """
        Forward pass for digit-based embedding.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, seq_len, 1].
        
        Returns:
            Tensor: Embedded tensor of shape [batch_size, channels, seq_len, embedding_dim].
        """
        batch_size, channels, seq_len, _ = x.shape
        x_flattened = x.view(-1)  # Flatten for easier processing

        digit_list = []
        position_list = []
        sign_list = []

        for number in x_flattened:
            is_negative = number.item() < 0
            abs_number = abs(number.item())

            # Convert number to string with a maximum of max_digits
            str_number = f"{abs_number:.{self.max_digits}g}"

            # Extract components
            digits = []
            positions = []
            if is_negative:
                sign_list.append(1)  # Negative
            else:
                sign_list.append(0)  # Positive

            for i, char in enumerate(str_number.zfill(self.max_digits)):
                if char.isdigit():
                    digits.append(int(char))
                    positions.append(i)
                elif char == '.':
                    digits.append(0)  # Placeholder for the decimal
                    positions.append(i)

            # Pad digits and positions to max_digits
            digits = digits[:self.max_digits] + [0] * (self.max_digits - len(digits))
            positions = positions[:self.max_digits] + [0] * (self.max_digits - len(positions))
            digit_list.append(digits)
            position_list.append(positions)

        # Convert to tensors
        digits_tensor = torch.tensor(digit_list, device=x.device)  # Shape: [batch_size * seq_len, max_digits]
        positions_tensor = torch.tensor(position_list, device=x.device)  # Shape: [batch_size * seq_len, max_digits]
        signs_tensor = torch.tensor(sign_list, device=x.device)  # Shape: [batch_size * seq_len]

        # Digit embeddings
        digit_embeddings = self.digit_embedding(digits_tensor)  # Shape: [batch_size * seq_len, max_digits, embedding_dim]
        
        # Positional encodings
        positional_encodings = self.positional_encoding(positions_tensor)  # Shape: [batch_size * seq_len, max_digits, embedding_dim]
        
        # Sign embeddings
        sign_embeddings = self.sign_embedding(signs_tensor).unsqueeze(1)  # Shape: [batch_size * seq_len, 1, embedding_dim]
        
        # Combine embeddings
        combined_embeddings = digit_embeddings + positional_encodings  # Add digit and positional embeddings
        combined_embeddings = combined_embeddings.sum(dim=1)  # Aggregate along the digits
        combined_embeddings = combined_embeddings + sign_embeddings.squeeze(1)  # Add sign embedding

        # Reshape back to [batch_size, channels, seq_len, embedding_dim]
        return combined_embeddings.view(batch_size, channels, seq_len, self.embedding_dim)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
    
"""
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        print(x.shape, x.permute(0, 2, 1).shape)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        print(x.shape)
        print("")
        return x
"""


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()



class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        #self.position_embedding = PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                            freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #    d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class DataEmbedding_wo_temp_five(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp_five, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, 720, 512))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)


class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model, freq, stride, dropout_embed):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.d_model = d_model
        self.linear_x_embed = nn.Linear(seg_len, d_model)
        self.dropout = nn.Dropout(p=dropout_embed)
    def forward(self, x, x_mark):
        batch, ts_len, ts_dim = x.shape
        x = x.permute(0,2,1)
        x = x.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        x = self.linear_x_embed(x)
        return self.dropout(x)

    

class DynamicPatching(nn.Module):
    def __init__(self):
        super(DynamicPatching, self).__init__()
    def forward(self, tensor, change_points_batch):
        max_segments = max(len(change_points) - 1 for change_points in change_points_batch)
        segmented_tensors = []
        for i, change_points in enumerate(change_points_batch):
            segments = [tensor[i, :, change_points[j]:change_points[j+1]] for j in range(len(change_points)-1)]
            segmented_tensors.append(segments)
        max_segments = max(len(segment) for segment in segmented_tensors)
        max_length = max(len(i) for segment in segmented_tensors for item in segment for i in item)
        segments = [[torch.nn.functional.pad(segment, (0, max_length - segment.size(1))) for segment in batch] for batch in segmented_tensors]
        for batch in segments:
            if len(batch) < max_segments:
                pad_segments = [torch.zeros_like(batch[0]) for _ in range(max_segments - len(batch))]
                batch.extend(pad_segments)
        tensor_segments = torch.stack([torch.stack(batch) for batch in segments])
        return tensor_segments
    
class enc_embedding_enc(nn.Module):
    def __init__(self, d_model, seq_len, seg_len, stride, dropout_embed, freq):
        super(enc_embedding_enc, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.d_model = d_model
        self.DynamicPatching = DynamicPatching()

        self.position_embedding = PositionalEmbeddingPatches(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbeddingPatches(d_model=d_model, freq=freq)
        self.linear = nn.AdaptiveAvgPool1d(16)

        self.dropout = nn.Dropout(p=dropout_embed)
    def forward(self, x, x_mark):
        batch, ts_len, ts_dim = x.shape
        
        pe = self.position_embedding(x)

        x = x.permute(0,2,1)
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2]).unsqueeze(1)
        
        xcp = x.cpu().numpy().squeeze()
        
        change_points_batch = []
        for i in tqdm(range(xcp.shape[0])):
            signal = xcp[i]
            algo = rpt.Pelt(model="rbf", min_size = 12).fit(signal)
            result = algo.predict(pen=10)
            result = [0] + result + [xcp.shape[-1]]
            change_points_batch.append(result)
        
        x = self.DynamicPatching(x, change_points_batch)
        x_embed = self.linear(x.squeeze())
        x_embed = rearrange(x_embed, '(b dim) seg_num d_model -> b dim seg_num d_model', b = batch, dim = 7)
        
        pe = pe.repeat(batch*7, 1, 1)
        pe = self.DynamicPatching(pe, change_points_batch)
        pe = self.linear(pe.squeeze())
        pe = rearrange(pe, '(b dim) seg_num d_model -> b dim seg_num d_model', b = batch, dim = 7)
        
        x_mark = x_mark.repeat(batch*7, 1, 1)
        time_embed = self.temporal_embedding(x_mark)
        time_embed = self.DynamicPatching(time_embed, change_points_batch)
        time_embed = self.linear(time_embed.squeeze())
        time_embed = rearrange(time_embed, '(b dim) seg_num d_model -> b dim seg_num d_model', b = batch, dim = 7)
        
        return self.dropout(x_embed + pe + time_embed)

class enc_embedding_dec(nn.Module):
    def __init__(self, d_model, seq_len, seg_len, stride, dropout_embed, freq):
        super(enc_embedding_dec, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.d_model = d_model
        self.position_embedding = PositionalEmbeddingPatches(d_model=d_model)
        self.linear_x_pe = nn.Linear(seg_len, d_model, bias = True)
        self.linear_x_embed = nn.Linear(seg_len, d_model, bias = True)
        self.temporal_embedding = TimeFeatureEmbeddingPatches(d_model=d_model, freq=freq)
        self.linear_x_temporal = nn.Linear(seg_len, d_model, bias = True)
        self.dropout = nn.Dropout(p=dropout_embed)
    def forward(self, x, x_mark):
        batch, ts_len, ts_dim = x.shape
        pe = self.position_embedding(x)
        pe = pe.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        pe = self.linear_x_pe(pe)
        x = x.permute(0,2,1)
        x = x.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        x_embed = self.linear_x_embed(x)
        time_embed = self.temporal_embedding(x_mark)
        time_embed = time_embed.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        time_embed = self.linear_x_temporal(time_embed)
        return self.dropout(x_embed + time_embed + pe)
    
class dec_embedding(nn.Module):
    def __init__(self, d_model, seq_len, seg_len, stride, dropout_embed, freq):
        super(dec_embedding, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.d_model = d_model
        if seq_len == seg_len:
            self.segments = 1
        else:
            self.segments = int(((seq_len - seg_len)/stride)+1)
        self.position_embedding = PositionalEmbeddingPatches(d_model=d_model)
        self.linear_x_pe = nn.Linear(seg_len, d_model, bias = False)
        self.linear_x_embed = nn.Conv1d(in_channels=self.seg_len, out_channels=self.d_model,
                 kernel_size=3, padding=1, padding_mode='circular')
        
        self.temporal_embedding = TimeFeatureEmbeddingPatches(d_model=d_model, freq=freq)
        self.linear_x_temporal = nn.Linear(seg_len, d_model, bias = False)
        self.dropout = nn.Dropout(p=dropout_embed)
    def forward(self, x, x_mark):
        batch, ts_len, ts_dim = x.shape
        pe = self.position_embedding(x)
        pe = pe.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        pe = self.linear_x_pe(pe)
        x = x.permute(0,2,1)
        x = x.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        x_embed = self.linear_x_embed(x)
        time_embed = self.temporal_embedding(x_mark)
        time_embed = time_embed.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        time_embed = self.linear_x_temporal(time_embed)
        return self.dropout(x_embed + time_embed + pe)
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)


class TimeFeatureEmbeddingPatches(nn.Module):
    def __init__(self, d_model, freq):
        super(TimeFeatureEmbeddingPatches, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'W': 2, 'd': 3, 'b': 3}
        self.d_inp = freq_map[freq]
        self.te = nn.Linear(self.d_inp, 1)

    def forward(self, x_mark):
        temporal_embedding = self.te(x_mark)
        return temporal_embedding.permute(0, 2, 1)


class PositionalEmbeddingPatches(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbeddingPatches, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)]
        pe = self.linear(pe).permute(0, 2, 1)
        return pe



#---------------------------------start here-----------------------------------------------------
class enc_embedding_scamp(nn.Module):
    def __init__(self, d_model, patch_len, stride, freq, max_motifs=5, max_discords=5):
        super(enc_embedding_scamp, self).__init__()
        self.d_model = d_model
        self.motif_window = patch_len
        self.max_motifs = max_motifs
        self.max_discords = max_discords
        self.linear_x_embed = nn.Linear(self.motif_window, self.d_model)

    def forward(self, x, x_mark=None):

        batch_size, seq_len, features = x.size()
        all_patches = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            time_series = x[b, :, 0].cpu().detach().numpy().astype(np.float64)

            profile, indices = pyscamp.selfjoin(time_series, m=self.motif_window)

            sorted_indices = np.argsort(profile)
            motif_indices = sorted_indices[:self.max_motifs]

            discord_indices = sorted_indices[-self.max_discords:]

            combined_indices = sorted(list(motif_indices) + list(discord_indices))

            prev_end_idx = 0
            for start_idx in combined_indices:
                if prev_end_idx < start_idx:
                    patch = time_series[prev_end_idx:start_idx]
                    all_patches[b].append(self.pad_to_window(patch))

                end_idx = start_idx + self.motif_window
                patch = time_series[start_idx:end_idx]
                all_patches[b].append(self.pad_to_window(patch))
                prev_end_idx = end_idx

            if prev_end_idx < len(time_series):
                patch = time_series[prev_end_idx:]
                all_patches[b].append(self.pad_to_window(patch))

        max_patches = max(len(patches) for patches in all_patches)
        padded_patches = []

        for patches in all_patches:
            while len(patches) < max_patches:
                patches.append(torch.zeros(self.motif_window))
            padded_patches.append(torch.stack(patches))

        padded_patches = torch.stack(padded_patches, dim=0)
        padded_patches = padded_patches.to(x.device)
        x_embed = self.linear_x_embed(padded_patches)

        x_embed = x_embed.unsqueeze(1)
        return x_embed

    def pad_to_window(self, patch):
        patch = torch.tensor(patch, dtype=torch.float32)
        if len(patch) < self.motif_window:
            pad_length = self.motif_window - len(patch)
            padded_patch = torch.nn.functional.pad(patch, (0, pad_length))
        else:
            padded_patch = patch[:self.motif_window]
        return padded_patch

class GradientBasedWindowing:
    def __init__(self, gradient_threshold, min_window_size):
        self.gradient_threshold = gradient_threshold
        self.min_window_size = min_window_size

    def forward(self, data):
        batch_size, num_features, seq_len = data.shape
        patches_list = []

        for i in range(batch_size):
            sample_patches = []
            for feature in range(num_features):
                current_patch = []
                grad_diff = torch.abs(torch.diff(data[i, feature])).cpu()

                for j in range(seq_len):
                    current_patch.append(data[i, feature, j].item())

                    if j < seq_len - 1:
                        if grad_diff[j].item() > self.gradient_threshold and len(current_patch) >= self.min_window_size:
                            # Create a new patch if the gradient threshold is exceeded
                            sample_patches.append(torch.tensor(current_patch).unsqueeze(0))  # Shape: [1, patch_len]
                            current_patch = []  # Reset current patch

                # Append any remaining values in the current patch
                if len(current_patch) > 0:
                    sample_patches.append(torch.tensor(current_patch).unsqueeze(0))

            patches_list.append(sample_patches)
        
        return patches_list  # Each sample in batch has a list of feature patches


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, device):
        super(PatchEmbedding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.conv1d = nn.Conv1d(1, d_model, kernel_size=3, stride=1, padding=0).to(device)
        self.pool = nn.AdaptiveAvgPool1d(1).to(device)
        self.linear = nn.Linear(2, d_model).to(device)  # Linear layer for very short patches

    def forward(self, patches):
        max_patch_len = max(patch.size(1) for patch in patches)
        embedded_patches = []

        for patch in patches:
            patch_len = patch.size(1)

            # Pad the patch to max_patch_len if needed
            if patch_len < max_patch_len:
                pad_amount = max_patch_len - patch_len
                patch = nn.functional.pad(patch, (0, pad_amount))

            # Reshape to [1, 1, patch_length] for conv1d
            patch = patch.unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: [1, 1, patch_len]

            # Apply conv1d if patch length >= kernel size, otherwise use linear or zero embedding
            if patch.size(2) >= 3:
                patch_embedding = self.conv1d(patch)
                patch_embedding = self.pool(patch_embedding).squeeze(-1)  # Pool to [1, d_model]
            elif patch.size(2) == 2:
                patch_embedding = self.linear(patch.squeeze(0))
            else:
                patch_embedding = torch.zeros(self.d_model).to(self.device)

            embedded_patches.append(patch_embedding.squeeze(0))  # Shape: [d_model]

        return torch.stack(embedded_patches)  # Shape: [num_patches, d_model]


class enc_embedding_dynamic(nn.Module):
    def __init__(self, d_model, patch_len, stride, freq):
        super(enc_embedding_dynamic, self).__init__()
        gradient_threshold = 0.2
        min_window_size = 10
        self.d_model = d_model
        self.device = 'cuda:0'
        self.position_embedding = PositionalEmbeddingPatches(d_model=self.d_model)
        self.temporal_embedding = TimeFeatureEmbeddingPatches(d_model=self.d_model, freq=freq)
        self.patch_embedding = PatchEmbedding(d_model=self.d_model, device=self.device)
        self.windowing = GradientBasedWindowing(gradient_threshold, min_window_size)

    def forward(self, x, x_mark):
        pe = self.position_embedding(x)
        te = self.temporal_embedding(x_mark)
        x = x.permute(0,2,1) + pe + te

        batch_size, num_features, _ = x.shape
        all_embedded_samples = []

        # Determine global max_patches across all features and samples
        global_max_patches = 0
        feature_patches_lists = [[] for _ in range(num_features)]

        # First pass: Collect patches and find the global max_patches
        for feature_idx in range(num_features):
            for sample_idx in range(batch_size):
                # Get gradient-based patches for each feature in the batch
                patches = self.windowing.forward(x[sample_idx:sample_idx + 1, feature_idx:feature_idx + 1, :])
                feature_patches_lists[feature_idx].append(patches[0])  # Append patches for this feature
                global_max_patches = max(global_max_patches, len(patches[0]))  # Update global max_patches

        # Second pass: Embed and pad patches to global_max_patches
        for feature_idx in range(num_features):
            embedded_feature_samples = []
            for patches in feature_patches_lists[feature_idx]:
                embedded_patches = self.patch_embedding(patches)  # Shape: [num_patches, d_model]

                # Pad to global_max_patches if necessary
                if embedded_patches.size(0) < global_max_patches:
                    padding = torch.zeros(global_max_patches - embedded_patches.size(0), self.d_model,
                                          device=self.device)
                    embedded_patches = torch.cat((embedded_patches, padding),
                                                 dim=0)  # Shape: [global_max_patches, d_model]

                embedded_feature_samples.append(embedded_patches)

            # Stack all embedded patches for this feature across batch
            embedded_feature_tensor = torch.stack(
                embedded_feature_samples)  # Shape: [batch_size, global_max_patches, d_model]
            all_embedded_samples.append(embedded_feature_tensor)

        # Stack all features into final tensor shape
        final_output = torch.stack(all_embedded_samples,
                                   dim=1)  # Shape: [batch_size, num_features, global_max_patches, d_model]
        return final_output
    
class encembed_scamp(nn.Module):
    def __init__(self, d_model, patch_len, stride, freq, k_neighbors=3):
        super(encembed_scamp, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.linear_x_embed = nn.Linear(self.patch_len, self.d_model)

    def forward(self, x, x_mark=None):

        batch_size, seq_len, channels = x.size()

        # Define a function to compute KNN for a single sequence
        def compute_knn_for_sequence(time_series, patch_len, k_neighbors):
            knn_results = pyscamp.selfjoin_knn(time_series, m=patch_len, k=k_neighbors)
            return [result[1] for result in knn_results[:k_neighbors]]

        # Parallelize KNN computation across the batch
        x_cpu = x[:, :, 0].cpu().detach().numpy()  # Move data to CPU for KNN computation
        neighbor_indices_batch = [
            compute_knn_for_sequence(x_cpu[b], self.patch_len, self.k_neighbors)
            for b in range(batch_size)
        ]

        # Create patches using gathered indices
        selected_patches = []
        for b, neighbor_indices in enumerate(neighbor_indices_batch):
            for idx in neighbor_indices:
                # Calculate start and end indices for the patch
                start = max(0, idx - self.patch_len // 2)
                end = start + self.patch_len

                # Ensure indices do not exceed sequence boundaries
                if end > seq_len:
                    end = seq_len
                    start = max(0, end - self.patch_len)

                # Extract the patch from the tensor
                patch = x[b, start:end, :]

                # Pad if the patch is smaller than patch_len
                if patch.size(0) < self.patch_len:
                    pad_len = self.patch_len - patch.size(0)
                    patch = torch.nn.functional.pad(patch, (0, 0, 0, pad_len))  # Pad along seq_len dimension

                selected_patches.append(patch)

        # Combine patches into a single tensor
        num_patches = len(selected_patches) // batch_size
        selected_patches = torch.stack(selected_patches, dim=0).view(batch_size, num_patches, self.patch_len, channels)

        # Permute for embedding layer input: [batch_size, channels, num_patches, patch_len]
        selected_patches = selected_patches.permute(0, 3, 1, 2)

        # Linear embedding transformation
        x_embed = self.linear_x_embed(selected_patches)
        return x_embed

class enc_embedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, freq):
        super(enc_embedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.freq = freq
        self.linear_x_embed = nn.Linear(self.patch_len, self.d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_embed = self.linear_x_embed(x)
        return x_embed

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)




class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
    
    
class PatchEmbed(nn.Module):
    def __init__(self, args, num_p=1, d_model=None):
        super(PatchEmbed, self).__init__()
        self.num_p = num_p
        self.patch = args.seq_len // self.num_p
        self.d_model = args.d_model if d_model is None else d_model

        self.proj = nn.Sequential(
            nn.Linear(self.patch, self.d_model, False),
            nn.Dropout(args.dropout)
        )

    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, -2)
        x = self.proj(x.reshape(*x.shape[:-1], self.num_p, self.patch))
        return x