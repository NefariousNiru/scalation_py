import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm
import math
import ruptures as rpt
from tqdm import tqdm
from joblib import Parallel, delayed
#from modeling.neuralmodeling.layers.sigmaReparam import nn.Linear

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
    def __init__(self, seg_len, stride, d_model, freq):
        super(TimeFeatureEmbeddingPatches, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'W': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.temp1 = nn.Linear(seg_len, d_model)
        self.temp2 = nn.Linear(seg_len, d_model)
        self.temp3 = nn.Linear(seg_len, d_model)
        self.temp4 = nn.Linear(seg_len, d_model)
        self.temp5 = nn.Linear(seg_len, d_model)
        self.temp6 = nn.Linear(seg_len, d_model)
        self.seg_len = seg_len
        self.stride = stride

    def forward(self, x):
        temp1 = x[:,:,0].unfold(dimension = -1, size = self.seg_len, step = self.stride)
        temp2 = x[:,:,1].unfold(dimension = -1, size = self.seg_len, step = self.stride)
        temp3 = x[:,:,2].unfold(dimension = -1, size = self.seg_len, step = self.stride)
        temp4 = x[:,:,3].unfold(dimension = -1, size = self.seg_len, step = self.stride)
        temp5 = x[:,:,4].unfold(dimension = -1, size = self.seg_len, step = self.stride)
        temp6 = x[:,:,5].unfold(dimension = -1, size = self.seg_len, step = self.stride)
        temp1 = self.temp1(temp1).unsqueeze(1)
        temp2 = self.temp2(temp2).unsqueeze(1)
        temp3 = self.temp3(temp3).unsqueeze(1)
        temp4 = self.temp4(temp4).unsqueeze(1)
        temp5 = self.temp5(temp5).unsqueeze(1)
        temp6 = self.temp6(temp6).unsqueeze(1)

        return temp1, temp2, temp3, temp4, temp5, temp6

class PositionalEmbeddingPatches(nn.Module):
    def __init__(self, seg_len, stride, d_model, max_len=5000):
        super(PositionalEmbeddingPatches, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
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
        pe = self.linear(pe).permute(0,2,1)
        return pe
    
class TempEmbedding(nn.Module):
    def __init__(self, frequency, seg_len, stride, d_model):
        super(TempEmbedding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.seg_len = seg_len
        self.stride = stride
        self.linear = nn.Linear(2, 1)
        self.frequency = frequency
    def forward(self, hour_indices):
        batch_size, sequence_length, _ = hour_indices.shape
        max_hour = self.frequency
        hour_indices = hour_indices.float()
        radians = (hour_indices)
        d_half = self.d_model // 2
        #sin_embeddings = torch.sin(radians)
        #cos_embeddings = torch.cos(radians)
        #cyclical_embeddings = torch.cat((sin_embeddings, cos_embeddings), dim=-1)
        cyclical_embeddings = radians.permute(0,2,1)
        return cyclical_embeddings

"""class enc_embedding(nn.Module):
    def __init__(self, d_model, seq_len, seg_len, stride, dropout_embed, freq):
        super(enc_embedding, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.d_model = d_model
        self.linear = nn.Linear(seg_len, d_model)
        self.dropout = nn.Dropout(p=dropout_embed)
        
    def forward(self, x, x_mark):
        batch, ts_len, ts_dim = x.shape
        x = x.permute(0,2,1)
        x = x.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        x_embed = self.linear(x)
        return self.dropout(x_embed)"""
    

class enc_embedding(nn.Module):
    def __init__(self, d_model, seq_len, seg_len, stride, dropout_embed, freq):
        super(enc_embedding, self).__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.d_model = d_model
        self.position_embedding = PositionalEmbeddingPatches(seg_len=seg_len, stride=stride, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbeddingPatches(seg_len=seg_len, stride=stride, d_model=d_model, freq=freq)
        self.linear_x_embed = nn.Linear(seg_len, d_model, bias = True)
        self.linear_temporal = nn.Linear(2,1)
        self.dropout = nn.Dropout(p=dropout_embed)
    def forward(self, x, x_mark):
        batch, ts_len, ts_dim = x.shape
        pe = self.position_embedding(x)        
        x = x.permute(0,2,1)
        temp = self.linear_temporal(x_mark)
        temp = temp.permute(0,2,1)
        x = x + pe + temp
        x = x.unfold(dimension = -1, size = self.seg_len, step = self.stride)
        x_embed = self.linear_x_embed(x)
        return self.dropout(x_embed)