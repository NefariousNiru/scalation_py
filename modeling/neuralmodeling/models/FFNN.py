import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn 
from torch import nn, Tensor
import torch.utils.data as data_utils
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(configs['seq_len'], configs['d_model'])
        self.linear2 = nn.Linear(configs['d_model'], configs['d_model']//2)
        self.linear3 = nn.Linear(configs['d_model']//2, configs['d_model']//4)
        self.linear4 = nn.Linear(configs['d_model']//4, self.pred_len)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
    def forward(self, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, use_tf=False):
        x_enc = x_enc.permute(0,2,1)
        x_enc = self.linear1(x_enc)
        x_enc = self.act1(self.linear2(x_enc))
        x_enc = self.act2(self.linear3(x_enc))
        output = self.linear4(x_enc)
        output = output.permute(0,2,1)
        return output
    
"""
class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(configs['seq_len'], self.pred_len)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, use_tf = False):
        x_enc = x_enc.permute(0,2,1)
        output = self.linear1(x_enc)
        output = output.permute(0,2,1)
        return output
"""