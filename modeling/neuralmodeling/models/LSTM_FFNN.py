import torch
from torch import nn
import math
from modeling.neuralmodeling.layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs['d_model']
        self.layer_dim = configs['rnn_layer_dim']
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.c_out = configs['c_out']
        self.revin = configs['revin']
        
        if self.revin: self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)
            
        self.lstm = nn.LSTM(self.enc_in, self.d_model, self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.d_model, self.pred_len*self.c_out)
    def forward(self, indices_x, indices_y, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False):
        if self.revin: 
            x_enc = self.revin_layer(x_enc, 'norm')
            
        out, (hidden, cell) = self.lstm(x_enc)
        output = self.fc(hidden[-1])
        output = torch.reshape(output, (output.shape[0],self.pred_len, self.c_out))
        
        if self.revin: 
            output = self.revin_layer(output, 'denorm')
            
        return output