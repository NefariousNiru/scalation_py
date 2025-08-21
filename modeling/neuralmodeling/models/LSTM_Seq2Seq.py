import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn 
from torch import nn, Tensor
import torch.utils.data as data_utils
import random
from modeling.neuralmodeling.layers.RevIN import RevIN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMEncoder(nn.Module):
    def __init__(self, hidden_dim, layer_dim, seq_len, pred_len, enc_in, c_out):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(enc_in, hidden_dim, layer_dim, batch_first=True)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        return hidden, cell
    
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, layer_dim, seq_len, pred_len, enc_in, c_out):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(enc_in, hidden_dim, layer_dim, batch_first=True)
        self.fc4 = nn.Linear(hidden_dim, c_out)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        out_c_out = self.fc4(hidden[-1])
        return out_c_out, hidden, cell
    
class Model(nn.Module):    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs['d_model']
        self.layer_dim = configs['rnn_layer_dim']
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.c_out = configs['c_out']
        self.tf_ratio = configs['tf_ratio']
        
        self.revin = configs['revin']
        
        if self.revin: self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)
        
        self.encoder = LSTMEncoder(self.d_model, self.layer_dim, self.seq_len, self.pred_len, self.enc_in, self.c_out)
        self.decoder = LSTMDecoder(self.d_model, self.layer_dim, self.seq_len, self.pred_len, self.enc_in, self.c_out)
        
    def forward(self, indices_x, indices_y, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, cycle_index, mode, use_tf=False):
        if self.revin: 
            x_enc = self.revin_layer(x_enc, 'norm')
            
        src = x_enc
        trg = x_dec
        hidden, cell = self.encoder(x_enc)
        outputs = torch.zeros(trg.shape[0], self.pred_len, self.c_out).to(device)
        input_trg = src[:,src.shape[1]-1:src.shape[1],:]
        start = 0
        end = 1
        if(use_tf == True):            
            for t in range(0, self.pred_len):
                out_c_out, hidden, cell = self.decoder(input_trg, hidden, cell)
                outputs[:,start:end,:] = out_c_out.unsqueeze(1)
                if random.random() < self.tf_ratio:
                    input_trg = trg[:,start:end,:]
                else:
                    input_trg = out_c_out.unsqueeze(1)
                start = end 
                end = end + 1
        elif(use_tf == False):
            for t in range(0, self.pred_len):
                out_c_out, hidden, cell = self.decoder(input_trg, hidden, cell)
                outputs[:,start:end,:] = out_c_out.unsqueeze(1)
                input_trg = out_c_out.unsqueeze(1)
                start = end 
                end = end + 1
                
                      
        if self.revin: 
            output = self.revin_layer(outputs, 'denorm')
            
        return outputs