import torch
import random
import torch.nn as nn
from einops import rearrange, repeat
from modeling.neuralmodeling.layers.Transformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer
from modeling.neuralmodeling.layers.SelfAttention_Family import FullAttention, AttentionLayer
from modeling.neuralmodeling.layers.Embed import enc_embedding
from modeling.neuralmodeling.layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.label_len = configs['label_len']
        self.pred_len = self.pred_len
        self.output_attention = configs['output_attention']

        self.enc_in = configs['enc_in']
        self.c_out = configs['c_out']

        self.patch_len = configs['patch_len']
        self.stride = configs['stride']

        self.patch_len_dec = configs['patch_len_dec']
        self.stride_dec = configs['patch_len_dec']
        self.d_model = configs['d_model']
        self.segments = int(((configs['seq_len'] - self.patch_len) / self.stride) + 1)
        self.freq = configs['freq']
        self.tf_ratio = configs['tf_ratio']
        self.revin_layer = RevIN(configs["enc_in"], affine=False, subtract_last=0)
        self.revin = configs["revin"]
        self.enc_value_embedding = enc_embedding(self.d_model, self.seq_len, configs['patch_len'], self.stride,
                                                 configs['dropout'], freq='h')

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=configs['output_attention']), configs['d_model'],
                        configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['d_model'])
        )
        self.dec_value_embedding = enc_embedding(self.d_model, self.pred_len, self.patch_len_dec, self.stride_dec,
                                                 configs['dropout'], freq='h')

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    AttentionLayer(
                        FullAttention(False, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                )
                for l in range(configs['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['d_model']),
            projection=nn.Linear(configs['d_model'], configs['patch_len_dec'])
        )


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, train=False):

        batch_size = x_enc.shape[0]
        x_enc_temp = x_enc

        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        x_seq = self.enc_value_embedding(x_enc, x_mark_enc)
        x_seq = rearrange(x_seq, 'b ts_d seg_dec_num seg_len -> (b ts_d) seg_dec_num seg_len', ts_d=self.enc_in,
                          b=batch_size)
        enc_out, attns = self.encoder(x_seq)

        decoder_input = x_enc[:, -self.patch_len_dec:, :]
        outputs = []
        for t in range(0, int(self.pred_len / self.patch_len_dec)):
            true_input = decoder_input
            dec_in = self.dec_value_embedding(decoder_input, x_mark_dec[:,
                                                             :self.label_len + t * self.patch_len_dec + self.patch_len_dec,
                                                             :])
            dec_in = rearrange(dec_in, 'b ts_d seg_dec_num d_model -> (b ts_d) seg_dec_num d_model', b=batch_size)
            dec_out = self.decoder(dec_in, enc_out)
            dec_out = rearrange(dec_out, '(b ts_d) seg_dec_num seg_len -> b (seg_dec_num seg_len) ts_d',
                                ts_d=self.c_out)
            r = random.random()
            if r < self.tf_ratio and train is True:
                decoder_input = torch.cat((true_input, x_dec[:,
                                                       self.label_len + (t * self.patch_len_dec):self.label_len + (
                                                                   t * self.patch_len_dec) + self.patch_len_dec, :]),
                                          dim=1)
            else:
                decoder_input = torch.cat((true_input, dec_out[:, -self.patch_len_dec:, :]), 1)
            outputs.append(dec_out[:, -self.patch_len_dec:, :])
        outputs = torch.cat(outputs, dim=1)

        if self.revin:
            outputs = self.revin_layer(outputs, 'denorm')
        if self.output_attention:
            return outputs, attns
        else:
            return outputs