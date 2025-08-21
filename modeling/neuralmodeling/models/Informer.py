import torch
import torch.nn as nn
import torch.nn.functional as F
from util.masking import TriangularCausalMask, ProbMask
from modeling.neuralmodeling.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from modeling.neuralmodeling.layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from modeling.neuralmodeling.layers.Embed import DataEmbedding
import numpy as np
from modeling.neuralmodeling.layers.RevIN import RevIN

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = self.pred_len
        self.output_attention = configs['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(configs['enc_in'], configs['d_model'], configs['embed'], configs['freq'],
                                           configs['dropout'])
        self.dec_embedding = DataEmbedding(configs['dec_in'], configs['d_model'], configs['embed'], configs['freq'],
                                           configs['dropout'])
        self.revin = configs["enc_in"]
        self.revin_layer = RevIN(configs["enc_in"], affine=False, subtract_last=0)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=configs['output_attention']),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            [
                ConvLayer(
                    configs['d_model']
                ) for l in range(configs['e_layers'] - 1)
            ] if configs['distil'] else None,
            norm_layer=torch.nn.LayerNorm(configs['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs['factor'], attention_dropout=configs['dropout'], output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    AttentionLayer(
                        ProbAttention(False, configs['factor'], attention_dropout=configs['dropout'], output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                )
                for l in range(configs['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['d_model']),
            projection=nn.Linear(configs['d_model'], configs['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, dec_inp, x_dec, x_mark_dec, use_tf=False):

        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]