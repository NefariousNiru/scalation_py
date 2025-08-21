import torch as t
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, basis_function: nn.Module,
                 layers: int, layer_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, layer_size)] +
            [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        x = x.flip(dims=(1,))
        for layer in self.layers:
            x = t.relu(layer(x))
        theta = self.basis_parameters(x)
        return self.basis_function(theta)


# ===== BASIS FUNCTIONS =====
class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        p = degree_of_polynomial + 1
        self.p = p
        backcast_grid = np.stack([np.power(np.arange(backcast_size) / backcast_size, i) for i in range(p)])
        forecast_grid = np.stack([np.power(np.arange(forecast_size) / forecast_size, i) for i in range(p)])
        self.backcast_basis = nn.Parameter(t.tensor(backcast_grid, dtype=t.float32), requires_grad=False)
        self.forecast_basis = nn.Parameter(t.tensor(forecast_grid, dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast_coeffs = theta[:, self.p:]
        forecast_coeffs = theta[:, :self.p]
        backcast = t.einsum('bp,pt->bt', backcast_coeffs, self.backcast_basis)
        forecast = t.einsum('bp,pt->bt', forecast_coeffs, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        freq = np.arange(1, harmonics + 1)

        def fourier(time, f):
            return np.concatenate([np.cos(2 * np.pi * f * time), np.sin(2 * np.pi * f * time)], axis=0)

        t_b = np.arange(backcast_size) / forecast_size
        t_f = np.arange(forecast_size) / forecast_size

        self.backcast_template = nn.Parameter(
            t.tensor(np.stack([fourier(t_b, f) for f in freq], axis=0).reshape(2 * harmonics, backcast_size),
                     dtype=t.float32), requires_grad=False)

        self.forecast_template = nn.Parameter(
            t.tensor(np.stack([fourier(t_f, f) for f in freq], axis=0).reshape(2 * harmonics, forecast_size),
                     dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        p = theta.shape[1] // 2
        backcast = t.einsum('bp,pt->bt', theta[:, :p], self.backcast_template)
        forecast = t.einsum('bp,pt->bt', theta[:, p:], self.forecast_template)
        return backcast, forecast


# ===== N-BEATS MODEL FOR PATCHTST STYLE =====
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, 'individual', True)
        self.enc_in = configs.enc_in
        print('configs.basis_type ', configs.basis_type)
        if configs.basis_type == 'trend':
            def make_block():
                return NBeatsBlock(
                    input_size=self.seq_len,
                    theta_size=2 * (configs.degree_of_polynomial + 1),
                    basis_function=TrendBasis(configs.degree_of_polynomial, self.seq_len, self.pred_len),
                    layers=configs.n_layers,
                    layer_size=configs.layer_size
                )
        elif configs.basis_type == 'seasonality':
            def make_block():
                return NBeatsBlock(
                    input_size=self.seq_len,
                    theta_size=4 * configs.harmonics,
                    basis_function=SeasonalityBasis(configs.harmonics, self.seq_len, self.pred_len),
                    layers=configs.n_layers,
                    layer_size=configs.layer_size
                )
        else:
            raise ValueError(f"Unsupported basis_type: {configs.basis_type}")

        if self.individual:
            self.models = nn.ModuleList([
                NBeats(nn.ModuleList([make_block() for _ in range(configs.n_blocks)]))
                for _ in range(self.enc_in)
            ])
        else:
            shared = NBeats(nn.ModuleList([make_block() for _ in range(configs.n_blocks)]))
            self.models = nn.ModuleList([shared for _ in range(self.enc_in)])

    def forward(self, indices_x, indices_y, x_enc, trend_x_short, trend_x_long, x_mark_enc, dec_inp, x_dec, x_mark_dec, mode, l1_trend, use_tf=False):
        mask = None
        x = x_enc
        B, L, C = x.shape
        outputs = []

        for i in range(C):
            xi = x[:, :, i]  # [B, L]
            mi = mask if mask is not None else t.ones_like(xi)
            yi = self.models[i](xi, mi)  # [B, pred_len]
            outputs.append(yi.unsqueeze(-1))  # [B, pred_len, 1]

        return t.cat(outputs, dim=-1)  # [B, pred_len, C]


class NBeats(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  # initialize with last known value
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast
