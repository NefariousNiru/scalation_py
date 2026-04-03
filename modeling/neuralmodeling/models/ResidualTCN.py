"""
__author__ = "Nirupom Bose Roy"
"""

__all__ = ["ResidualTCN"]

import torch
import torch.nn as nn

# ============================================================
# Helpers
# ============================================================


class Chomp1d(nn.Module):
    """
    Remove the extra right-padding introduced to emulate causal convolutions.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Standard residual TCN block with causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        out = out + residual
        out = self.norm(out)
        return out


class TCNBackbone(nn.Module):
    """
    Dilated causal convolution stack.
    """

    def __init__(
        self,
        c_in: int,
        hidden_channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        blocks = []
        in_channels = c_in

        for i in range(levels):
            dilation = 2**i
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels

        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# Public model
# ============================================================


class Model(nn.Module):
    """
    Residual TCN forecaster that plugs into the existing Experiment interface.

    Expected use
    ------------
    Train this model on a residual target series, e.g.

        residual_t = y_t - yhat_sarimax_t

    Then outside the model, combine:
        final_forecast = sarimax_forecast + residual_tcn_forecast
    """

    def __init__(self, configs, **kwargs) -> None:
        super().__init__()

        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]

        self.c_in = configs["enc_in"]
        self.c_out = configs["c_out"]

        self.hidden_channels = configs.get("tcn_hidden", 64)
        self.levels = configs.get("tcn_levels", 5)
        self.kernel_size = configs.get("tcn_kernel_size", 3)
        self.dropout = configs.get("tcn_dropout", configs.get("dropout", 0.2))

        if self.c_out != 1:
            raise ValueError(
                f"ResidualTCN currently expects c_out=1 for univariate residual forecasting, got c_out={self.c_out}."
            )

        self.backbone = TCNBackbone(
            c_in=self.c_in,
            hidden_channels=self.hidden_channels,
            levels=self.levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        # Forecast head:
        # [B, hidden, seq_len] -> take final timestep hidden -> [B, hidden]
        # -> map to [B, pred_len]
        self.head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.pred_len),
        )

    def forward(
        self,
        indices_x,
        indices_y,
        x_enc,
        x_mark_enc,
        dec_inp,
        x_dec,
        x_mark_dec,
        cycle_index,
        mode,
        use_tf: bool = False,
        count: int = -1,
    ) -> torch.Tensor:
        """
        Parameters follow the same library-wide interface as PatchTST,
        but this model only uses x_enc.

        Expected x_enc shape:
            [batch, seq_len, enc_in]

        Returns:
            [batch, pred_len, 1]
        """

        # [B, T, C] -> [B, C, T]
        x = x_enc.permute(0, 2, 1)

        # [B, hidden, T]
        features = self.backbone(x)

        # use final temporal state
        last_hidden = features[:, :, -1]

        # [B, pred_len]
        forecast = self.head(last_hidden)

        # [B, pred_len, 1]
        forecast = forecast.unsqueeze(-1)

        return forecast
