import torch
import torch.nn as nn
import torch.nn.functional as F

class TATLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_derivative=True,
        use_decay=True,
        use_output_embedding=False,
        embed_dim=16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_derivative = use_derivative
        self.use_decay = use_decay
        self.use_output_embedding = use_output_embedding
        self.embed_dim = embed_dim

        # Adjust input size if using derivative
        self.effective_in_features = in_features * (2 if use_derivative else 1)

        # Base weight setup
        if use_output_embedding:
            self.output_embed = nn.Parameter(torch.randn(out_features, embed_dim) * 0.01)
            self.weight_base = nn.Parameter(torch.randn(embed_dim, self.effective_in_features) * 0.01)
        else:
            self.weight = nn.Parameter(torch.randn(out_features, self.effective_in_features) * 0.01)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if use_decay:
            base_decay = torch.linspace(1.0, 0.5, in_features)
            decay = base_decay.repeat(2 if use_derivative else 1)  # [effective_in_features]
            self.decay = nn.Parameter(decay)

    def forward(self, x):
        """
        x: [*, in_features] – supports 2D or 3D inputs
        """
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)  # flatten to [N, in_features]

        # Derivative input
        if self.use_derivative:
            dx = x[:, 1:] - x[:, :-1]
            dx = F.pad(dx, (1, 0))  # pad left
            x = torch.cat([x, dx], dim=-1)  # [N, 2 * in_features]

        # Decay weighting
        if self.use_decay:
            x = x * self.decay  # [N, effective_in_features] * [effective_in_features]

        # Weight projection
        if self.use_output_embedding:
            W = torch.matmul(self.output_embed, self.weight_base)  # [out_features, effective_in_features]
        else:
            W = self.weight  # [out_features, effective_in_features]

        # Linear projection
        out = F.linear(x, W, self.bias)  # [N, out_features]
        return out.reshape(*original_shape[:-1], self.out_features)