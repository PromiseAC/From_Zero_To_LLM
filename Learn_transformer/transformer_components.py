import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    #x [B, T, D]
    def forward(self, x):
        # rms. [B, T, 1]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        #x_norm  [B, T, D]
        x_norm = x / rms
        # selft_weight [D]
        return x_norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = F.silu(gate) * up

        return self.down_proj(hidden)


if __name__ == "__main__":

    x = torch.randn(2, 4, 8)

    norm = RMSNorm(hidden_size=8)
    ffn = SwiGLU(hidden_size=8, intermediate_size=32)

    x_norm = norm(x)
    ffn_out = ffn(x_norm)
    output = x + ffn_out

    print("x shape:", x.shape)
    print("x_norm shape:", x_norm.shape)
    print("ffn_out shape:", ffn_out.shape)
    print("output shape:", output.shape)
