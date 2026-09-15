import torch
import torch.nn as nn

from transformer_components import RMSNorm, SwiGLU
from multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_size):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(
            hidden_size=d_model,
            intermediate_size=intermediate_size,
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, weights = self.attention(
            norm_x,
            causal=True,
        )
        x = x + attn_out

        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x, weights


if __name__ == "__main__":
    B = 2
    T = 4
    D = 8
    H = 2
    intermediate_size = 32

    x = torch.randn(B, T, D)

    block = TransformerBlock(
        d_model=D,
        num_heads=H,
        intermediate_size=intermediate_size
    )

    output, weights = block(x)

    print("input shape:", x.shape)
    print("attention weights shape:", weights.shape)
    print("output shape:", output.shape)