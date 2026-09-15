import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        B, T, D = x.shape

        # 1. Q K V projection
        # [B,T,D] -> [B,T,D]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Split heads
        # [B,T,D] -> [B,T,H,Dh] -> [B,H,T,Dh]
        Q = Q.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 3. Attention scores
        # [B,H,T,Dh] @ [B,H,Dh,T]
        # -> [B,H,T,T]
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # 4. Causal mask
        if causal:
            mask = torch.triu(
                torch.ones(
                    T,
                    T,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                mask,
                float("-inf")
            )

        # 5. Softmax
        # [B,H,T,T]
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # 6. Weighted sum of V
        # [B,H,T,T] @ [B,H,T,Dh]
        # -> [B,H,T,Dh]
        attention_output = attention_weights @ V

        # 7. Concat heads
        # [B,H,T,Dh] -> [B,T,H,Dh]
        attention_output = attention_output.transpose(1, 2)

        # [B,T,H,Dh] -> [B,T,D]
        attention_output = attention_output.reshape(
            B, T, D
        )

        # 8. Output projection
        output = self.out_proj(attention_output)

        return output, attention_weights



if __name__ == "__main__":
    # ====================
    # Test
    # ====================
    B = 2
    T = 4
    D = 8
    H = 2

    x = torch.randn(B, T, D)

    mha = MultiHeadAttention(
        d_model=D,
        num_heads=H
    )

    output, weights = mha(x, causal=True)

    print("input shape:", x.shape)
    print("weights shape:", weights.shape)
    print("output shape:", output.shape)