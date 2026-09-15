import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q,K,V,casual = False):
    """
    Q: [B, T, D]
    K: [B, T, D]
    V: [B, T, D]

    causal=False:
        普通 Self-Attention

    causal=True:
        Causal Self-Attention

    return:
        output:  [B, T, D]
        weights: [B, T, T]
    """

    # 1. 取出 d_k
    d_k = Q.shape[-1]
    # 2. Q @ K^T
    scores = Q @ K.transpose(-2, -1)
    # 3. Scale
    scaled_scores = scores / math.sqrt(d_k)

    if casual == True:

        # 4. Causal Mask
        T = Q.shape[-2]

        mask = torch.triu(
            torch.ones(T, T, device=Q.device),
            diagonal=1,
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

    # 5. Softmax
    # 每个 Query 对所有 Key 的权重和为 1
    attention_weights = torch.softmax(
        scores,
        dim=-1
    )
    # 6. Attention Weight @ V
    # [B, T, T] @ [B, T, D]
    # -> [B, T, D]
    output = attention_weights @ V

    return output, attention_weights, scores



torch.manual_seed(42)

B = 2
T = 4
D = 8

# 模拟 Transformer 输入
x = torch.randn(B, T, D)

# Q K V 三个线性投影
W_q = nn.Linear(D, D, bias=False)
W_k = nn.Linear(D, D, bias=False)
W_v = nn.Linear(D, D, bias=False)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

print("x shape:", x.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

output, attention_weights, masked_scores = \
    scaled_dot_product_attention(Q, K, V, casual=True)

print("\nmasked scores:")
print(masked_scores[0])

print("\nattention weights:")
print(attention_weights[0])

print("\n每一行的和:")
print(attention_weights[0].sum(dim=-1))

print("\noutput shape:")
print(output.shape)

print("\n第0个样本 output:")
print(output[0])
