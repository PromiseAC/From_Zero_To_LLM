# Week 4 Day 2 - MiniMind 模型源码与 Tensor Shape

## 0. 中文复习总结

今天的目标不是重新学习 Transformer，而是把第二周学过的 Transformer 结构映射到 MiniMind 的真实源码中，并且把关键 Tensor Shape 全部串起来。

MiniMind 从输入到输出的主线是：

```text
input_ids
[B,T]

↓ Embedding

hidden_states
[B,T,D]

↓ Transformer Block × N

hidden_states
[B,T,D]

↓ Final RMSNorm

hidden_states
[B,T,D]

↓ LM Head

logits
[B,T,V]
```

其中：

```text
B = batch size
T = sequence length
D = hidden size
V = vocabulary size
```

MiniMind 默认配置：

```text
vocab_size = 6400
hidden_size = 768
num_hidden_layers = 8
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 96
intermediate_size = 2432
```

因此：

```text
V = 6400
D = 768
N = 8
Hq = 8
Hkv = 4
Dh = 96
I = 2432
```

MiniMind 默认使用 GQA：

```text
Q heads = 8
KV heads = 4

n_rep = 8 / 4 = 2
```

也就是：

```text
每 2 个 Query Head
共享 1 个 K/V Head
```

一个 MiniMind Transformer Block 的结构是：

```text
x
[B,T,D]

↓ RMSNorm

[B,T,D]

↓ Attention

[B,T,D]

↓ Residual Add

[B,T,D]

↓ RMSNorm

[B,T,D]

↓ SwiGLU FeedForward

[B,T,D]

↓ Residual Add

[B,T,D]
```

因此一个 Block 的输入输出始终都是：

```text
[B,T,D] → [B,T,D]
```

连续经过 8 个 Block 后，shape 仍然是：

```text
[B,T,D]
```

因为 RMSNorm 不改变 shape，Attention 最终会把多头重新 concat 回 D，FeedForward 虽然中间会从 D 扩展到 I，但最后 `down_proj` 会重新投影回 D，而 Residual Add 两边也必须保持相同 shape。

最后 LM Head：

```text
hidden_states
[B,T,D]

↓

Linear(D,V)

↓

logits
[B,T,V]
```

也就是把每个 token position 的 D 维 hidden vector 投影成对整个 vocabulary 的 V 个预测分数。

---

# 1. 整体模型结构

MiniMind 的完整结构：

```text
Token IDs
[B,T]

↓
Token Embedding

[B,T,D]

↓
MiniMindBlock 1

[B,T,D]

↓
MiniMindBlock 2

[B,T,D]

↓
...

↓
MiniMindBlock N

[B,T,D]

↓
Final RMSNorm

[B,T,D]

↓
LM Head

[B,T,V]
```

源码：

```text
model/model_minimind.py
```

主要类：

```text
MiniMindConfig
RMSNorm
Attention
FeedForward
MOEFeedForward
MiniMindBlock
MiniMindModel
MiniMindForCausalLM
```

---

# 2. MiniMindConfig

源码：

```python
class MiniMindConfig(PretrainedConfig):
```

默认关键参数：

```text
vocab_size              = 6400
hidden_size             = 768
num_hidden_layers       = 8
num_attention_heads     = 8
num_key_value_heads     = 4
head_dim                = 96
intermediate_size       = 2432
max_position_embeddings = 32768
rms_norm_eps            = 1e-6
rope_theta              = 1e6
tie_word_embeddings     = True
use_moe                 = False
```

符号映射：

```text
V   = vocab_size
D   = hidden_size
N   = num_hidden_layers
Hq  = num_attention_heads
Hkv = num_key_value_heads
Dh  = head_dim
I   = intermediate_size
```

其中：

```text
Dh = D / Hq
   = 768 / 8
   = 96
```

`intermediate_size` 的默认计算：

```python
math.ceil(hidden_size * math.pi / 64) * 64
```

代入：

```text
hidden_size = 768
```

得到：

```text
intermediate_size = 2432
```

---

# 3. Embedding

源码：

```python
self.embed_tokens = nn.Embedding(
    config.vocab_size,
    config.hidden_size
)
```

默认：

```text
Embedding(6400,768)
```

输入：

```text
input_ids
[B,T]
```

每个位置只保存一个整数 Token ID。

经过 Embedding：

```text
[B,T]
↓
Embedding
[B,T,D]
```

例如：

```text
[2,16]
↓
[2,16,768]
```

注意：

```text
input_ids [B,T]
```

不是：

```text
[B,T,D]
```

只有经过 Embedding 后才得到 hidden states：

```text
[B,T,D]
```

---

# 4. RMSNorm

源码：

```python
class RMSNorm(torch.nn.Module):
    def norm(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )
```

核心公式：

```text
RMSNorm(x)
=
x / sqrt(mean(x²) + eps)
```

然后再乘可学习参数：

```text
weight [D]
```

Tensor Shape：

```text
[B,T,D]
↓ RMSNorm
[B,T,D]
```

RMSNorm 不改变 Tensor Shape。

MiniMind 中还会：

```python
x.float()
```

先转成 float32 计算，提高数值稳定性，再转回原 dtype。

---

# 5. RMSNorm 的三个位置

MiniMind 中需要记住三个 Norm 位置。

Block 内：

```text
input_layernorm
→ Pre-Attention Norm
```

以及：

```text
post_attention_layernorm
→ Pre-FFN Norm
```

全部 Block 结束后：

```text
self.norm
→ Final RMSNorm
```

所以：

```text
Pre-Attention Norm
Pre-FFN Norm
Final Norm
```

都已经在真实源码中找到。

---

# 6. RoPE

相关函数：

```text
precompute_freqs_cis()
apply_rotary_pos_emb()
```

MiniMind 先预计算：

```text
cos
sin
```

默认：

```text
head_dim = 96
max_position_embeddings = 32768
```

因此大致：

```text
freqs_cos [32768,96]
freqs_sin [32768,96]
```

RoPE 真正应用在：

```python
xq, xk = apply_rotary_pos_emb(
    xq,
    xk,
    cos,
    sin
)
```

所以：

```text
RoPE 作用于 Q
RoPE 作用于 K
RoPE 不作用于 V
```

Shape 不变：

```text
Q [B,T,Hq,Dh]
→ [B,T,Hq,Dh]

K [B,T,Hkv,Dh]
→ [B,T,Hkv,Dh]
```

---

# 7. Attention 总流程

MiniMind Attention：

```text
x
[B,T,D]

↓ q_proj / k_proj / v_proj

Q
K
V

↓ Split Heads

Q [B,T,Hq,Dh]
K [B,T,Hkv,Dh]
V [B,T,Hkv,Dh]

↓ QK RMSNorm

shape 不变

↓ RoPE(Q,K)

shape 不变

↓ repeat_kv

K/V heads 从 Hkv 扩到 Hq

↓ transpose

[B,Hq,T,Dh]

↓ Q @ K^T

attention_scores
[B,Hq,T,T]

↓ Causal Mask
↓ Attention Mask
↓ Softmax

attention_weights
[B,Hq,T,T]

↓ attention_weights @ V

attention_output
[B,Hq,T,Dh]

↓ transpose

[B,T,Hq,Dh]

↓ concat heads

[B,T,D]

↓ o_proj

[B,T,D]
```

---

# 8. Q / K / V Projection

Attention 输入：

```text
x
[B,T,D]
=
[B,T,768]
```

Q Projection：

```python
self.q_proj = nn.Linear(
    config.hidden_size,
    config.num_attention_heads * self.head_dim
)
```

默认：

```text
768 → 8 × 96 = 768
```

所以：

```text
Q before split
[B,T,768]
```

K Projection：

```text
768 → 4 × 96 = 384
```

所以：

```text
K before split
[B,T,384]
```

V 同理：

```text
V before split
[B,T,384]
```

---

# 9. Split Heads

源码：

```python
xq = xq.view(
    bsz,
    seq_len,
    self.n_local_heads,
    self.head_dim
)

xk = xk.view(
    bsz,
    seq_len,
    self.n_local_kv_heads,
    self.head_dim
)

xv = xv.view(
    bsz,
    seq_len,
    self.n_local_kv_heads,
    self.head_dim
)
```

得到：

```text
Q
[B,T,8,96]

K
[B,T,4,96]

V
[B,T,4,96]
```

通用形式：

```text
Q [B,T,Hq,Dh]
K [B,T,Hkv,Dh]
V [B,T,Hkv,Dh]
```

---

# 10. QK Norm

源码：

```python
xq = self.q_norm(xq)
xk = self.k_norm(xk)
```

这里 RMSNorm 的维度是：

```text
Dh = 96
```

也就是对每一个 Head 的最后一维做 RMSNorm。

Shape：

```text
Q [B,T,8,96]
→ [B,T,8,96]

K [B,T,4,96]
→ [B,T,4,96]
```

---

# 11. GQA 与 repeat_kv

MiniMind 默认：

```text
Hq = 8
Hkv = 4
```

所以：

```text
n_rep = Hq / Hkv
      = 2
```

源码：

```python
repeat_kv(xk, self.n_rep)
repeat_kv(xv, self.n_rep)
```

Shape：

```text
K
[B,T,4,96]

↓ repeat 2

[B,T,8,96]
```

V：

```text
[B,T,4,96]
↓
[B,T,8,96]
```

概念对应：

```text
Q0 Q1 → KV0
Q2 Q3 → KV1
Q4 Q5 → KV2
Q6 Q7 → KV3
```

所以 MiniMind 默认是 GQA，而不是普通 MHA。

---

# 12. transpose

Attention 计算前：

```text
Q [B,T,8,96]
K [B,T,8,96]
V [B,T,8,96]
```

经过：

```python
transpose(1,2)
```

得到：

```text
Q [B,8,T,96]
K [B,8,T,96]
V [B,8,T,96]
```

即：

```text
[B,T,H,Dh]
→
[B,H,T,Dh]
```

这样每个 Head 可以独立做 Attention。

---

# 13. Attention Scores

源码：

```python
scores = (
    xq @ xk.transpose(-2, -1)
) / math.sqrt(self.head_dim)
```

Shape：

```text
Q
[B,8,T,96]

@

K^T
[B,8,96,T]

↓

scores
[B,8,T,T]
```

通用：

```text
attention_scores
[B,Hq,Tq,Tk]
```

无 KV Cache 时：

```text
Tq = Tk = T
```

所以：

```text
[B,Hq,T,T]
```

---

# 14. Causal Mask

Causal Mask 的作用：

```text
当前 token
不能看到未来 token
```

例如：

```text
0   -inf -inf -inf
0    0   -inf -inf
0    0    0   -inf
0    0    0    0
```

Shape 不变：

```text
[B,H,T,T]
```

---

# 15. Padding / Attention Mask

如果有：

```text
attention_mask [B,T]
```

MiniMind 会扩展成：

```text
[B,1,1,T]
```

再广播到 Attention Scores。

作用：

```text
Causal Mask
→ 防止看到未来 token

Attention / Padding Mask
→ 防止关注 PAD token
```

两者作用不同。

---

# 16. Softmax

Attention Scores：

```text
[B,H,T,T]
```

经过：

```python
F.softmax(scores, dim=-1)
```

得到：

```text
attention_weights
[B,H,T,T]
```

Shape 不变。

Softmax 在最后一个 Key 维度上进行。

---

# 17. Attention Weights × V

```text
attention_weights
[B,H,T,T]

@

V
[B,H,T,Dh]

↓

attention_output
[B,H,T,Dh]
```

默认：

```text
[B,8,T,96]
```

所以：

```text
Attention Scores
[B,H,T,T]
```

经过和 V 相乘后：

```text
Attention Output
[B,H,T,Dh]
```

---

# 18. Concat Heads

Attention Output：

```text
[B,8,T,96]
```

先 transpose：

```text
[B,T,8,96]
```

然后 reshape：

```text
[B,T,8×96]
```

即：

```text
[B,T,768]
```

最后：

```python
self.o_proj(...)
```

默认：

```text
Linear(768,768)
```

所以最终：

```text
Attention Input
[B,T,D]

↓

Attention Output
[B,T,D]
```

---

# 19. Attention Shape 总表

| Tensor | Shape |
|---|---|
| Attention input | `[B,T,D]` |
| Q before split | `[B,T,Hq×Dh]` |
| K before split | `[B,T,Hkv×Dh]` |
| V before split | `[B,T,Hkv×Dh]` |
| Q after split | `[B,T,Hq,Dh]` |
| K after split | `[B,T,Hkv,Dh]` |
| V after split | `[B,T,Hkv,Dh]` |
| K after repeat_kv | `[B,T,Hq,Dh]` |
| V after repeat_kv | `[B,T,Hq,Dh]` |
| Q/K/V after transpose | `[B,Hq,T,Dh]` |
| Attention Scores | `[B,Hq,T,T]` |
| Attention Weights | `[B,Hq,T,T]` |
| Attention Output | `[B,Hq,T,Dh]` |
| Concat Heads | `[B,T,D]` |
| Final Attention Output | `[B,T,D]` |

使用 KV Cache 时，更严格：

```text
Attention Scores
[B,Hq,Tq,Tk]
```

因为：

```text
Tq
```

可能只是当前 token 数，而：

```text
Tk
```

还包含历史 KV Cache。

---

# 20. KV Cache

Attention 中：

```python
if past_key_value is not None:
    xk = torch.cat(
        [past_key_value[0], xk],
        dim=1
    )

    xv = torch.cat(
        [past_key_value[1], xv],
        dim=1
    )
```

历史：

```text
past K
[B,Tpast,Hkv,Dh]
```

当前：

```text
current K
[B,Tcurrent,Hkv,Dh]
```

拼接：

```text
[B,Tpast+Tcurrent,Hkv,Dh]
```

V 同理。

MiniMind 保存 KV Cache 时是在 `repeat_kv()` 之前：

```text
K [B,T,Hkv,Dh]
V [B,T,Hkv,Dh]
```

默认：

```text
[B,T,4,96]
```

所以 GQA 可以降低 KV Cache 内存。

---

# 21. FeedForward / SwiGLU

源码：

```python
return self.down_proj(
    self.act_fn(self.gate_proj(x))
    *
    self.up_proj(x)
)
```

默认：

```text
D = 768
I = 2432
```

两条分支：

```text
                 ┌→ gate_proj → SiLU ─┐
x [B,T,D] ───────┤                    ×
                 └→ up_proj ──────────┘
                                      ↓
                                   [B,T,I]
                                      ↓
                                  down_proj
                                      ↓
                                   [B,T,D]
```

Shape：

```text
x
[B,T,D]

↓ gate_proj

[B,T,I]

↓ SiLU

[B,T,I]
```

另一支：

```text
x
[B,T,D]

↓ up_proj

[B,T,I]
```

然后：

```text
[B,T,I]
*
[B,T,I]

↓

[B,T,I]
```

这里 `*` 是：

```text
element-wise multiplication
```

不是矩阵乘法。

最后：

```text
[B,T,I]
↓ down_proj
[B,T,D]
```

默认：

```text
[B,T,768]
→
[B,T,2432]
→
[B,T,768]
```

---

# 22. 为什么这是 SwiGLU

核心：

```python
SiLU(gate_proj(x)) * up_proj(x)
```

可以理解为：

```text
gate branch
×
value branch
```

其中 gate 使用：

```text
SiLU / Swish
```

所以属于：

```text
SwiGLU
```

虽然源码中没有：

```python
class SwiGLU
```

但实现本身就是 SwiGLU。

---

# 23. MiniMindBlock

源码逻辑：

```python
residual = hidden_states

hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)

hidden_states += residual

hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)
)
```

数学形式：

```text
h = x + Attention(RMSNorm(x))

y = h + FFN(RMSNorm(h))
```

因此 MiniMind 使用：

```text
Pre-Norm Transformer Block
```

完整流程：

```text
x
[B,T,D]

↓ RMSNorm

[B,T,D]

↓ Attention

[B,T,D]

↓ Residual Add

[B,T,D]

↓ RMSNorm

[B,T,D]

↓ SwiGLU FeedForward

[B,T,D]

↓ Residual Add

[B,T,D]
```

最终：

```text
MiniMindBlock
[B,T,D]
→
[B,T,D]
```

---

# 24. MiniMindModel

初始化：

```python
self.embed_tokens = nn.Embedding(...)
self.layers = nn.ModuleList(...)
self.norm = RMSNorm(...)
```

整体：

```text
input_ids
[B,T]

↓ Embedding

[B,T,D]

↓ MiniMindBlock × N

[B,T,D]

↓ Final RMSNorm

[B,T,D]
```

默认：

```text
[B,T]
↓
[B,T,768]
↓ 8 Blocks
[B,T,768]
↓ Final RMSNorm
[B,T,768]
```

MiniMindModel 的输出仍然是：

```text
hidden_states [B,T,D]
```

还不是 logits。

---

# 25. Final RMSNorm

所有 Block 结束后：

```python
hidden_states = self.norm(hidden_states)
```

Shape：

```text
[B,T,D]
→
[B,T,D]
```

因此整个 Base Model：

```text
input_ids [B,T]
→
hidden_states [B,T,D]
```

---

# 26. MiniMindForCausalLM

`MiniMindForCausalLM` 在 `MiniMindModel` 外面再加：

```python
self.lm_head = nn.Linear(
    hidden_size,
    vocab_size,
    bias=False
)
```

因此：

```text
MiniMindModel
input_ids [B,T]
→
hidden_states [B,T,D]

MiniMindForCausalLM
input_ids [B,T]
→
logits [B,T,V]
```

---

# 27. LM Head

默认：

```text
Linear(768,6400)
```

所以：

```text
hidden_states
[B,T,768]

↓

LM Head

↓

logits
[B,T,6400]
```

通用：

```text
[B,T,D]
→
[B,T,V]
```

每一个 token position 最终得到：

```text
V 个 logits
```

代表对整个 vocabulary 的预测分数。

---

# 28. Weight Tying

默认：

```text
tie_word_embeddings = True
```

MiniMind 共享：

```text
Token Embedding Weight
和
LM Head Weight
```

两者 shape 都是：

```text
[V,D]
```

默认：

```text
[6400,768]
```

作用之一是减少参数量。

---

# 29. Next-token Prediction

如果：

```text
logits [B,T,V]
labels [B,T]
```

模型内部：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

所以：

```text
x [B,T-1,V]
y [B,T-1]
```

再 reshape：

```text
x
[B×(T-1),V]

y
[B×(T-1)]
```

然后：

```text
CrossEntropy
→ scalar loss
```

例如：

```text
B = 2
T = 16
V = 6400
```

则：

```text
logits
[2,16,6400]

↓

x
[2,15,6400]

y
[2,15]

↓

reshape

x
[30,6400]

y
[30]
```

---

# 30. 三种 Tensor 必须区分

这是今天最容易混淆的地方：

```text
input_ids / labels
[B,T]
```

每个位置是：

```text
一个 Token ID
```

例如：

```text
5412
```

---

```text
hidden_states
[B,T,D]
```

每个位置是：

```text
一个 D 维 hidden vector
```

---

```text
logits
[B,T,V]
```

每个位置是：

```text
对 V 个 token 的预测分数
```

必须牢记：

```text
input_ids 不是 [B,T,D]
labels 不是 [B,T,V]
```

---

# 31. Generation 与第三周对应

Autoregressive generation 中：

```text
input_ids
[B,T]

↓

forward

↓

logits
[B,T,V]

↓

取最后一个 position

logits[:, -1, :]
[B,V]

↓

sample / argmax

↓

next_token
[B,1]

↓

concat

↓

input_ids
[B,T+1]
```

然后继续下一轮。

MiniMind 比自己第三周写的 toy generate 多了：

```text
KV Cache
Temperature
Top-k
Top-p
Repetition Penalty
EOS stopping
```

但核心还是：

```text
预测一个 token
↓
拼回 input_ids
↓
继续预测
```

---

# 32. 完整 Tensor Shape 表

| Tensor | Shape |
|---|---|
| `input_ids` | `[B,T]` |
| Embedding Output | `[B,T,D]` |
| Q before split | `[B,T,Hq×Dh]` |
| K before split | `[B,T,Hkv×Dh]` |
| V before split | `[B,T,Hkv×Dh]` |
| Q after split | `[B,T,Hq,Dh]` |
| K after split | `[B,T,Hkv,Dh]` |
| V after split | `[B,T,Hkv,Dh]` |
| K/V after repeat_kv | `[B,T,Hq,Dh]` |
| Q/K/V after transpose | `[B,Hq,T,Dh]` |
| Attention Scores | `[B,Hq,T,T]` |
| Attention Weights | `[B,Hq,T,T]` |
| Attention Output | `[B,Hq,T,Dh]` |
| Concat Heads | `[B,T,D]` |
| Block Output | `[B,T,D]` |
| Final RMSNorm Output | `[B,T,D]` |
| Logits | `[B,T,V]` |
| Shifted Logits | `[B,T-1,V]` |
| Shifted Labels | `[B,T-1]` |

---

# 33. MiniMind 与自己的 MiniTransformer 对应

第二周自己的 MiniTransformer：

```text
Embedding
↓
Attention
↓
Residual
↓
Norm
↓
FFN
↓
Residual
↓
LM Head
```

MiniMind 核心思想完全相同，但增加了真实工程中的组件：

```text
GQA
QK Norm
RoPE
KV Cache
Flash / SDPA Attention
Attention Mask
SwiGLU
Weight Tying
HuggingFace Model Interface
Autoregressive Generation
```

因此可以理解为：

```text
自己的 MiniTransformer
→ 教学版 Transformer

MiniMind
→ 更接近真实 LLM 工程的 Transformer
```

底层 Tensor Flow 仍然是同一套逻辑。

---

# 34. 当日中文口头总结

可以用下面这段话完整复述今天的内容：

> MiniMind 的输入首先是 `input_ids [B,T]`，每个位置保存一个 Token ID。经过 Token Embedding 后，每个 Token ID 被映射成 D 维 hidden vector，因此得到 `hidden_states [B,T,D]`。
>
> hidden states 接下来经过多个 MiniMind Transformer Block。每个 Block 采用 Pre-Norm 结构，先经过 RMSNorm，再进入 GQA Attention，然后与原输入做 Residual Add；之后再经过 RMSNorm 和 SwiGLU FeedForward，并进行第二次 Residual Add。RMSNorm 不改变 Tensor Shape，Attention 最终通过 concat heads 和 output projection 回到 `[B,T,D]`，SwiGLU 虽然中间将 D 扩展到 intermediate size I，但最后 `down_proj` 又投影回 D，因此每个 Transformer Block 的输入输出都保持 `[B,T,D]`。
>
> MiniMind 的 Attention 默认使用 GQA。Q 有 8 个 head，而 K/V 只有 4 个 head。Q 的 shape 从 `[B,T,D]` 投影并拆头后变成 `[B,T,8,96]`，K/V 变成 `[B,T,4,96]`。Q 和 K 先经过 QK RMSNorm 和 RoPE，随后 K/V 通过 `repeat_kv` 扩展到 8 个 head，再 transpose 成 `[B,8,T,96]`。Q 和 K 做矩阵乘法后得到 Attention Scores `[B,8,T,T]`，经过 Causal Mask 和 Softmax，再与 V 相乘得到 `[B,8,T,96]`，最后 concat heads 回到 `[B,T,768]`。
>
> 所有 Transformer Block 结束后，还会经过一次 Final RMSNorm，因此 Base Model 最终输出仍然是 `hidden_states [B,T,D]`。最后 `MiniMindForCausalLM` 使用 LM Head，也就是 `Linear(D,V)`，把 hidden dimension 从 D 投影到 vocabulary size V，得到 `logits [B,T,V]`。
>
> 如果进行语言模型训练，模型会使用 `logits[:, :-1, :]` 与 `labels[:, 1:]` 做 next-token prediction，得到 `[B,T-1,V]` 的预测和 `[B,T-1]` 的目标，再经过 CrossEntropy 得到标量 loss。

---

# 35. 最短复习版

如果只剩 1 分钟复习，记住：

```text
input_ids
[B,T]

↓ Embedding

[B,T,D]

↓ N × Transformer Block

[B,T,D]

↓ Final RMSNorm

[B,T,D]

↓ LM Head

[B,T,V]
```

一个 Block：

```text
RMSNorm
→ Attention
→ Residual
→ RMSNorm
→ SwiGLU
→ Residual
```

Attention：

```text
Q [B,T,Hq,Dh]
K [B,T,Hkv,Dh]
V [B,T,Hkv,Dh]

↓ repeat_kv

K/V [B,T,Hq,Dh]

↓ transpose

[B,Hq,T,Dh]

↓ QKᵀ

scores [B,Hq,T,T]

↓ softmax @ V

[B,Hq,T,Dh]

↓ concat heads

[B,T,D]
```

MiniMind 默认：

```text
V = 6400
D = 768
N = 8
Hq = 8
Hkv = 4
Dh = 96
I = 2432
```
