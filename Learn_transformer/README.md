# Transformer From Scratch

本项目用于从零理解并实现一个最小的 Decoder-only Transformer 前向传播流程。

本项目重点不是训练一个完整大语言模型，而是通过手写代码理解 Transformer 中最核心的结构，包括：

* Token Embedding
* Scaled Dot-Product Attention
* Causal Mask
* Multi-Head Attention
* RMSNorm
* Residual Connection
* SwiGLU FFN
* Transformer Block
* Multi-Layer Transformer
* LM Head
* RoPE 基本原理
* KV Cache 基本原理
* MHA / MQA / GQA

---

## 1. 项目结构

```text
Learn_transformer/
├── softmax.py
├── attention.py
├── multi_head_attention.py
├── transformer_components.py
├── transformer_block.py
├── mini_transformer.py
└── README.md
```

各文件作用：

```text
softmax.py
→ 手写 Softmax

attention.py
→ 实现 Scaled Dot-Product Attention
→ 实现 Causal Mask

multi_head_attention.py
→ 实现 Multi-Head Attention

transformer_components.py
→ RMSNorm
→ SwiGLU FFN

transformer_block.py
→ 将 Attention、Norm、FFN、Residual 组合成 TransformerBlock

mini_transformer.py
→ 组合 Embedding、多层 TransformerBlock、Final Norm 和 LM Head
```

---

# 2. Decoder-only Transformer 整体数据流

整体数据流：

```text
Token IDs
[B,T]

↓ Token Embedding

[B,T,D]

↓ Transformer Block × N

[B,T,D]

↓ Final RMSNorm

[B,T,D]

↓ LM Head

[B,T,V]
```

其中：

```text
B = Batch Size
T = Sequence Length
D = Hidden Size / d_model
V = Vocabulary Size
H = Number of Attention Heads
Dh = Head Dimension = D / H
```

例如：

```text
B = 2
T = 8
D = 64
V = 100
H = 4
```

输入：

```text
tokens.shape = [2,8]
```

最终：

```text
logits.shape = [2,8,100]
```

---

# 3. Token、Token ID 与 Embedding

文本首先经过 Tokenizer，被转换成 Token。

例如：

```text
"I like AI"

↓

["I", "like", "AI"]
```

每个 Token 会对应词表中的一个整数 ID：

```text
[12, 53, 87]
```

这些整数就是 Token ID。

Token ID 本身没有语义，因此需要使用 Embedding 将其转换成向量：

```python
nn.Embedding(vocab_size, d_model)
```

Shape：

```text
Token IDs
[B,T]

↓

Embedding

[B,T,D]
```

例如：

```text
[2,8]
→
[2,8,64]
```

Embedding 权重矩阵：

```text
[vocab_size, d_model]
```

例如：

```text
[50000,768]
```

表示词表中有 50000 个 Token，每个 Token 对应一个 768 维向量。

---

# 4. Q、K、V

Self-Attention 首先根据输入 `X` 计算：

```text
Q = XWq
K = XWk
V = XWv
```

可以直观理解为：

```text
Q：当前 Token 想找什么信息

K：当前 Token 可以被匹配的特征

V：当前 Token 实际提供的信息
```

Shape：

```text
X       [B,T,D]

Q       [B,T,D]
K       [B,T,D]
V       [B,T,D]
```

---

# 5. Scaled Dot-Product Attention

Attention 的核心公式：

```text
Attention(Q,K,V)
=
softmax(QKᵀ / sqrt(d_k)) V
```

计算过程：

```text
QKᵀ
↓
计算 Token 之间的相关性

↓

除以 sqrt(d_k)

↓

Causal Mask

↓

Softmax

↓

Attention Weight

↓

Attention Weight × V

↓

新的 Token 表示
```

Shape：

```text
Q               [B,T,D]
K               [B,T,D]
V               [B,T,D]

QKᵀ             [B,T,T]

Attention Weight
                [B,T,T]

Output          [B,T,D]
```

`[B,T,T]` 表示：

> 对每个样本，每一个 Query Token 对序列中所有 Key Token 的相关程度。

---

# 6. 为什么除以 sqrt(d_k)

当 `d_k` 很大时，Q 和 K 做点积后的数值可能变得很大。

如果直接进入 Softmax：

```text
softmax(很大的数)
```

容易使概率分布过于极端，导致梯度变小。

因此使用：

```text
QKᵀ / sqrt(d_k)
```

控制 Attention Score 的数值尺度。

---

# 7. Softmax

Softmax 将 Attention Score 转换为概率分布。

例如：

```text
scores
[1,2,3]

↓

softmax

[0.09,0.24,0.67]
```

每一行权重之和：

```text
= 1
```

Attention 中通常：

```python
softmax(scores, dim=-1)
```

表示：

> 每一个 Query 对所有 Key 的 Attention Weight 之和为 1。

---

# 8. Causal Mask

Decoder-only Transformer 进行自回归预测时：

```text
当前位置不能看到未来 Token。
```

例如：

```text
Token 1 → 只能看到 Token 1

Token 2 → 可以看到 Token 1、2

Token 3 → 可以看到 Token 1、2、3
```

Causal Mask：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

实现时通常将未来位置设为：

```text
-inf
```

然后再执行 Softmax。

因为：

```text
softmax(-inf) = 0
```

因此模型不会关注未来 Token。

Mask 必须放在 Softmax 之前。

---

# 9. Causal Mask 与 Padding Mask

两者作用不同：

```text
Causal Mask
→ 防止模型看到未来 Token

Padding Mask
→ 防止模型关注 Padding Token
```

---

# 10. Multi-Head Attention

Multi-Head Attention 将 hidden dimension 拆成多个 Attention Head。

假设：

```text
D = 512
H = 8
```

那么：

```text
Dh = D / H
   = 512 / 8
   = 64
```

完整 Shape：

```text
X
[B,T,D]

↓

Q / K / V
[B,T,D]

↓

Split Heads
[B,H,T,Dh]

↓

Attention Scores
[B,H,T,T]

↓

Attention Output
[B,H,T,Dh]

↓

Concat

[B,T,D]

↓

Output Projection

[B,T,D]
```

其中：

```text
[B,H,T,T]
```

表示：

> 每个 Batch、每个 Attention Head 中，每个 Query Token 对所有 Key Token 的 Attention Score。

---

# 11. MHA、MQA、GQA

## MHA

Multi-Head Attention：

```text
每一个 Q Head
都有自己独立的 K Head 和 V Head
```

例如：

```text
8 Q Heads
8 K Heads
8 V Heads
```

---

## MQA

Multi-Query Attention：

```text
所有 Q Heads
共享同一组 K/V
```

例如：

```text
8 Q Heads
1 K Head
1 V Head
```

因此 KV Cache 更小。

---

## GQA

Grouped-Query Attention：

```text
多个 Q Heads 分成若干组

同一组 Q Heads
共享一组 K/V
```

例如：

```text
8 Q Heads
2 K/V Heads
```

因此：

```text
KV Cache 大小：

MHA > GQA > MQA
```

需要注意：

> GQA 分组的是 Q Heads，不是 Token。

---

# 12. Residual Connection

Residual Connection：

```text
y = x + F(x)
```

例如：

```python
x = x + attention(norm(x))
```

或者：

```python
x = x + ffn(norm(x))
```

作用：

```text
保留原始信息
+
加入当前层学习的新信息
```

Residual 还可以改善深层网络中的梯度传播。

为了进行 Residual Add：

```text
x.shape
必须与
F(x).shape
一致
```

因此 Transformer Block 通常保持：

```text
[B,T,D]
→
[B,T,D]
```

---

# 13. LayerNorm 与 RMSNorm

LayerNorm 主要进行：

```text
减均值
↓
除标准差
↓
可学习缩放
```

RMSNorm 则不进行减均值：

```text
RMS(x)
=
sqrt(mean(x²))
```

简化形式：

```text
RMSNorm(x)
=
x / sqrt(mean(x²) + eps)
× weight
```

其中：

```text
weight.shape = [D]
```

它是一个：

```python
nn.Parameter
```

因此可以被 optimizer 更新。

Shape：

```text
输入
[B,T,D]

↓

RMSNorm

[B,T,D]
```

RMSNorm 不改变 Tensor shape。

---

# 14. Pre-Norm

本项目使用 Pre-Norm：

```python
x = x + attention(norm1(x))
x = x + ffn(norm2(x))
```

也就是：

```text
先 Norm
↓
再 Attention / FFN
↓
最后 Residual Add
```

这就是：

```text
Pre-Norm
```

与之对应，Post-Norm 类似：

```text
Norm(x + Attention(x))
```

---

# 15. FFN

普通 Feed Forward Network：

```text
Linear
↓
Activation
↓
Linear
```

通常：

```text
[B,T,D]

↓

[B,T,4D]

↓

[B,T,D]
```

FFN 与 Attention 的作用不同：

```text
Attention
→ 负责不同 Token 之间的信息交互

FFN
→ 负责每个 Token 自己内部的特征变换
```

FFN 不会直接让 Token 20 读取 Token 50 的信息。

---

# 16. SwiGLU

现代 LLM 中常使用 SwiGLU 替代简单的 ReLU FFN。

本项目使用：

```python
gate = gate_proj(x)

up = up_proj(x)

hidden = silu(gate) * up

output = down_proj(hidden)
```

其中：

```text
*
```

表示逐元素乘法，不是矩阵乘法。

Shape：

```text
x
[B,T,D]

↓ gate_proj

[B,T,I]

↓ up_proj

[B,T,I]

↓

SiLU(gate) * up

[B,T,I]

↓

down_proj

[B,T,D]
```

其中：

```text
I = intermediate_size
```

Gate 可以直观理解为：

> 控制不同信息通过多少。

---

# 17. RoPE

RoPE：

```text
Rotary Position Embedding
```

用于向 Transformer 加入位置信息。

普通 Attention 本身无法天然知道 Token 的顺序，因此需要位置编码。

RoPE 主要作用于：

```text
Q
K
```

而不是 V。

原因是：

```text
QKᵀ
```

负责计算 Attention Scores。

因此将位置信息加入 Q/K 后，可以直接影响 Token 之间的 Attention 关系。

RoPE 可以理解为：

> 根据 Token 的位置，对 Q/K 向量中的维度进行旋转。

不同位置：

```text
position 1
position 2
position 3
```

会使用不同旋转角度。

RoPE 不改变 Shape：

```text
Q
[B,H,T,Dh]

↓

RoPE

[B,H,T,Dh]
```

K 同理。

RoPE 的一个重要特点是能够帮助模型表示相对位置关系。

本项目目前重点理解其原理，没有完整从零实现 RoPE。

---

# 18. KV Cache

Decoder-only Transformer 使用自回归生成：

```text
Prompt
↓
Token 1
↓
Token 2
↓
Token 3
...
```

如果没有 KV Cache，每生成一个新的 Token，都可能重新计算历史 Token 的 K/V。

但历史 Token 已经确定，因此：

```text
K_history
V_history
```

可以缓存。

例如：

```text
历史：

K1 K2 K3
V1 V2 V3
```

生成新 Token 后，只需要计算：

```text
Q4
K4
V4
```

然后：

```text
Q4
@
[K1 K2 K3 K4]ᵀ
```

得到 Attention Scores。

接着：

```text
Attention Weight
@
[V1 V2 V3 V4]
```

得到 Attention Output。

因此 KV Cache：

```text
缓存历史 K
缓存历史 V
```

不需要缓存历史 Q，因为下一次推理时只需要当前 Token 的 Q。

KV Cache 主要用于：

```text
推理阶段
```

序列越长：

```text
KV Cache 越大
```

而 MQA / GQA 由于 K/V Head 数量减少，可以明显降低 KV Cache 占用。

---

# 19. Transformer Block

本项目中的 Transformer Block：

```text
x
[B,T,D]

↓

RMSNorm

↓

Causal Multi-Head Self-Attention

↓

Residual Add

↓

RMSNorm

↓

SwiGLU FFN

↓

Residual Add

↓

output
[B,T,D]
```

代码逻辑可以简化为：

```python
x = x + attention(norm1(x))

x = x + ffn(norm2(x))
```

因此：

```text
TransformerBlock

[B,T,D]
→
[B,T,D]
```

多个 Block 可以直接堆叠。

---

# 20. Mini Transformer

最终 MiniTransformer：

```text
Token IDs
[B,T]

↓

Token Embedding

[B,T,D]

↓

TransformerBlock × N

[B,T,D]

↓

Final RMSNorm

[B,T,D]

↓

LM Head

[B,T,V]
```

核心结构：

```python
Embedding
→
Transformer Blocks
→
Final RMSNorm
→
LM Head
```

多层 TransformerBlock 使用：

```python
nn.ModuleList
```

依次执行：

```text
Embedding Output

↓

Block 1

↓

Block 2

↓

...

↓

Block N
```

每一层的输出都会作为下一层的输入。

---

# 21. LM Head

LM Head：

```python
nn.Linear(d_model, vocab_size)
```

完成：

```text
[B,T,D]
→
[B,T,V]
```

例如：

```text
[B,T,64]
→
[B,T,100]
```

最终：

```text
logits[b,t,:]
```

表示：

> 第 b 个样本、第 t 个位置，对整个 vocabulary 中每一个 Token 给出的预测分数。

---

# 22. Tensor Shape 总结

Transformer 中最重要的 Shape：

```text
Token IDs
[B,T]

↓

Embedding
[B,T,D]

↓

Q/K/V
[B,T,D]

↓

Split Heads
[B,H,T,Dh]

↓

Q × Kᵀ
[B,H,T,T]

↓

Attention Weight
[B,H,T,T]

↓

Attention Weight × V
[B,H,T,Dh]

↓

Concat Heads
[B,T,D]

↓

TransformerBlock
[B,T,D]

↓

LM Head
[B,T,V]
```

其中：

```text
Dh = D / H
```

---

# 23. 最终测试

测试参数：

```python
batch_size = 2
sequence_length = 8
vocab_size = 100
d_model = 64
num_heads = 4
num_layers = 2
intermediate_size = 256
```

输入：

```text
tokens.shape
=
[2,8]
```

模型前向传播：

```text
Token IDs
[2,8]

↓

Embedding
[2,8,64]

↓

TransformerBlock × 2
[2,8,64]

↓

Final RMSNorm
[2,8,64]

↓

LM Head
[2,8,100]
```

最终测试结果：

```text
tokens shape: torch.Size([2, 8])

logits shape: torch.Size([2, 8, 100])
```

说明最小 Decoder-only Transformer 前向传播成功运行。

---

# 24. 本项目完成内容

通过本项目已经完成：

```text
✓ 手写 Softmax

✓ 手写 Scaled Dot-Product Attention

✓ 实现 Causal Mask

✓ 手写 Multi-Head Attention

✓ 理解 MHA / MQA / GQA

✓ 手写 RMSNorm

✓ 手写 SwiGLU FFN

✓ 实现 Residual Connection

✓ 实现 Pre-Norm TransformerBlock

✓ 实现多层 TransformerBlock

✓ 实现 Token Embedding

✓ 实现 Final RMSNorm

✓ 实现 LM Head

✓ 完成 MiniTransformer 前向传播

✓ 理解 RoPE 基本原理

✓ 理解 KV Cache 基本原理
```

---

# 25. 当前项目边界

本项目主要用于理解 Transformer 核心结构，目前暂未实现：

```text
Tokenizer / BPE 训练

RoPE 完整实现

KV Cache 代码实现

模型训练

CrossEntropy Loss

文本生成循环

权重共享

真实 LLM 数据集

FlashAttention

LoRA

DPO / RLHF

Distributed Training
```

这些将在后续学习中逐步实现。

---

# 26. 本周核心收获

完成本项目后，应能够独立解释：

```text
[B,T]
↓
[B,T,D]
↓
[B,H,T,Dh]
↓
[B,H,T,T]
↓
[B,H,T,Dh]
↓
[B,T,D]
↓
[B,T,V]
```

并理解：

```text
Embedding
→ 把 Token ID 变成向量

Attention
→ 不同 Token 之间交换信息

Causal Mask
→ 防止看到未来 Token

Multi-Head
→ 从多个子空间进行 Attention

RMSNorm
→ 稳定 hidden state 的数值尺度

Residual
→ 保留原始信息并加入新信息

SwiGLU
→ 对每个 Token 内部特征做门控非线性变换

TransformerBlock
→ Attention + FFN

RoPE
→ 给 Q/K 加入位置信息

KV Cache
→ 推理时缓存历史 K/V

LM Head
→ 将 D 维表示映射为整个词表上的 logits
```

本阶段最终目标：

> 不依赖完整答案，可以从 `[B,T,D]` 出发推导 Attention 的主要 Tensor Shape，并理解如何将 Attention、Norm、FFN、Residual 组合成一个可以完成前向传播的 Decoder-only Transformer。
