# LLM Training From Scratch

本项目用于学习大语言模型训练的基础流程，是 LLM 学习计划第 3 周的代码实践与学习总结。

上一阶段已经从零实现了最小 Decoder-only Transformer。本阶段在此基础上加入真实文本、Tokenizer、训练数据、Loss、Optimizer、Learning Rate Scheduler 和生成循环，跑通一个语言模型从数据到训练、再到推理的完整闭环。

本周重点包括：

* Token、Token ID 与 Vocabulary
* BPE、SentencePiece 与中文 Tokenizer
* Causal Language Modeling
* Next Token Prediction 与 Cross Entropy Loss
* 数据清洗、过滤、去重与数据集划分
* Padding、Mask 与 Packing
* Dataset、DataLoader 与 Batch
* Gradient Accumulation 与训练规模
* AdamW、Warmup、Cosine Decay
* Gradient Norm 与 Gradient Clipping
* Validation、Checkpoint 与自回归生成

---

## 1. 项目结构

```text
Learn_llm_training/
├── tokenizer_demo.py
├── causal_lm_demo.py
├── data_pipeline_demo.py
├── training_scale_demo.py
├── training_recipe_demo.py
├── mini_lm_loss_demo.py
├── tiny_lm_train.py
├── small_pretrain_demo.py
├── generate.py
├── checkpoint.pt
├── checkpoint_epoch_0.pt
├── checkpoint_epoch_3.pt
├── checkpoint_epoch_9.pt
└── README.md
```

各文件作用：

```text
tokenizer_demo.py
→ 中英文 Tokenize、Encode、Decode
→ 查看 Vocabulary 与 Special Tokens

causal_lm_demo.py
→ 构造 Causal LM 的 input_ids 和 labels

data_pipeline_demo.py
→ 文本清洗、过滤、去重、数据划分
→ Padding、Mask、Packing 和固定长度切块

training_scale_demo.py
→ 计算 Global Batch Size、Tokens per Step 和训练 Token 数

training_recipe_demo.py
→ AdamW、Warmup、Cosine Decay 和 Gradient Clipping

mini_lm_loss_demo.py
→ 连接真实 Tokenizer 与 MiniTransformer
→ 计算 Loss、Perplexity 和 Next-token Accuracy

tiny_lm_train.py
→ 在 Toy Dataset 上跑通最小训练循环

small_pretrain_demo.py
→ 字符级小规模预训练、验证和 Checkpoint 保存

generate.py
→ 加载不同阶段的 Checkpoint 并进行自回归生成
```

模型主体复用上一阶段实现的 [`MiniTransformer`](../Learn_transformer/mini_transformer.py)。

---

# 2. 语言模型训练的完整流程

```text
Raw Text

↓ Cleaning / Filtering / Deduplication

Clean Text

↓ Tokenizer

Token IDs

↓ Chunking / Padding / Packing

Training Samples

↓ Dataset / DataLoader

Batch [B,T]

↓ MiniTransformer

Logits [B,T,V]

↓ Cross Entropy Loss

Loss

↓ Backward

Gradients

↓ AdamW + LR Scheduler

Updated Parameters

↓ Validation / Checkpoint

Trained Model

↓ Autoregressive Generation

Generated Text
```

其中：

```text
B = Batch Size
T = Sequence Length
V = Vocabulary Size
D = Hidden Size
```

---

# 3. Token、Token ID 与 Vocabulary

神经网络不能直接处理文本，原始文本需要先被 Tokenizer 转换为 Token。

```text
"我正在学习语言模型"

↓ Tokenizer

["我", "正在", "学习", "语言", "模型"]

↓ Vocabulary

[1024, 8531, 2234, 6789, 3456]
```

这些整数就是 Token ID。

完整关系：

```text
Raw Text
↓
Tokens
↓
Token IDs
↓
Embedding Vectors
```

Vocabulary 是 Token 与 Token ID 之间的映射表。词表越大，单个 Token 可能表示更完整的内容，但 Embedding 和 LM Head 的参数量也会随之增加。

---

# 4. Encode、Decode 与 Special Tokens

Tokenizer 的两个核心操作：

```text
encode
文本 → Token IDs

decode
Token IDs → 文本
```

常见特殊 Token：

```text
BOS → Beginning of Sequence
EOS → End of Sequence
PAD → Padding
UNK → Unknown Token
```

不同模型的 Special Token 设计可能不同，不能假设所有 Tokenizer 都同时具有这些 Token，需要检查：

```python
tokenizer.bos_token
tokenizer.eos_token
tokenizer.pad_token
tokenizer.unk_token
```

[`tokenizer_demo.py`](tokenizer_demo.py) 使用 `Qwen/Qwen2.5-0.5B` 的 Tokenizer，测试了中英文文本的切分、编码、解码、词表和特殊 Token。

---

# 5. BPE

BPE：

```text
Byte Pair Encoding
```

基本过程：

```text
从较小的符号单元开始
↓
统计相邻 Token Pair 的频率
↓
合并最高频的 Token Pair
↓
重复统计与合并
↓
得到最终 Vocabulary
```

BPE 在字符级和词级 Tokenizer 之间取得平衡：

```text
字符级
→ 词表小，但序列较长

词级
→ 序列短，但词表大且容易出现未知词

子词级
→ 常见词使用较完整的 Token
→ 生僻词仍可拆分，减少未知词
```

---

# 6. SentencePiece 与中文 Tokenizer

英文可以使用空格预分词，但中文句子通常没有天然空格：

```text
我正在学习大语言模型
```

SentencePiece 可以直接从原始文本学习子词模型，不依赖英文式空格分词，因此适合中文和多语言场景。

中文文本可能被切分为：

```text
单个汉字
多个汉字组成的词
字节或更细粒度的单元
```

实际结果由训练语料、词表大小和算法共同决定，不能简单认为一个汉字一定对应一个 Token。

---

# 7. Vocabulary Size

需要区分：

```python
tokenizer.vocab_size
```

与：

```python
len(tokenizer)
```

通常：

```text
vocab_size
→ 基础词表大小

len(tokenizer)
→ 当前 Tokenizer 的完整长度
→ 可能包含后来添加的 Token
```

模型的 Embedding 和 LM Head 必须使用正确的词表大小，否则可能出现 Token ID 越界或输出维度不匹配。

---

# 8. Causal Language Modeling

Decoder-only Language Model 的核心目标：

```text
根据前面的 Token，预测下一个 Token。
```

这就是 Next Token Prediction。

原始序列：

```text
[t0, t1, t2, t3, t4]
```

训练数据：

```text
input_ids = [t0, t1, t2, t3]
labels    = [t1, t2, t3, t4]
```

模型实际学习：

```text
t0          → 预测 t1
t0 t1       → 预测 t2
t0 t1 t2    → 预测 t3
t0 t1 t2 t3 → 预测 t4
```

[`causal_lm_demo.py`](causal_lm_demo.py) 展示了 input 和 label 错开一位的构造方式。

---

# 9. Shift Labels 与 Causal Mask

训练时整段序列可以同时输入模型，但当前位置不能读取未来 Token，因此还需要 Causal Mask：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

两者作用不同：

```text
Shift Labels
→ 决定每个位置需要预测什么

Causal Mask
→ 决定每个位置能够看到什么
```

如果没有 Causal Mask，模型训练时可以直接看到未来的答案，Next Token Prediction 就失去了意义。

---

# 10. Logits 与 Cross Entropy Loss

MiniTransformer 输出：

```text
logits [B,T,V]
```

它表示每个样本、每个位置对词表中全部 Token 的预测分数。

Labels：

```text
labels [B,T]
```

计算 Cross Entropy Loss 时需要展平：

```text
logits [B,T,V] → [B × T,V]
labels [B,T]   → [B × T]
```

代码形式：

```python
loss = criterion(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1),
)
```

Cross Entropy 会提高正确 Token 的预测概率，并压低错误 Token 的概率。

---

# 11. Perplexity 与 Next-token Accuracy

Perplexity：

```text
Perplexity = exp(Cross Entropy Loss)
```

通常：

```text
Loss 越低
→ Perplexity 越低
→ 模型对下一个 Token 的预测越确定
```

Next-token Accuracy 则判断 `argmax(logits)` 是否等于正确 label。

语言模型可能存在多个合理的下一个 Token，因此训练时一般以 Cross Entropy Loss 为主要指标，Accuracy 作为辅助观察。

---

# 12. 数据清洗、过滤与去重

真实文本需要先经过数据处理：

```text
Raw Text
↓
统一异常空白
↓
删除空文本
↓
过滤过短或过长文本
↓
Exact Deduplication
↓
Clean Text
```

数据清洗可以减少无效内容、降低噪声并提高有效 Token 的比例。

如果重复文本同时进入训练集和验证集，会造成 Data Leakage，使验证结果虚高。因此合理顺序是：

```text
清洗
↓
去重
↓
随机打乱
↓
Train / Validation / Test Split
```

[`data_pipeline_demo.py`](data_pipeline_demo.py) 实现了以上基础处理流程。

---

# 13. Padding

同一个 Batch 中的 Tensor 需要具有相同长度。

```text
样本 A：[12, 25, 39, 41]
样本 B：[17, 20]
```

Padding 后：

```text
样本 A：[12, 25, 39, 41]
样本 B：[17, 20, PAD, PAD]
```

Padding 便于组成 `[B,T]` 的 Batch，但 PAD Token 不包含真实训练信息，因此必须排除它对 Attention 和 Loss 的影响。

---

# 14. 三种 Mask

```text
Causal Mask
→ 防止当前位置看到未来 Token

Padding Attention Mask
→ 防止真实 Token 关注 PAD Token

Loss Mask
→ 防止 PAD 位置参与 Loss
```

Loss Mask 可以通过把 PAD 位置对应的 label 设置为 `-100` 实现：

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

这三种 Mask 解决的是不同问题，不能相互替代。

---

# 15. Padding Ratio 与 Packing

Padding Ratio：

```text
Padding Ratio
= Padding Token 数量 ÷ Batch 中全部 Token 数量
```

Padding Ratio 越高，浪费的显存和计算越多。

Packing 会把多个短样本连接起来，尽量填满固定长度的 Block：

```text
Document A
↓ EOS
Document B
↓ EOS
Document C

↓ 拼接

Continuous Token Stream

↓ 固定长度切块

Packed Blocks
```

Packing 能减少 Padding，但需要保留 EOS 等文档边界，避免把两个独立文档错误理解成同一句话。

---

# 16. Dataset 与 DataLoader

Dataset 定义单个样本：

```text
input_ids [T]
labels    [T]
```

DataLoader 将多个样本组成 Batch：

```text
input_ids [B,T]
labels    [B,T]
```

完整 Shape：

```text
input_ids [B,T]
↓ MiniTransformer
logits [B,T,V]
↓ Flatten + Cross Entropy
loss Scalar
```

[`tiny_lm_train.py`](tiny_lm_train.py) 使用小型整数序列验证了 Dataset、DataLoader、Forward、Loss、Backward 和参数更新。

---

# 17. Micro Batch 与 Gradient Accumulation

Micro Batch Size 是一次前向和反向传播实际放入设备的样本数量。

显存不足时，可以累积多个 Micro Batch 的梯度：

```text
Micro Batch 1 → backward
Micro Batch 2 → backward
Micro Batch 3 → backward
Micro Batch 4 → backward

↓

optimizer.step()
```

Global Batch Size：

```text
global_batch_size
= micro_batch_size
× gradient_accumulation_steps
× world_size
```

本项目示例：

```text
micro_batch_size = 4
gradient_accumulation_steps = 8
world_size = 2

global_batch_size = 4 × 8 × 2 = 64
```

Gradient Accumulation 可以在显存有限时模拟更大的 Batch，但不会减少处理相同 Token 数所需的总计算量。

---

# 18. Tokens per Step 与训练规模

```text
tokens_per_step
= global_batch_size × sequence_length

total_training_tokens
= tokens_per_step × training_steps

estimated_epochs
= total_training_tokens ÷ dataset_tokens
```

本项目示例：

```text
global_batch_size = 64
sequence_length = 2048
training_steps = 10000
dataset_tokens = 2,000,000,000
```

结果：

```text
tokens_per_step = 131,072
total_training_tokens = 1,310,720,000
estimated_epochs = 0.65536
```

对于大语言模型，使用训练 Token 数描述训练规模通常比只看 Epoch 更直观。

---

# 19. AdamW 与 Weight Decay

本项目使用：

```python
torch.optim.AdamW(...)
```

AdamW 将参数更新与 Weight Decay 分开处理。

Weight Decay 可以限制参数无限增大、降低过拟合风险并改善泛化。实际训练中，Bias 和 Norm 参数经常不使用 Weight Decay。

一个训练 Step 的基本顺序：

```text
Forward
↓
Loss
↓
Backward
↓
Gradient Clipping
↓
Optimizer Step
↓
LR Scheduler Step
↓
Zero Grad
```

---

# 20. Warmup 与 Cosine Decay

训练刚开始时，模型参数和优化器状态还不稳定。如果一开始就使用较大的 Learning Rate，可能导致 Loss Spike 或梯度异常。

Warmup：

```text
较小 Learning Rate
↓
逐步增大
↓
Peak Learning Rate
```

Warmup 后使用 Cosine Decay：

```text
Peak Learning Rate
↓
平滑下降
↓
Minimum Learning Rate
```

训练前期较大的 Learning Rate 有利于搜索参数空间，后期较小的 Learning Rate 能让更新更加稳定。

[`training_recipe_demo.py`](training_recipe_demo.py) 实现并绘制了完整的 Learning Rate 曲线。

---

# 21. Gradient Norm 与 Gradient Clipping

Gradient Norm 用于观察梯度的整体大小。

如果突然变大，可能说明：

```text
当前 Batch 异常
Learning Rate 过大
数值不稳定
出现梯度爆炸
```

Gradient Clipping：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm,
)
```

当梯度 Norm 超过阈值时，它会按比例缩小梯度。

Gradient Clipping 是保护机制，不是解决所有训练异常的根本方法。如果频繁触发，还需要检查数据、Learning Rate、初始化和数值精度。

---

# 22. 最小训练循环

```python
for input_ids, labels in dataloader:
    optimizer.zero_grad()

    logits = model(input_ids)

    loss = criterion(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )

    optimizer.step()
    scheduler.step()
```

数据流：

```text
[B,T]
↓ Model
[B,T,V]
↓ Cross Entropy
Scalar Loss
↓ Backward
Gradients
↓ Optimizer
Updated Parameters
```

如果模型能够在小数据上快速降低 Loss，说明数据、模型、Loss 和 Optimizer 的基本连接是正确的。

---

# 23. Validation

训练集用于更新参数，验证集用于检查模型对未参与训练数据的泛化能力。

验证阶段：

```python
model.eval()

with torch.no_grad():
    ...
```

不会执行：

```text
loss.backward()
optimizer.step()
```

如果 Train Loss 持续下降，而 Validation Loss 开始上升，模型可能出现过拟合。

---

# 24. 小规模预训练实验

[`small_pretrain_demo.py`](small_pretrain_demo.py) 使用中文文本和字符级 Tokenizer 完成了一次小规模预训练。

模型配置：

```text
vocab_size = 48
d_model = 32
num_heads = 4
num_layers = 2
intermediate_size = 128
max_seq_len = 32
micro_batch_size = 8
num_epochs = 10
learning_rate = 1e-3
weight_decay = 0.01
max_grad_norm = 1.0
warmup_steps = 20
```

训练流程：

```text
中文 Raw Text
↓
字符级 Vocabulary
↓
Token IDs
↓
固定长度 Chunk
↓
input_ids / labels
↓
Train / Validation DataLoader
↓
MiniTransformer
↓
Cross Entropy Loss
↓
AdamW + Scheduler
↓
Validation
↓
Checkpoint
```

虽然数据和模型都很小，但已经包含真实语言模型训练所需的主要组件。

---

# 25. Checkpoint

Checkpoint 用于保存训练状态：

```text
Model State
Optimizer State
Scheduler State
Epoch
Global Step
Model Config
stoi / itos Vocabulary
```

当前 Checkpoint：

```text
checkpoint_epoch_0.pt
→ epoch 0，global_step 27

checkpoint_epoch_3.pt
→ epoch 3，global_step 108

checkpoint_epoch_9.pt
→ epoch 9，global_step 270
```

只保存模型参数可以用于推理；如果希望准确恢复训练，还需要保存 Optimizer、Scheduler 和训练进度。

---

# 26. 自回归生成

```text
Prompt
↓ Tokenizer
input_ids
↓ Model Forward
logits [B,T,V]
↓ 取最后一个位置
next_token_logits [B,V]
↓ 选择 Next Token
追加到 input_ids
↓
重复以上过程
```

贪心生成：

```python
next_token = torch.argmax(next_token_logits, dim=-1)
```

它每次选择分数最高的 Token，结果稳定，但容易重复或缺少多样性。

[`generate.py`](generate.py) 使用“语言模型”作为 Prompt，分别加载 Epoch 0、3、9 的 Checkpoint，观察训练过程中的生成结果变化。

---

# 27. 训练与推理的区别

训练阶段：

```text
一次输入完整序列
使用 Causal Mask 防止看到未来
每个位置都计算 Next Token Loss
执行 Backward 和参数更新
```

推理阶段：

```text
从 Prompt 开始
每次生成一个新 Token
将新 Token 追加到输入
不执行 Backward
```

训练可以并行计算序列中所有位置的预测，自回归推理则必须按 Token 顺序逐步进行。

---

# 28. 训练过程需要观察的指标

```text
Train Loss
→ 模型是否在学习训练数据

Validation Loss
→ 模型能否泛化到未见数据

Perplexity
→ 模型对下一个 Token 的不确定程度

Learning Rate
→ Scheduler 是否正常运行

Gradient Norm
→ 是否出现梯度异常

Tokens per Second
→ 训练吞吐

Padding Ratio
→ 数据计算效率
```

只观察 Loss 不足以完整判断训练是否健康。

---

# 29. 本项目完成内容

通过本项目已经完成：

```text
✓ 理解 Token、Token ID 与 Vocabulary

✓ 使用真实 Tokenizer 处理中文和英文文本

✓ 理解 BPE 与 SentencePiece 的基本原理

✓ 检查 Special Tokens 并完成 Encode / Decode

✓ 理解 Causal Language Modeling

✓ 构造错开一位的 input_ids 和 labels

✓ 使用 Cross Entropy Loss 训练 Next Token Prediction

✓ 理解 logits [B,T,V] 与 labels [B,T]

✓ 实现文本清洗、长度过滤与精确去重

✓ 完成 Train / Validation / Test Split

✓ 理解 Padding、Attention Mask 与 Loss Mask

✓ 实现短文本 Packing 和固定长度 Chunking

✓ 使用 Dataset 与 DataLoader 组成 Batch

✓ 计算 Global Batch Size 与 Tokens per Step

✓ 理解 Gradient Accumulation

✓ 使用 AdamW 与 Weight Decay

✓ 实现 Warmup 与 Cosine Learning Rate Decay

✓ 监控 Gradient Norm 并进行 Gradient Clipping

✓ 跑通 MiniTransformer 的完整训练循环

✓ 计算 Validation Loss

✓ 保存并加载 Checkpoint

✓ 完成最基础的自回归文本生成
```

---

# 30. 当前项目边界

本项目主要用于理解语言模型训练闭环，目前暂未实现：

```text
从零训练生产级 BPE / SentencePiece Tokenizer

大规模真实语料清洗与近似去重

高性能动态 Packing

混合精度训练

多 GPU 分布式训练

ZeRO / FSDP

FlashAttention

KV Cache 推理加速

Top-k / Top-p Sampling

Beam Search

生产级训练监控与断点续训
```

这些内容将在后续扩大模型和数据规模时继续学习。

---

# 31. 本周核心收获

完成本阶段后，应能够独立解释：

```text
Raw Text
↓
Tokenizer
↓
Token IDs [B,T]
↓
MiniTransformer
↓
Logits [B,T,V]
↓
Cross Entropy Loss
↓
Backward
↓
Gradient Clipping
↓
AdamW
↓
LR Scheduler
↓
Checkpoint
↓
Autoregressive Generation
```

并理解：

```text
Tokenizer
→ 将原始文本转换成模型可以处理的 Token ID

BPE / SentencePiece
→ 在字符和完整词之间构造可复用的子词单元

Causal LM
→ 根据历史 Token 预测下一个 Token

Shift Labels
→ 为每个输入位置构造下一个 Token 的监督信号

Cross Entropy
→ 衡量模型预测分布与正确 Token 的差异

Padding Mask
→ 排除填充位置对 Attention 和 Loss 的影响

Packing
→ 减少无效 Padding，提高 Token 利用率

Gradient Accumulation
→ 在显存有限时模拟更大的 Global Batch

AdamW
→ 根据梯度更新参数并独立处理 Weight Decay

Warmup
→ 降低训练初期的不稳定风险

Cosine Decay
→ 在训练后期逐步减小参数更新幅度

Gradient Clipping
→ 限制异常梯度对训练的破坏

Validation
→ 检查模型在未参与训练的数据上的表现

Checkpoint
→ 保存和恢复训练或推理所需的状态

Autoregressive Generation
→ 循环使用 Next Token Prediction 能力生成文本
```

本阶段最终目标：

> 不只知道 Transformer 如何完成一次前向传播，还能够从原始文本出发，独立构造训练数据、计算 Causal LM Loss、完成参数更新、保存 Checkpoint，并使用训练后的模型逐 Token 生成文本。
