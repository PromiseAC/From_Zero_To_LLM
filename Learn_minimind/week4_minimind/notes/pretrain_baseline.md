# Week 4 周四：MiniMind Pretrain Baseline

## 一、今日目标

今天的核心不是“把完整预训练跑完”，而是建立一套可复现、可解释的 MiniMind 预训练 Baseline 流程：

```text
真实数据准备
→ Baseline 配置冻结
→ 显存 Probe
→ 确定 micro batch
→ 真实数据 Benchmark
→ 估算吞吐与 ETA
→ 小规模真实数据完整训练
→ Checkpoint / Resume 验证
```

---

## 二、什么是 Baseline

Baseline 就是后续实验比较时的基准实验。

核心思想是控制变量法：

- 先固定模型、数据、序列长度、学习率、优化器、随机种子等配置。
- 后续做实验时一次只改变少量关键变量。
- 这样结果发生变化时，才能判断主要是哪个变量导致的。

例如：

```text
Baseline:
learning_rate = 5e-4
batch = 32
seq_len = 768

Experiment A:
learning_rate = 3e-4
batch = 32
seq_len = 768
```

这时两组实验的主要差异就是 learning rate，可以分析学习率对训练结果的影响。

---

## 三、Micro Batch、Gradient Accumulation 与 Effective Batch

本次计划使用：

```text
micro_batch_size = 8
gradient_accumulation_steps = 4
world_size = 1
```

有效 Batch Size：

```text
effective_batch_size
= micro_batch_size
× gradient_accumulation_steps
× world_size

= 8 × 4 × 1
= 32
```

### 1. Micro Batch

`micro_batch_size = 8` 表示：

> GPU 每次 forward / backward 真正处理 8 条 sequence。

一次输入大致为：

```text
input_ids: [8, 768]
labels:    [8, 768]
```

---

### 2. Gradient Accumulation

`gradient_accumulation_steps = 4` 表示连续进行 4 次 backward 后才更新一次参数：

```text
micro batch 1
→ forward
→ loss / 4
→ backward
→ 累积梯度

micro batch 2
→ forward
→ loss / 4
→ backward
→ 累积梯度

micro batch 3
→ forward
→ loss / 4
→ backward
→ 累积梯度

micro batch 4
→ forward
→ loss / 4
→ backward
→ 累积梯度

→ clip_grad_norm_
→ optimizer.step()
→ optimizer.zero_grad()
```

注意：

```text
backward()
= 计算并累积梯度

optimizer.step()
= 根据已经累积的梯度更新参数
```

不能把二者混为一谈。

---

### 3. 为什么 loss 要除以 accumulation_steps

如果连续 4 次 backward 都使用完整 loss，那么累计梯度大约会被放大 4 倍。

所以通常：

```python
loss = loss / gradient_accumulation_steps
loss.backward()
```

这样 4 次 micro batch 累积后的梯度更接近对 4 个 micro batch 梯度求平均。

---

## 四、为什么 Effective Batch = 32，但显存主要看 Micro Batch = 8

虽然一次 `optimizer.step()` 前总共看过 32 条 sequence，但 GPU 并不会一次把 32 条全部放入显存。

GPU 每次真正执行 forward / backward 的只有：

```text
micro_batch_size = 8
```

因此显存压力主要由单次 micro batch 决定。

Gradient Accumulation 的意义之一就是：

> 用较小的单次显存占用，模拟更大的有效 Batch Size。

---

## 五、什么是 Activation

Activation 是前向传播过程中，各层产生的中间结果。

例如：

```text
input_ids [B,T]
↓ Embedding
x [B,T,D]
```

`x` 就是一种 activation。

Attention 中还会产生：

```text
Q [B,H,T,Dh]
K [B,H,T,Dh]
V [B,H,T,Dh]

attention scores [B,H,T,T]

attention output [B,T,D]
```

FFN 中也会产生：

```text
[B,T,D]
→ [B,T,I]
→ [B,T,D]
```

这些前向传播产生的中间张量，都属于 activation。

训练时不能立即全部删除，因为：

```python
loss.backward()
```

反向传播需要使用前向传播保存的中间结果，根据链式法则计算梯度。

---

## 六、训练显存由什么组成

可以粗略理解为：

```text
总显存
≈ 模型参数
+ 参数梯度
+ AdamW optimizer states
+ activations
+ CUDA runtime / cache
```

其中：

```text
模型参数
参数梯度
AdamW 状态
```

基本不会随着 batch size 成比例增加。

而：

```text
activations
attention 中间张量
每条样本对应的中间结果
```

会明显随着 micro batch 增大。

因此：

> Batch Size 增大时显存会上升，但通常不会简单按 Batch Size 的倍数整体同比增长。

---

## 七、RTX 3060 6GB 显存 Probe

模型配置：

```text
Model Params ≈ 63.91M
hidden_size = 768
num_hidden_layers = 8
max_seq_len = 768
```

实测：

```text
B=2 → 2792 MiB
B=4 → 3590 MiB
B=8 → 5043 MiB
```

结果：

```text
micro_batch_size = 8
```

可以稳定完成 forward、backward 和 AdamW 更新。

RTX 3060 总显存：

```text
6144 MiB
```

B=8 时峰值显存约：

```text
5043 MiB
```

约占 82%，仍保留一定安全余量。

因此不继续追求更极限的 batch，而把 B=8 作为正式 baseline 的 micro batch。

---

## 八、MiniMind Pretrain 完整数据流

真实预训练样本：

```json
{"text": "一段文本……"}
```

Dataset：

```text
text
↓ Tokenizer
token ids
↓ 加 BOS / EOS
↓ truncation / padding
input_ids [T]
labels    [T]
```

DataLoader：

```text
多条 [T]
↓ batch
input_ids [B,T]
labels    [B,T]
```

正式配置：

```text
B = 8
T = 768
```

所以：

```text
input_ids [8,768]
labels    [8,768]
```

模型内部：

```text
input_ids [B,T]
↓ Embedding
[B,T,D]
↓ 8 × Transformer Block
[B,T,D]
↓ LM Head
[B,T,V]
↓ shift
↓ CrossEntropy
loss
```

本模型：

```text
D = 768
V = 6400
```

因此 logits：

```text
[B,T,V]
=
[8,768,6400]
```

Next-token prediction 内部 shift 后大致：

```text
logits: [8,767,6400]
labels: [8,767]
```

再 flatten：

```text
logits: [8×767,6400]
labels: [8×767]
```

然后计算 CrossEntropyLoss。

---

## 九、真实预训练数据

数据文件：

```text
dataset/pretrain_t2t_mini.jsonl
```

实际检查：

```text
大小：约 1.2 GB
样本数：1,270,238
字段：text
格式：JSONL
```

---

## 十、正式 Baseline 配置

实验名称：

```text
pretrain_baseline_001
```

模型：

```text
params ≈ 63.91M
hidden_size = 768
num_hidden_layers = 8
num_attention_heads = 8
num_key_value_heads = 4
intermediate_size = 2432
vocab_size = 6400
```

训练：

```text
epochs = 2
micro_batch_size = 8
gradient_accumulation_steps = 4
world_size = 1
effective_batch_size = 32
max_seq_len = 768
learning_rate = 5e-4
optimizer = AdamW
grad_clip = 1.0
precision = bfloat16
warmup_steps = 0
```

学习率策略：

```text
MiniMind get_lr cosine decay
```

不是独立 scheduler 对象。

---

## 十一、为什么正式训练前要 Benchmark

显存能装下，只代表“能运行”。

还必须知道：

```text
训练速度
吞吐量
完整实验 ETA
```

否则可能直接启动一个需要几十小时的任务。

本次真实数据 Benchmark：

```text
运行时间：184 秒
完成：730 个 micro steps
```

估算：

```text
micro steps/s
≈ 730 / 184
≈ 3.97

sequences/s
≈ 3.97 × 8
≈ 31.7

padded tokens/s
≈ 31.7 × 768
≈ 24.4k
```

注意这里是：

```text
padded tokens/s
```

因为 Dataset 会 pad 到 max_seq_len，不能等价理解为真实有效 token 吞吐。

---

## 十二、完整 Baseline 的 ETA

总样本：

```text
1,270,238
```

micro batch：

```text
8
```

每个 epoch：

```text
micro steps
≈ 1,270,238 / 8
≈ 158,780
```

GA=4：

```text
optimizer steps / epoch
≈ 158,780 / 4
≈ 39,695
```

2 epochs：

```text
total micro steps ≈ 317,560
total optimizer steps ≈ 79,390
```

实际 benchmark 推算：

```text
约 10～11 小时 / epoch
约 20～22 小时 / 2 epochs
```

所以不需要为了学习目的立即完整跑完 20 多小时。

---

## 十三、实验分层

今天实际用了几个不同层级：

### 1. Smoke Test

目的：

```text
验证代码链路能不能跑
```

检查：

```text
Dataset
Forward
Loss
Backward
Optimizer
Checkpoint
```

---

### 2. Memory Probe

目的：

```text
测试显存是否装得下
```

得到：

```text
B=2 → 2792 MiB
B=4 → 3590 MiB
B=8 → 5043 MiB
```

---

### 3. Benchmark

目的：

```text
测真实训练速度
估算完整训练 ETA
```

结果：

```text
≈ 3.97 micro steps/s
≈ 31.7 sequences/s
≈ 24.4k padded tokens/s
```

---

### 4. Learning Baseline

目的：

```text
使用真实数据
完整跑完一个小规模实验
验证 Loss / LR / Checkpoint / Resume
```

从完整数据中截取：

```text
10,000 samples
```

文件：

```text
dataset/pretrain_learning_10k.jsonl
```

大小：

```text
7.2 MB
```

---

## 十四、Learning Baseline 实验结果

实验：

```text
pretrain_learning_001
```

配置：

```text
samples = 10,000
epochs = 1
micro_batch = 8
GA = 4
seq_len = 768
hidden_size = 768
layers = 8
learning_rate = 5e-4
```

总 micro steps：

```text
10000 / 8 = 1250
```

参数更新次数：

```text
1250 / 4 = 312.5
```

当前 MiniMind 会在 epoch 结束时处理 leftover 梯度，所以总 optimizer update 约：

```text
313 次
```

训练结束：

```text
step 1100 loss = 5.2079
step 1150 loss = 5.4748
step 1200 loss = 5.1358
step 1250 loss = 5.1858

final lr = 0.00005000
```

Loss 不要求每一步都下降。

正确观察方式：

> 看整体趋势，而不是要求单步单调递减。

---

## 十五、Checkpoint 结果

生成：

```text
../out/pretrain_learning_001_768.pth
≈ 131 MB

../checkpoints/pretrain_learning_001_768.pth
≈ 131 MB

../checkpoints/pretrain_learning_001_768_resume.pth
≈ 619 MB
```

### 普通 `.pth`

主要保存：

```text
model state_dict
```

适合：

```text
加载模型权重
推理
作为后续训练初始化权重
```

---

### `_resume.pth`

用于真正恢复训练。

包含大致：

```text
model
optimizer
epoch
step
world_size
scaler
wandb_id
...
```

其中 AdamW optimizer state 会保存：

```text
exp_avg
exp_avg_sq
step
```

所以 Resume Checkpoint 明显比普通模型权重大。

注意：

> 当前 MiniMind 没有独立 scheduler 对象，因此不要写“resume checkpoint 保存 scheduler”。

学习率是通过：

```text
get_lr(...)
```

根据 step 动态计算。

---

## 十六、tmux

远程服务器训练建议使用 tmux。

创建：

```bash
tmux new -s pretrain_learning
```

查看：

```bash
tmux ls
```

重新进入：

```bash
tmux attach -t pretrain_learning
```

如果快捷键 detach 不方便，可以直接：

```bash
tmux detach-client
```

tmux 的意义：

```text
Mac SSH 断开
≠
服务器训练进程结束
```

非常适合长时间 GPU 训练。

---

# 十七、今日最重要的工程流程

以后做 LLM 训练实验，可以优先按这个顺序：

```text
1. Smoke Test
   ↓
   验证训练代码链路

2. Memory Probe
   ↓
   找到合理的 micro batch

3. Benchmark
   ↓
   测吞吐量和 ETA

4. 冻结 Baseline
   ↓
   固定实验配置

5. 小规模完整实验
   ↓
   验证 Loss / LR / Checkpoint / Resume

6. Full Training
   ↓
   再决定是否值得跑长时间实验
```

不要一开始就直接启动几十小时的完整训练。

---

# 十八、当日口头总结

今天主要完成了 MiniMind 预训练 Baseline 的设计和验证。

首先，Baseline 本质上就是控制变量法中的基准实验，后续修改学习率、Batch Size 或其他参数时，都应该尽量基于固定 Baseline 比较。

训练中需要区分 micro batch 和 effective batch。本次 micro batch 是 8，gradient accumulation 是 4，world size 是 1，因此 effective batch 是 32。但是 GPU 每次 forward 和 backward 实际只处理 8 条 sequence，所以显存压力主要取决于 micro batch，而不是 effective batch。

Gradient Accumulation 的过程是连续做 4 次 backward 累积梯度，再执行一次 optimizer.step 更新参数。backward 是计算和累积梯度，optimizer.step 才是真正更新参数。

训练显存除了模型参数，还包括参数梯度、AdamW optimizer state、activation 和 CUDA cache。Batch Size 增大时，模型参数不会复制，但每条数据都会产生自己的 activation，所以显存会上升。

实际显存测试中，B=2 使用约 2792 MiB，B=4 约 3590 MiB，B=8 约 5043 MiB，因此 RTX 3060 6GB 可以使用 micro batch 8。

真实数据集 pretrain_t2t_mini 有约 127 万条样本。Benchmark 184 秒跑了 730 个 micro steps，大约是 3.97 micro steps/s、31.7 sequences/s 和 24.4k padded tokens/s。完整跑 2 epochs 大约需要 20～22 小时。

因此没有直接盲目跑完整实验，而是抽取 10k 真实数据跑了一个完整 Learning Baseline。共 1250 个 micro steps，大约 313 次 optimizer update，最终 loss 约 5.19，learning rate 从接近 5e-4 cosine decay 到 5e-5。

最后生成了普通模型 checkpoint 和 resume checkpoint。普通权重约 131MB，resume checkpoint 约 619MB，因为后者还包含 AdamW optimizer state、epoch、step 等训练恢复信息。

今天最重要的流程是：

Smoke Test → Memory Probe → Benchmark → Baseline → 小规模完整实验 → 再决定是否 Full Training。

---

# 十九、最短复习版

```text
Baseline
= 固定配置的基准实验
= 控制变量法

effective batch
= micro batch × GA × world size

本次：
8 × 4 × 1 = 32

backward()
= 计算 / 累积梯度

optimizer.step()
= 更新参数

显存主要看 micro batch

训练显存 ≈
参数
+ 梯度
+ optimizer states
+ activation
+ CUDA cache

显存 Probe：
B=2 → 2792 MiB
B=4 → 3590 MiB
B=8 → 5043 MiB

选：
micro batch = 8

真实数据：
1,270,238 samples
1.2 GB

Benchmark：
730 micro steps / 184 s
≈ 3.97 micro steps/s
≈ 31.7 seq/s
≈ 24.4k padded tokens/s

完整 2 epochs ETA：
≈ 20～22 h

Learning Baseline：
10,000 samples
1250 micro steps
≈313 optimizer steps
final loss ≈ 5.1858
final lr = 5e-5

Checkpoint：
普通 pth ≈ 131 MB
resume pth ≈ 619 MB

训练实验流程：
Smoke Test
→ Memory Probe
→ Benchmark
→ Baseline
→ 小规模完整实验
→ Full Training
```
