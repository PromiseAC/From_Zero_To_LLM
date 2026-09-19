# Week 4 Day 3 - MiniMind Dataset 与 Training Loop

## 0. 中文复习总结

今天的目标不是重新学习 PyTorch 训练循环，而是把第三周已经学过的：

```text
Dataset / DataLoader
→ input_ids / labels
→ Forward
→ logits
→ Next-token Loss
→ backward
→ Gradient Accumulation
→ Gradient Clipping
→ AdamW
→ Learning Rate
→ Checkpoint / Resume
```

映射到 MiniMind 的真实预训练源码中。

完整主线：

```text
JSON 数据
↓
PretrainDataset
↓
DataLoader
↓
input_ids [B,T]
labels [B,T]
↓
MiniMindForCausalLM
↓
logits [B,T,V]
↓
Next-token Shift
↓
CrossEntropy
↓
loss
↓
loss / accumulation_steps
↓
backward()
↓
累积 N 个 micro batch 的梯度
↓
unscale
↓
clip_grad_norm_
↓
optimizer.step()
↓
zero_grad()
↓
Checkpoint / Resume
```

今天最重要的几个结论：

```text
Dataset 单个样本       → [T]
DataLoader 一个 batch  → [B,T]

input_ids / labels     → [B,T]
hidden_states          → [B,T,D]
logits                 → [B,T,V]
loss                   → scalar
```

MiniMind 的 `PretrainDataset` 不提前做 next-token shift。

它返回：

```text
input_ids [B,T]
labels    [B,T]
```

其中：

```text
labels ≈ input_ids.clone()
```

但 PAD 位置会被替换成：

```text
-100
```

真正的 next-token shift 在模型内部：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

于是：

```text
x [B,T-1,V]
y [B,T-1]
```

Gradient Accumulation 中一定要区分：

```text
micro step
≠
optimizer.step()
```

`backward()` 负责计算并累积梯度：

```text
parameter.grad
```

而：

```python
optimizer.step()
```

才真正修改模型参数。

MiniMind 默认：

```text
batch_size = 32
accumulation_steps = 8
```

所以单进程下：

```text
8 个 micro batch
×
每个 32 条 sequence
=
256 条 sequence
```

共同形成一次正常的 optimizer update。

更新顺序必须记住：

```text
forward
↓
loss / accumulation_steps
↓
backward
↓
累计 N 次
↓
unscale
↓
clip_grad_norm_
↓
optimizer.step()
↓
zero_grad()
```

MiniMind 当前的学习率实现不是独立的 `scheduler.step()`，而是每个 micro step 调用：

```python
get_lr(...)
```

直接把当前学习率写进 optimizer。

当前 `get_lr()` 是：

```text
Cosine Decay
```

没有显式 Warmup，并且最终学习率下降到 base learning rate 的约 10%。

Resume Training 也不能只恢复：

```text
model.state_dict()
```

因为 AdamW 还维护：

```text
m
v
optimizer step
```

等历史状态。

因此完整 resume 需要尽量恢复：

```text
model
optimizer
scaler
epoch
step
...
```

可以把 Resume Checkpoint 理解为：

```text
训练现场快照
```

---

# 1. 关键源码位置

Dataset：

```text
dataset/lm_dataset.py
```

预训练 Dataset：

```python
class PretrainDataset(Dataset):
```

预训练入口：

```text
trainer/train_pretrain.py
```

Checkpoint / Learning Rate 等工具：

```text
trainer/trainer_utils.py
```

模型：

```text
model/model_minimind.py
```

---

# 2. PretrainDataset

核心逻辑：

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset(
            'json',
            data_files=data_path,
            split='train'
        )
```

说明 MiniMind 预训练数据来自：

```text
JSON / JSONL
```

每个样本主要读取：

```python
sample['text']
```

概念上类似：

```json
{
  "text": "这里是一段用于预训练的文本。"
}
```

---

# 3. Tokenizer 与 BOS / EOS

Dataset 中：

```python
tokens = self.tokenizer(
    str(sample['text']),
    add_special_tokens=False,
    max_length=self.max_length - 2,
    truncation=True
).input_ids
```

然后手动加入：

```python
tokens = [
    self.tokenizer.bos_token_id
] + tokens + [
    self.tokenizer.eos_token_id
]
```

所以：

```text
正文最大长度
=
max_length - 2
```

预留两个位置给：

```text
BOS
EOS
```

例如：

```text
max_length = 340
```

则正文最多：

```text
338 tokens
```

再加：

```text
BOS + EOS
```

总长度最多：

```text
340
```

这与周一验证的 tokenizer 行为一致：

```text
普通 encode
不会自动加入 BOS / EOS
```

所以 Dataset 手动添加。

---

# 4. Padding

如果实际 token 数不足 `max_length`：

```python
input_ids = tokens + [
    self.tokenizer.pad_token_id
] * (self.max_length - len(tokens))
```

例如：

```text
[BOS, 10, 20, 30, EOS]
```

补成：

```text
[BOS, 10, 20, 30, EOS, PAD, PAD, ...]
```

这样不同长度的样本才能组成统一 Tensor。

---

# 5. Dataset 与 DataLoader 的 Shape

这是今天非常容易混淆的一点。

Dataset 一次：

```python
__getitem__()
```

只返回一个样本。

所以：

```text
input_ids [T]
labels    [T]
```

不是：

```text
[B,T]
```

例如 MiniMind 默认：

```text
max_seq_len = 340
```

单个样本：

```text
input_ids [340]
labels    [340]
```

DataLoader 才会把多个样本组成 batch。

默认：

```text
batch_size = 32
```

因此：

```text
input_ids [32,340]
labels    [32,340]
```

必须牢记：

```text
Dataset
→ [T]

DataLoader
→ [B,T]
```

---

# 6. Labels

Dataset 中：

```python
labels = input_ids.clone()
```

所以一开始：

```text
input_ids:
[BOS, 10, 20, 30, EOS, PAD, PAD]

labels:
[BOS, 10, 20, 30, EOS, PAD, PAD]
```

然后：

```python
labels[
    input_ids == self.tokenizer.pad_token_id
] = -100
```

得到：

```text
input_ids:
[BOS, 10, 20, 30, EOS, PAD, PAD]

labels:
[BOS, 10, 20, 30, EOS, -100, -100]
```

注意：

```text
input_ids 中 PAD 仍然存在
```

只有 labels 中 PAD 位置变成：

```text
-100
```

---

# 7. 为什么用 -100

模型内部 CrossEntropy：

```python
F.cross_entropy(
    ...,
    ignore_index=-100
)
```

所以：

```text
label = -100
```

的位置不会参与 Loss。

因此：

```text
PAD Token
→ 不计算训练损失
```

---

# 8. Dataset 不做 Next-token Shift

MiniMind Dataset 返回：

```text
input_ids [B,T]
labels    [B,T]
```

Dataset 中没有提前做：

```text
input_ids[:-1]
labels[1:]
```

真正 shift 在：

```text
MiniMindForCausalLM.forward()
```

中完成：

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
```

所以：

```text
logits
[B,T,V]

↓

x
[B,T-1,V]
```

而：

```text
labels
[B,T]

↓

y
[B,T-1]
```

---

# 9. Next-token Prediction 示例

假设 Dataset：

```text
input_ids:
[BOS, A, B, C, EOS, PAD]

labels:
[BOS, A, B, C, EOS, -100]
```

模型 shift 后：

```text
预测位置          Target

BOS       →       A
A         →       B
B         →       C
C         →       EOS
EOS       →       -100
```

所以最后一项不计算 Loss。

这就是：

```text
Causal Language Modeling
=
Next-token Prediction
```

---

# 10. DataLoader 到 Model

训练循环：

```python
for step, (input_ids, labels) in enumerate(loader, ...):
```

一个 micro batch：

```text
input_ids [B,T]
labels    [B,T]
```

默认：

```text
[32,340]
[32,340]
```

然后：

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
```

只是移动设备，Shape 不改变。

---

# 11. Forward 完整 Shape

输入：

```text
input_ids
[B,T]
```

进入 MiniMind：

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

默认：

```text
B = 32
T = 340
D = 768
V = 6400
```

因此：

```text
input_ids
[32,340]

↓ Embedding

[32,340,768]

↓ 8 × Transformer Block

[32,340,768]

↓ Final RMSNorm

[32,340,768]

↓ LM Head

logits
[32,340,6400]
```

---

# 12. Next-token Loss Shape

模型内部：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
```

所以：

```text
x
[32,339,6400]

y
[32,339]
```

然后：

```text
x
[32×339,6400]

y
[32×339]
```

再送入：

```text
CrossEntropy
```

最终：

```text
loss
scalar
```

---

# 13. 四种 Tensor 必须区分

```text
input_ids
[B,T]
```

每个位置是一个：

```text
Token ID
```

---

```text
labels
[B,T]
```

每个位置是目标 Token ID，或者：

```text
-100
```

---

```text
hidden_states
[B,T,D]
```

每个位置是一个 D 维向量。

---

```text
logits
[B,T,V]
```

每个位置是对整个 vocabulary 的 V 个预测分数。

---

# 14. Training Loop 主体

MiniMind：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()
```

这里可以拆成：

```text
Forward
↓
Language Model Loss
↓
MoE Aux Loss（如果使用 MoE）
↓
Total Loss
↓
除以 accumulation_steps
↓
backward
```

默认：

```text
use_moe = False
```

所以主线主要是：

```text
Causal LM CrossEntropy Loss
```

---

# 15. Gradient Accumulation

默认参数：

```text
batch_size = 32
accumulation_steps = 8
```

MiniMind 每一个 DataLoader batch 都会：

```text
forward
↓
loss / 8
↓
backward
```

但不会每个 batch 都更新参数。

流程：

```text
micro step 1
→ backward

micro step 2
→ backward

...

micro step 8
→ backward
→ optimizer.step()
→ zero_grad()
```

因此：

```text
8 个 micro step
→ 1 次 optimizer update
```

---

# 16. micro step 与 optimizer.step()

一定要分开：

```text
micro step
=
DataLoader 处理了一个 batch
```

而：

```text
optimizer.step()
=
模型参数真正更新一次
```

所以：

```text
step
≠
optimizer.step()
```

Gradient Accumulation 的情况下，一个 optimizer update 之前会有多次 backward。

---

# 17. Effective Batch Size

如果：

```text
batch_size = 32
accumulation_steps = 8
world_size = 1
```

那么：

```text
effective batch size
=
32 × 8
=
256 sequences
```

即：

```text
micro step 1 → 32
micro step 2 → 32
...
micro step 8 → 32

总计：
256 条 sequence

↓

一次 optimizer.step()
```

如果 DDP：

```text
global batch size
=
batch_size
× accumulation_steps
× world_size
```

例如：

```text
32 × 8 × 2
=
512
```

---

# 18. backward() 到底做什么

```python
loss.backward()
```

并不会直接修改模型参数。

它负责：

```text
计算梯度
+
把梯度累积到 parameter.grad
```

如果连续执行：

```text
backward()
backward()
backward()
```

而中间没有：

```text
zero_grad()
```

梯度就会继续累加。

这正是 Gradient Accumulation 的基础。

---

# 19. optimizer.step() 到底做什么

真正修改参数的是：

```python
optimizer.step()
```

因此必须区分：

```text
backward()
→ 算梯度 / 累积梯度

optimizer.step()
→ 根据梯度更新模型参数

zero_grad()
→ 清空梯度
```

---

# 20. 为什么 Loss 要除 accumulation_steps

源码：

```python
loss = loss / args.accumulation_steps
```

如果累积：

```text
8 个 micro batch
```

而每一个 Loss 都不除以 8，那么累计梯度的尺度会大约放大 8 倍。

因此：

```text
loss / 8
↓ backward

loss / 8
↓ backward

...
```

最终更接近多个 micro batch 平均 Loss 的梯度。

---

# 21. 参数更新条件

MiniMind：

```python
if step % args.accumulation_steps == 0:
```

默认：

```text
accumulation_steps = 8
```

所以：

```text
step 8
step 16
step 24
...
```

会进行参数更新。

第一次：

```text
step 1 ~ 8
```

共有：

```text
8 次 backward
```

然后：

```text
1 次 optimizer.step
```

---

# 22. 参数更新完整顺序

MiniMind：

```python
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)

scaler.step(optimizer)
scaler.update()

optimizer.zero_grad(set_to_none=True)
```

顺序：

```text
累计 N 次 backward
↓
unscale
↓
clip_grad_norm_
↓
optimizer.step()
↓
scaler.update()
↓
zero_grad()
```

必须记住这个顺序。

---

# 23. Gradient Clipping

默认：

```text
grad_clip = 1.0
```

源码：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)
```

MiniMind 在准备参数更新时都会调用这个函数。

如果：

```text
global grad norm <= 1.0
```

梯度基本不会被改变。

如果：

```text
global grad norm > 1.0
```

梯度会整体按比例缩小，使范数受到限制。

注意它不是：

```text
把每一个梯度元素都裁剪到 [-1,1]
```

而是：

```text
Global Gradient Norm Clipping
```

---

# 24. GradScaler

MiniMind：

```python
scaler = torch.cuda.amp.GradScaler(
    enabled=(args.dtype == 'float16')
)
```

默认：

```text
dtype = bfloat16
```

所以默认情况下：

```text
GradScaler disabled
```

如果使用 FP16：

```text
loss
↓ scale
scaled loss
↓ backward
scaled gradients
↓ unscale
真实 gradients
↓ gradient clipping
```

所以必须：

```text
先 unscale
再 clip
```

---

# 25. zero_grad()

完成参数更新后：

```python
optimizer.zero_grad(set_to_none=True)
```

用于清空本轮累计梯度。

如果在每个 micro batch 后都：

```python
zero_grad()
```

那么前面 batch 的梯度就会丢失，Gradient Accumulation 失效。

正确：

```text
batch 1 → backward
batch 2 → backward
...
batch 8 → backward
↓
optimizer.step()
↓
zero_grad()
```

---

# 26. Epoch 最后不足 accumulation_steps

MiniMind 还处理了 epoch 最后的剩余梯度。

例如：

```text
accumulation_steps = 4
一个 epoch = 10 个 batch
```

流程：

```text
batch 1~4
→ optimizer.step() # 第1次

batch 5~8
→ optimizer.step() # 第2次

batch 9~10
→ epoch 结束
→ 再 optimizer.step() # 第3次
```

所以剩余 batch 不会直接丢弃。

源码最后：

```python
if last_step > start_step and \
   last_step % args.accumulation_steps != 0:
```

会对剩余梯度做一次参数更新。

---

# 27. 剩余 batch 的一个实现细节

即使最后只剩：

```text
2 个 batch
```

而：

```text
accumulation_steps = 4
```

MiniMind 前面仍然执行：

```python
loss = loss / 4
```

不会因为最后只有 2 个 batch 就临时改成：

```text
loss / 2
```

所以最后一次 update 的梯度尺度会比完整累积组更小。

这是当前源码的真实行为。

---

# 28. AdamW

MiniMind：

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)
```

AdamW 不仅依赖当前 gradient。

它还维护历史状态：

```text
m
→ 一阶动量 / 梯度历史

v
→ 二阶动量 / 梯度平方历史

step
→ optimizer 已更新次数
```

因此：

```text
AdamW
=
当前梯度
+
历史优化状态
```

---

# 29. Learning Rate

MiniMind 没有创建独立：

```text
scheduler
```

而是在每个 micro step 手动调用：

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)
```

然后：

```python
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

也就是：

```text
当前 micro step
↓
get_lr()
↓
计算 LR
↓
写入 optimizer
```

---

# 30. get_lr()

源码：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (
        0.1 +
        0.45 * (
            1 + math.cos(
                math.pi * current_step / total_steps
            )
        )
    )
```

这是：

```text
Cosine Decay
```

没有显式 Warmup。

大致：

```text
训练开始
≈ 1.0 × base_lr

训练中间
≈ 0.55 × base_lr

训练结束
≈ 0.1 × base_lr
```

默认：

```text
base_lr = 5e-4
```

所以：

```text
开始 ≈ 5e-4
中间 ≈ 2.75e-4
结束 ≈ 5e-5
```

---

# 31. LR 按 micro step 前进

注意：

```python
get_lr(...)
```

每一个 micro batch 都会执行。

如果：

```text
accumulation_steps = 8
```

则：

```text
step 1 → lr1 → 不更新参数
step 2 → lr2 → 不更新参数
...
step 7 → lr7 → 不更新参数
step 8 → lr8 → optimizer.step()
```

所以这次 optimizer update 实际使用的是：

```text
lr8
```

当前 MiniMind 的学习率 schedule 是按：

```text
micro step
```

前进，而不是按：

```text
optimizer update 次数
```

前进。

---

# 32. Checkpoint 保存

MiniMind：

```python
if (
    step % args.save_interval == 0
    or step == iters
):
```

默认：

```text
save_interval = 1000
```

所以不是每个 step 都保存。

通常：

```text
step 1000
step 2000
...
epoch 最后
```

保存。

---

# 33. 普通模型权重与 Resume Checkpoint

普通模型权重：

```text
*.pth
```

主要表示：

```text
model.state_dict()
```

可以理解为：

```text
模型当前参数是什么
```

Resume Checkpoint：

```text
*_resume.pth
```

更像：

```text
训练现场快照
```

包含：

```text
model
optimizer
scaler
epoch
step
world_size
wandb_id
...
```

---

# 34. 为什么不能只恢复 Model

如果只：

```python
model.load_state_dict(...)
```

模型权重虽然回来了，但是 AdamW 的：

```text
m
v
optimizer step
```

等历史状态会丢失。

所以这不算完整 Resume。

MiniMind 恢复：

```python
model.load_state_dict(
    ckp_data['model']
)

optimizer.load_state_dict(
    ckp_data['optimizer']
)

scaler.load_state_dict(
    ckp_data['scaler']
)
```

同时恢复：

```text
epoch
step
```

目标是：

```text
尽可能恢复上次中断时的完整训练状态
```

---

# 35. Resume Position

Checkpoint 中：

```text
epoch
step
```

用于知道训练进行到哪里。

恢复时：

```python
start_epoch = ckp_data['epoch']
start_step = ckp_data.get('step', 0)
```

并配合：

```text
SkipBatchSampler
```

跳过已经训练过的 batch。

例如：

```text
checkpoint:
epoch = 1
step = 500
```

恢复后不会简单重新从：

```text
epoch 1, step 1
```

开始，而是尽量从之前的训练位置继续。

---

# 36. 今日完整中文口头总结

下面这段可以直接作为周三复习时的口头总结：

> 原始 JSON 数据首先经过 `PretrainDataset`。单个样本会被 Tokenizer 转成固定长度的 `input_ids [T]` 和 `labels [T]`，其中 labels 基本复制 input_ids，但 PAD 位置改为 `-100`。DataLoader 再把多个样本组成一个 batch，因此得到 `input_ids [B,T]` 和 `labels [B,T]`。
>
> `input_ids` 输入 MiniMind 后，先经过 Embedding 得到 `[B,T,D]`，再经过多个 Transformer Block。每个 Block 内部依次进行 RMSNorm、Attention、Residual、RMSNorm、SwiGLU FeedForward、Residual，因此 Block 的输入输出都保持 `[B,T,D]`。经过所有 Block 和 Final RMSNorm 后仍为 `[B,T,D]`，再经过 LM Head 将 `D` 投影到 vocabulary size `V`，得到 logits `[B,T,V]`。
>
> 模型内部用 `logits[:, :-1, :]` 与 `labels[:, 1:]` 做 next-token prediction，经过 CrossEntropy 得到一个标量 loss。每个 micro batch 都会执行 `loss / accumulation_steps` 和 `backward()` 来累积梯度；当 `step % accumulation_steps == 0` 时，先 unscale，再进行 gradient clipping，然后执行 `optimizer.step()` 真正更新参数，最后 `zero_grad()` 清空梯度。
>
> 学习率由 `get_lr()` 按 micro step 计算并写入 optimizer。到达保存间隔或 epoch 末尾时，会保存模型权重以及用于恢复训练的 resume checkpoint，其中包含 model、optimizer、scaler、epoch、step 等训练状态。
>
> Gradient Accumulation 中需要特别区分 micro step 和 optimizer step。一个 micro step 只是处理一个 batch 并执行一次 backward，而 optimizer.step() 才真正修改模型参数。例如 batch size 为 32、accumulation steps 为 8 时，会连续处理 8 个 batch、执行 8 次 backward，再执行一次 optimizer.step()，也就是单进程下大约汇总 256 条 sequence 的梯度完成一次参数更新。
>
> Resume Training 也不能只加载 model 权重，因为 AdamW 还维护 m、v 和 optimizer step 等历史状态。因此完整恢复训练时还要加载 optimizer、scaler、epoch 和 step 等信息，目标是尽可能恢复上一次中断时的完整训练状态。

---

# 37. 完整 Shape Flow

```text
JSON Sample
↓
Tokenizer

input_ids [T]
labels    [T]

↓ DataLoader

input_ids [B,T]
labels    [B,T]

↓ MiniMind

Embedding
[B,T,D]

↓ Transformer Block × N

[B,T,D]

↓ Final RMSNorm

[B,T,D]

↓ LM Head

logits
[B,T,V]

↓ Next-token Shift

x [B,T-1,V]
y [B,T-1]

↓ Flatten

x [B×(T-1),V]
y [B×(T-1)]

↓ CrossEntropy

loss
scalar

↓ / accumulation_steps

scaled training loss

↓ backward × N

accumulated gradients

↓ unscale
↓ clip_grad_norm_
↓ optimizer.step()
↓ zero_grad()

parameters updated
```

---

# 38. MiniMind 默认训练参数

```text
epochs              = 2
batch_size          = 32
learning_rate       = 5e-4
accumulation_steps  = 8
grad_clip           = 1.0
max_seq_len         = 340
hidden_size         = 768
num_hidden_layers   = 8
dtype               = bfloat16
save_interval       = 1000
seed                = 42
use_moe             = False
```

单进程 Effective Batch：

```text
32 × 8
=
256 sequences
```

---

# 39. 常见易错点

### 易错点 1

错误：

```text
input_ids [B,T,D]
```

正确：

```text
input_ids [B,T]
```

Embedding 后才是：

```text
[B,T,D]
```

---

### 易错点 2

错误：

```text
labels [B,T,V]
```

正确：

```text
labels [B,T]
```

每个位置只存一个目标 Token ID。

---

### 易错点 3

错误：

```text
SwiGLU 后直接得到 logits
```

正确：

```text
SwiGLU / Block
→ [B,T,D]

Final RMSNorm
→ [B,T,D]

LM Head
→ [B,T,V]
```

---

### 易错点 4

错误：

```text
backward()
=
更新模型参数
```

正确：

```text
backward()
=
计算 / 累积梯度
```

真正更新：

```text
optimizer.step()
```

---

### 易错点 5

错误：

```text
一个 DataLoader step
=
一次 optimizer.step()
```

使用 Gradient Accumulation 时：

```text
多个 micro step
→ 一次 optimizer.step()
```

---

### 易错点 6

错误：

```text
MiniMind 当前 Pretrain 使用 Warmup + Cosine
```

当前源码中的 `get_lr()` 是：

```text
Cosine Decay
```

没有显式 Warmup。

---

### 易错点 7

错误：

```text
只恢复 model.state_dict()
就等于完整 Resume
```

正确：

```text
完整 Resume
还要尽量恢复：
optimizer
scaler
epoch
step
...
```

---

# 40. 最短复习版

如果只剩一分钟，记住：

```text
JSON
↓
PretrainDataset
[T]

↓ DataLoader

input_ids [B,T]
labels [B,T]

↓ Model

hidden_states [B,T,D]

↓ LM Head

logits [B,T,V]

↓ Shift

x [B,T-1,V]
y [B,T-1]

↓ CrossEntropy

loss

↓ loss / accumulation_steps
↓ backward × N

累计梯度

↓ unscale
↓ clip
↓ optimizer.step
↓ zero_grad

更新参数

↓ checkpoint

保存模型和训练状态
```

再记一句：

```text
backward() = 算梯度
optimizer.step() = 更新参数
```

以及：

```text
micro step ≠ optimizer step
```

默认：

```text
batch_size = 32
accumulation_steps = 8

8 个 micro batch
→ 8 次 backward
→ 256 条 sequence
→ 1 次 optimizer.step()
```
