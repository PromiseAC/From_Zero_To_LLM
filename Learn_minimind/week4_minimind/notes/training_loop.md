# Week 4 Day 3 - MiniMind Dataset & Training Loop

## 1. Today's Goal

Today the goal is not to relearn PyTorch training from scratch, but to map the concepts learned in Week 3 onto MiniMind's real pretraining code.

Main path:

```text
JSON Data
↓
PretrainDataset
↓
DataLoader
↓
input_ids / labels
↓
MiniMindForCausalLM
↓
logits
↓
Next-token CrossEntropy
↓
backward
↓
Gradient Accumulation
↓
Gradient Clipping
↓
optimizer.step()
↓
zero_grad()
↓
Checkpoint / Resume
```

---

# 2. PretrainDataset

Source:

```text
dataset/lm_dataset.py
```

Core class:

```python
class PretrainDataset(Dataset):
```

MiniMind reads pretraining data through:

```python
self.samples = load_dataset(
    'json',
    data_files=data_path,
    split='train'
)
```

Each sample uses:

```python
sample['text']
```

So a pretraining sample is essentially plain text.

Example conceptually:

```json
{
  "text": "这是一段用于语言模型预训练的文本。"
}
```

---

# 3. Tokenization and Fixed Length

Core logic:

```python
tokens = self.tokenizer(
    str(sample['text']),
    add_special_tokens=False,
    max_length=self.max_length - 2,
    truncation=True
).input_ids

tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
```

Why `max_length - 2`?

Because MiniMind manually adds:

```text
BOS
+
text tokens
+
EOS
```

If:

```text
max_length = 340
```

then the text itself can use at most:

```text
338 tokens
```

and after adding BOS and EOS the maximum becomes:

```text
340 tokens
```

If the sample is shorter, MiniMind pads it:

```python
input_ids = tokens + [
    self.tokenizer.pad_token_id
] * (self.max_length - len(tokens))
```

Therefore one sample always has a fixed sequence length.

---

# 4. Dataset Shape

A very important distinction:

`Dataset.__getitem__()` returns one sample, not one batch.

Single sample:

```text
input_ids [T]
labels    [T]
```

With the default pretrain setting:

```text
T = max_seq_len = 340
```

so one item is:

```text
input_ids [340]
labels    [340]
```

After `DataLoader` batches multiple samples:

```text
input_ids [B,T]
labels    [B,T]
```

For the default:

```text
batch_size = 32
max_seq_len = 340
```

one full batch is approximately:

```text
input_ids [32,340]
labels    [32,340]
```

Remember:

```text
Dataset
→ one sample [T]

DataLoader
→ one batch [B,T]
```

---

# 5. Labels and PAD Masking

MiniMind creates labels using:

```python
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```

Example:

```text
input_ids:
[BOS, 10, 20, 30, EOS, PAD, PAD]

labels:
[BOS, 10, 20, 30, EOS, -100, -100]
```

Important:

```text
input_ids keeps PAD
labels changes PAD to -100
```

Why `-100`?

Because the model later uses:

```python
F.cross_entropy(..., ignore_index=-100)
```

So PAD positions do not contribute to the language-model loss.

---

# 6. Dataset Does NOT Shift Labels

The Dataset does not create:

```text
x = input_ids[:-1]
y = input_ids[1:]
```

Instead it returns:

```text
input_ids [B,T]
labels    [B,T]
```

The next-token shift happens inside:

```text
MiniMindForCausalLM.forward()
```

with:

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
```

Therefore:

```text
logits [B,T,V]
↓
x [B,T-1,V]

labels [B,T]
↓
y [B,T-1]
```

Example:

```text
input_ids:
[BOS, A, B, C, EOS, PAD]

labels:
[BOS, A, B, C, EOS, -100]
```

After shifting:

```text
prediction position   target

BOS                → A
A                  → B
B                  → C
C                  → EOS
EOS                → -100
```

This is standard causal next-token prediction.

---

# 7. DataLoader to Model

The training loop receives:

```python
for step, (input_ids, labels) in enumerate(loader, ...):
```

So one training micro-batch contains:

```text
input_ids [B,T]
labels    [B,T]
```

Then:

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
```

Only the device changes; shape does not.

---

# 8. Forward Path and Tensor Shapes

MiniMind receives:

```text
input_ids [B,T]
```

Then:

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

With MiniMind's default model configuration:

```text
D = 768
V = 6400
N = 8
```

and with the default training batch:

```text
B = 32
T = 340
```

the main shapes are:

```text
input_ids
[32,340]

↓ Embedding

[32,340,768]

↓ 8 × MiniMindBlock

[32,340,768]

↓ Final RMSNorm

[32,340,768]

↓ LM Head

logits
[32,340,6400]
```

Inside each Transformer Block:

```text
RMSNorm
↓
Attention
↓
Residual
↓
RMSNorm
↓
SwiGLU FeedForward
↓
Residual
```

The Block output remains:

```text
[B,T,D]
```

The `LM Head` is what changes the last dimension:

```text
D → V
```

and produces the actual logits.

---

# 9. Next-token Loss

The model internally performs:

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
```

Therefore, with:

```text
B = 32
T = 340
V = 6400
```

we get:

```text
x [32,339,6400]
y [32,339]
```

Then:

```python
x.view(-1, x.size(-1))
y.view(-1)
```

becomes:

```text
x [32×339, 6400]
y [32×339]
```

Finally:

```text
CrossEntropy
↓
scalar loss
```

Important distinction:

```text
input_ids / labels  [B,T]
hidden_states       [B,T,D]
logits              [B,T,V]
loss                scalar
```

---

# 10. Total Loss

Training uses:

```python
res = model(input_ids, labels=labels)

loss = res.loss + res.aux_loss
```

Where:

```text
res.loss
→ causal language-model CrossEntropy loss

res.aux_loss
→ MoE auxiliary/router loss
```

The default MiniMind pretrain configuration has:

```text
use_moe = False
```

so the main path is the standard language-model loss.

---

# 11. Gradient Accumulation

Default:

```text
batch_size = 32
accumulation_steps = 8
```

MiniMind first does:

```python
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()
```

This means every micro-batch performs a backward pass, but parameters are not updated after every micro-batch.

The key distinction:

```text
micro step
→ process one DataLoader batch

backward()
→ calculate / accumulate gradients

optimizer.step()
→ actually update model parameters
```

These are not the same thing.

With:

```text
batch_size = 32
accumulation_steps = 8
```

the process is:

```text
micro step 1
32 sequences
→ forward
→ loss / 8
→ backward

micro step 2
32 sequences
→ forward
→ loss / 8
→ backward

...

micro step 8
32 sequences
→ forward
→ loss / 8
→ backward

↓
optimizer.step()
```

Therefore:

```text
8 micro-batches
×
32 sequences per micro-batch
=
256 sequences
```

contribute to one normal parameter update on a single process.

So:

```text
effective batch size
=
micro_batch_size
× accumulation_steps
```

Single process:

```text
32 × 8 = 256 sequences
```

With DDP:

```text
global batch size
=
micro_batch_size
× accumulation_steps
× world_size
```

---

# 12. Why Divide Loss by accumulation_steps?

MiniMind uses:

```python
loss = loss / args.accumulation_steps
```

before every backward.

Without this division, accumulating 8 micro-batches would approximately make the accumulated gradient 8 times larger than the average-gradient version.

Conceptually:

```text
micro batch 1 → loss / 8 → backward
micro batch 2 → loss / 8 → backward
...
micro batch 8 → loss / 8 → backward
```

The gradients accumulate in:

```text
parameter.grad
```

until the optimizer update.

---

# 13. Complete Parameter Update Order

This is the most important order to remember:

```text
forward
↓
loss
↓
loss / accumulation_steps
↓
backward
↓
accumulate gradients for N micro-batches
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

MiniMind code:

```python
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        args.grad_clip
    )

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
```

---

# 14. Gradient Clipping

Default:

```text
grad_clip = 1.0
```

Before an optimizer update MiniMind calls:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)
```

This should be understood precisely:

MiniMind calls `clip_grad_norm_()` whenever it is about to update parameters.

The function then effectively behaves as:

```text
global grad norm <= threshold
→ gradients remain essentially unchanged

global grad norm > threshold
→ gradients are rescaled
```

It is NOT element-wise clipping such as:

```text
each gradient value clipped to [-1,1]
```

Instead, it controls the global gradient norm.

---

# 15. Mixed Precision and GradScaler

MiniMind defines:

```python
scaler = torch.cuda.amp.GradScaler(
    enabled=(args.dtype == 'float16')
)
```

So GradScaler is mainly active for:

```text
float16
```

The default pretrain dtype is:

```text
bfloat16
```

Therefore, in the default configuration GradScaler is disabled.

When FP16 scaling is active, the order matters:

```text
scaled loss
↓
backward
↓
scaled gradients
↓
scaler.unscale_(optimizer)
↓
real gradients
↓
clip_grad_norm_
↓
optimizer step
```

Gradient clipping should operate on unscaled gradients.

---

# 16. optimizer.step() and zero_grad()

`backward()` does not update model parameters.

It only calculates and accumulates gradients.

The actual parameter update happens at:

```python
scaler.step(optimizer)
```

which corresponds to the optimizer performing its update.

After the update:

```python
optimizer.zero_grad(set_to_none=True)
```

clears gradients so the next accumulation cycle can start from zero.

Therefore:

```text
backward()
→ calculate / accumulate gradients

optimizer.step()
→ modify parameters

zero_grad()
→ clear gradients
```

---

# 17. Leftover Micro-batches at Epoch End

Suppose:

```text
accumulation_steps = 4
```

and one epoch contains:

```text
10 micro-batches
```

The flow is:

```text
batch 1~4
→ optimizer.step() # update 1

batch 5~8
→ optimizer.step() # update 2

batch 9~10
→ epoch ends with remaining gradients
→ optimizer.step() # update 3
```

MiniMind explicitly handles this:

```python
if last_step > start_step and \
   last_step % args.accumulation_steps != 0:

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(...)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

So leftover gradients are not simply discarded.

Important implementation detail:

MiniMind still divides every micro-batch loss by the configured:

```text
accumulation_steps
```

even if the final accumulation group contains fewer micro-batches.

For example, if only 2 batches remain while:

```text
accumulation_steps = 4
```

those two losses are still divided by 4.

---

# 18. Learning Rate Schedule

MiniMind does not use a separate scheduler object in this pretraining script.

Instead, every micro step calculates the learning rate manually:

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)

for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

The function is:

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

This is cosine decay.

It does not contain a warmup stage.

Key positions:

```text
start:
approximately 1.0 × base_lr

middle:
0.55 × base_lr

end:
0.1 × base_lr
```

With:

```text
base_lr = 5e-4
```

approximately:

```text
start  ≈ 5e-4
middle = 2.75e-4
end    = 5e-5
```

So the schedule is:

```text
high LR
↓
cosine decay
↓
10% of base LR
```

not:

```text
warmup
→ peak
→ cosine decay
```

---

# 19. LR Advances by Micro Step

An important source-code detail:

```python
get_lr(...)
```

is called for every DataLoader micro step.

But:

```python
optimizer.step()
```

only occurs after gradient accumulation completes.

For:

```text
accumulation_steps = 8
```

the flow is:

```text
step 1 → calculate lr1 → no optimizer update
step 2 → calculate lr2 → no optimizer update
...
step 7 → calculate lr7 → no optimizer update
step 8 → calculate lr8 → optimizer.step()
```

So the actual parameter update uses the optimizer's LR value at step 8.

Therefore the LR schedule advances by:

```text
micro steps
```

rather than:

```text
optimizer updates
```

in this implementation.

---

# 20. AdamW

MiniMind uses:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)
```

AdamW does not depend only on the current gradient.

It also maintains optimizer history for each parameter, including concepts such as:

```text
m
→ first moment / momentum-like state

v
→ second moment / squared-gradient state

optimizer step
→ update count
```

Therefore these optimizer states matter when resuming training.

---

# 21. Checkpoint

MiniMind saves model weights periodically when:

```text
step % save_interval == 0
```

or at the final step of the epoch.

Default:

```text
save_interval = 1000
```

So checkpoint saving is not performed after every training micro step.

There are two useful concepts:

```text
model weights
→ what the model parameters currently are

resume checkpoint
→ a snapshot of the training state
```

---

# 22. Resume Checkpoint

MiniMind calls:

```python
lm_checkpoint(
    lm_config,
    weight=args.save_weight,
    model=model,
    optimizer=optimizer,
    scaler=scaler,
    epoch=epoch,
    step=step,
    wandb=wandb,
    save_dir='../checkpoints'
)
```

The resume state includes training information such as:

```text
model
optimizer
scaler
epoch
step
world_size
wandb_id
```

The exact additional states depend on what is passed into `lm_checkpoint`.

---

# 23. Why model.state_dict() Is Not Enough for Resume

Loading only:

```python
model.load_state_dict(...)
```

restores model parameters but does not fully restore the previous training state.

For AdamW, the optimizer also has historical states such as:

```text
m
v
optimizer step
```

If they are lost, the model weights may be restored, but AdamW behaves like a newly initialized optimizer from that point.

Therefore MiniMind also loads:

```python
optimizer.load_state_dict(
    ckp_data['optimizer']
)
```

and:

```python
scaler.load_state_dict(
    ckp_data['scaler']
)
```

as well as:

```text
epoch
step
```

The goal of resume training is:

```text
restore the training state as completely as possible
```

rather than only restoring the model weights.

---

# 24. Resume Position

MiniMind restores:

```python
start_epoch = ckp_data['epoch']
start_step = ckp_data.get('step', 0)
```

It then uses:

```python
SkipBatchSampler(...)
```

to skip batches that have already been processed.

Conceptually:

```text
checkpoint:
epoch = 1
step = 500

resume:
start from that saved training position
instead of restarting the epoch from batch 1
```

---

# 25. Full Training Loop Summary

The complete MiniMind pretraining sequence can be explained as follows:

> Raw JSON text first enters `PretrainDataset`. The tokenizer converts one sample into a fixed-length `input_ids [T]`; `labels [T]` are copied from `input_ids`, while PAD positions are replaced with `-100`. `DataLoader` combines multiple samples into a batch, so the training loop receives `input_ids [B,T]` and `labels [B,T]`.
>
> `input_ids` enters MiniMind and first passes through the Embedding layer, becoming hidden states `[B,T,D]`. It then passes through multiple Transformer Blocks. Each Block contains RMSNorm, Attention, Residual, RMSNorm, SwiGLU FeedForward and another Residual connection, while the overall Block shape remains `[B,T,D]`. After all Blocks and the Final RMSNorm, the hidden states are still `[B,T,D]`. The LM Head projects the last dimension from `D` to vocabulary size `V`, producing logits `[B,T,V]`.
>
> The model performs next-token shifting internally: `logits[:, :-1, :]` is paired with `labels[:, 1:]`, and CrossEntropy produces a scalar language-model loss. Each micro-batch divides the loss by `accumulation_steps` and calls `backward()`, so gradients accumulate in `parameter.grad`.
>
> Once `step % accumulation_steps == 0`, MiniMind first unscales gradients when required, then calls `clip_grad_norm_()`, performs the optimizer update, updates the GradScaler state, and finally clears gradients with `zero_grad()`. Thus, `backward()` calculates and accumulates gradients, while `optimizer.step()` is the operation that actually modifies model parameters.
>
> The learning rate is calculated by `get_lr()` on every micro step using cosine decay and written directly into the optimizer. At checkpoint intervals or the end of the epoch, MiniMind saves model weights and a resume checkpoint. A proper resume restores not only model weights but also optimizer state, scaler state, epoch and step information, because AdamW depends on historical states such as `m`, `v`, and optimizer step.

---

## 25.1 中文完整训练流程总结

原始 JSON 数据首先经过 `PretrainDataset`。单个样本会被 Tokenizer 转成固定长度的 `input_ids [T]` 和 `labels [T]`，其中 labels 基本复制 input_ids，但 PAD 位置改为 `-100`。DataLoader 再把多个样本组成一个 batch，因此得到 `input_ids [B,T]` 和 `labels [B,T]`。

`input_ids` 输入 MiniMind 后，先经过 Embedding 得到 `[B,T,D]`，再经过多个 Transformer Block。每个 Block 内部依次进行 RMSNorm、Attention、Residual、RMSNorm、SwiGLU FeedForward、Residual，因此 Block 的输入输出都保持 `[B,T,D]`。经过所有 Block 和 Final RMSNorm 后仍为 `[B,T,D]`，再经过 LM Head 将 `D` 投影到 vocabulary size `V`，得到 logits `[B,T,V]`。

模型内部用 `logits[:, :-1, :]` 与 `labels[:, 1:]` 做 next-token prediction，经过 CrossEntropy 得到一个标量 loss。每个 micro batch 都会执行 `loss / accumulation_steps` 和 `backward()` 来累积梯度；当 `step % accumulation_steps == 0` 时，先 unscale，再进行 gradient clipping，然后执行 `optimizer.step()` 真正更新参数，最后 `zero_grad()` 清空梯度。

学习率由 `get_lr()` 按 micro step 计算并写入 optimizer。到达保存间隔或 epoch 末尾时，会保存模型权重以及用于恢复训练的 resume checkpoint，其中包含 model、optimizer、scaler、epoch、step 等训练状态。

# 26. Full Shape Flow

With:

```text
B = batch size
T = sequence length
D = hidden size
V = vocabulary size
```

the complete shape flow is:

```text
Dataset item

input_ids [T]
labels    [T]

↓

DataLoader

input_ids [B,T]
labels    [B,T]

↓

Embedding

hidden_states [B,T,D]

↓

Transformer Block × N

hidden_states [B,T,D]

↓

Final RMSNorm

hidden_states [B,T,D]

↓

LM Head

logits [B,T,V]

↓

Next-token shift

x [B,T-1,V]
y [B,T-1]

↓

Flatten

x [B×(T-1),V]
y [B×(T-1)]

↓

CrossEntropy

loss scalar
```

Default MiniMind pretrain example:

```text
input_ids
[32,340]

↓

Embedding

[32,340,768]

↓

8 Transformer Blocks

[32,340,768]

↓

Final RMSNorm

[32,340,768]

↓

LM Head

[32,340,6400]

↓

Shift

x [32,339,6400]
y [32,339]

↓

CrossEntropy

scalar loss
```

---

# 27. Key Default Training Arguments

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

Single-process effective batch size:

```text
32 × 8 = 256 sequences
```

---

# 28. Common Mistakes to Avoid

## Mistake 1

```text
input_ids [B,T,D]
```

Wrong.

Correct:

```text
input_ids [B,T]
```

Only after Embedding:

```text
hidden_states [B,T,D]
```

## Mistake 2

Thinking SwiGLU directly outputs logits.

Wrong.

SwiGLU / Transformer Block ends at:

```text
[B,T,D]
```

The LM Head creates:

```text
[B,T,V]
```

## Mistake 3

Thinking `backward()` updates model parameters.

Wrong.

```text
backward()
→ calculate / accumulate gradients

optimizer.step()
→ update model parameters
```

## Mistake 4

Thinking `step` in the DataLoader loop is the same as `optimizer.step()`.

Wrong.

With gradient accumulation:

```text
multiple micro steps
→ one optimizer update
```

## Mistake 5

Thinking MiniMind pretrain currently uses Warmup + Cosine.

In this source version, `get_lr()` implements cosine decay without an explicit warmup phase.

## Mistake 6

Thinking a model weight file alone is a full resume checkpoint.

A proper resume also needs optimizer and other training states.

---

# 29. Day 3 Core Memory

The shortest version to remember:

```text
JSON
↓
Dataset
[T]

↓ DataLoader

[B,T]

↓ Model

[B,T,V]

↓ shift

x [B,T-1,V]
y [B,T-1]

↓ CE

loss

↓ loss / accumulation_steps
↓ backward × N

accumulated gradients

↓ unscale
↓ clip
↓ optimizer.step
↓ zero_grad

parameter update

↓ checkpoint

save model + training state
```
