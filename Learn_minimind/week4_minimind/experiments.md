# Week 4 MiniMind Experiments

## pretrain_smoke_001

### Purpose

验证 MiniMind Pretrain 完整训练链路是否可以正常运行：

```text
Dataset
→ DataLoader
→ Model
→ Loss
→ Backward
→ Gradient Accumulation
→ AdamW
→ Learning Rate
→ Checkpoint
→ Resume Checkpoint
```

### Environment

```text
experiment_id: pretrain_smoke_001
git_commit: 7a9137d
device: Apple MPS
world_size: 1
```

### Dataset

```text
data_version: pretrain_smoke_v1
dataset: pretrain_smoke.jsonl
samples: 128
```

数据由 8 条测试文本重复 16 次组成，仅用于 Smoke Test，不用于评价模型能力。

### Model

```text
vocab_size: 6400
hidden_size: 128
num_hidden_layers: 2
num_attention_heads: 8
num_key_value_heads: 4
model_params: 1.26M
use_moe: false
```

### Training Config

```text
epochs: 1
max_seq_len: 64

micro_batch_size: 2
gradient_accumulation_steps: 2
world_size: 1
effective_batch_size: 4

micro_steps: 64
optimizer_updates: 32

optimizer: AdamW
learning_rate: 5e-4
lr_schedule: cosine_decay
warmup_steps: 0

grad_clip: 1.0
precision: bfloat16
num_workers: 0
seed: 42
log_interval: 1
save_interval: 1000
```

### Run Command

```bash
cd minimind/trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_smoke.jsonl \
  --epochs 1 \
  --batch_size 2 \
  --accumulation_steps 2 \
  --hidden_size 128 \
  --num_hidden_layers 2 \
  --max_seq_len 64 \
  --learning_rate 5e-4 \
  --device mps \
  --num_workers 0 \
  --log_interval 1 \
  --save_interval 1000 \
  --save_weight smoke_pretrain
```

### Result

```text
Model Params: 1.26M
Trainable Params: 1.262M
```

关键训练日志：

```text
step 1:
loss = 8.7192
lr   = 0.00049973

step 16:
loss = 7.3243
lr   = 0.00043410

step 32:
loss = 6.5182
lr   = 0.00027500

step 48:
loss = 6.5735
lr   = 0.00011590

step 64:
loss = 6.3709
lr   = 0.00005000
```

Loss 并不是每个 step 单调下降，但整体从约 8.7 下降到约 6.3，说明训练链路工作正常。

Learning Rate 从接近 `5e-4` 按 cosine decay 下降到 `5e-5`，与当前 `get_lr()` 实现一致。

由于 `use_moe = false`，所以 `aux_loss = 0.0000`；本次主要训练损失来自 Causal LM 的 CrossEntropy Loss。

### Gradient Accumulation

本次配置：

```text
micro_batch_size = 2
gradient_accumulation_steps = 2
world_size = 1
```

所以：

```text
effective_batch_size
= 2 × 2 × 1
= 4 sequences / optimizer update
```

共有：

```text
128 samples / batch_size 2
= 64 micro steps
```

因此约有：

```text
64 / 2
= 32 optimizer updates
```

核心顺序：

```text
forward
↓
loss / accumulation_steps
↓
backward
↓
累计 2 个 micro batch
↓
unscale
↓
clip_grad_norm_
↓
optimizer.step()
↓
zero_grad()
```

### Checkpoint

生成文件：

```text
../out/smoke_pretrain_128.pth
../checkpoints/smoke_pretrain_128.pth
../checkpoints/smoke_pretrain_128_resume.pth
```

文件大小约：

```text
model weight: 4.17 MB
resume checkpoint: 14.29 MB
```

Resume Checkpoint 内容：

```text
keys:
model
optimizer
epoch
step
world_size
wandb_id
scaler
```

实际状态：

```text
epoch: 0
step: 64
world_size: 1
```

其中 `epoch = 0` 代表第一个 epoch，因为源码从 0 开始计数。

Optimizer State：

```text
state
param_groups
```

其中 `state` 保存 AdamW 的历史优化状态，例如一阶动量、二阶动量和 optimizer step；`param_groups` 保存 learning rate、weight decay 等优化器配置。

### Conclusion

```text
Dataset               ✅
DataLoader             ✅
Forward                ✅
CrossEntropy Loss      ✅
Backward               ✅
Gradient Accumulation  ✅
Gradient Clipping      ✅
AdamW                  ✅
Cosine LR              ✅
Checkpoint             ✅
Resume State           ✅
```

Smoke Test 成功。

### Notes

- 本实验数据极小且高度重复，只用于验证代码链路，不用于评价模型能力。
- 当前日志没有直接打印 Gradient Norm，因此只能确认源码执行了 Gradient Clipping，不能从日志直接观察具体 `grad_norm`。
- 当前 MPS 路径主要验证 forward / backward / optimizer；CUDA mixed precision 需要后续在 RTX 3060 环境继续验证。
- 当前 pretrain 脚本没有独立 scheduler 对象，Learning Rate 由 `get_lr()` 每个 micro step 计算并写入 optimizer。
- 当前 `get_lr()` 没有显式 Warmup。
- 正式 `pretrain_baseline_001` 后续计划在 RTX 3060 环境运行。

---

## pretrain_baseline_001

### Status

```text
status: planned
```

后续将在 RTX 3060 环境中确定正式 Baseline 配置并运行。

计划记录：

```text
experiment_id
git_commit
data_version
config_path
run_command

vocab_size
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
max_seq_len

learning_rate
micro_batch_size
gradient_accumulation_steps
world_size

optimizer
weight_decay
precision

train_loss
learning_rate_curve
gradient_norm
tokens_per_second
training_time
peak_gpu_memory

checkpoint
resume_checkpoint
notes
```
