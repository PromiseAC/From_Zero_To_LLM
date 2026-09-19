# MiniMind Pretrain Baseline

## Goal

本项目用于完成第四周 MiniMind Pretrain Baseline 的最终整理与验收。

目标不是“会运行 MiniMind”，而是能够把前三周学习过的 Transformer、Causal LM、Training Recipe 映射到真实 MiniMind 源码，并完成一套可复现的：

```text
MiniMind Repo
↓
Tokenizer
↓
Dataset
↓
Pretrain
↓
Train Loss / Validation Loss
↓
Checkpoint
↓
Resume Training
↓
Fixed Prompt
↓
Different Checkpoints
↓
Generation Comparison
↓
Experiment Record
↓
README
```

最终要求：

- 能解释 MiniMind 整体模型结构；
- 能解释完整的 Causal LM Training Pipeline；
- 能运行可复现的 Pretrain Baseline；
- 能保存、加载和恢复 Checkpoint；
- 能使用固定 Prompt 比较不同训练阶段的 Checkpoint；
- 能记录 Git Commit、Config、Data Version、Loss 和 Generation Sample；
- 能解释为什么 Loss 下降不等于 Generation Quality 一定提升。

---

## Environment

```text
OS: Ubuntu 24.04 under WSL2
Python: 3.12.3
PyTorch: 2.14.0+cu130
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
GPU Memory: 6 GB
Device: CUDA
```

Python 虚拟环境：

```bash
source /home/promisea/Environments/llm/bin/activate
```

仓库：

```text
/home/promisea/Projects/From_Zero_To_LLM
```

MiniMind 工作目录：

```text
Learn_minimind/minimind
```

MiniMind 上游源码基线：

```text
7a9137d
```

---

## Repository Structure

```text
From_Zero_To_LLM/
├── Learn_minimind/
│   ├── minimind/
│   │   ├── model/
│   │   ├── trainer/
│   │   ├── dataset/
│   │   ├── configs/
│   │   ├── checkpoints/
│   │   ├── eval_llm.py
│   │   └── eval_fixed_prompt.py
│   │
│   ├── week4_minimind/
│   │   ├── README.md
│   │   ├── generation_results.md
│   │   ├── notes/
│   │   └── experiments/
│   │
│   └── workflow/
│       └── llm_algorithm_development_workflow.md
│
├── Learn_llm_training/
├── Learn_pytorch/
├── .gitignore
└── README.md
```

Dataset、Checkpoint、日志等训练产物保留在本地，不进入 Git。

---

## Model Architecture

MiniMind 使用 Decoder-only Transformer。

```text
Token IDs
[B,T]

↓

Embedding
[B,T,D]

↓

Transformer Block × N
│
├── RMSNorm
├── GQA Attention
│   ├── Q Projection
│   ├── K Projection
│   ├── V Projection
│   ├── RoPE on Q/K
│   └── Attention Output
├── Residual
├── RMSNorm
├── SwiGLU
└── Residual

↓

Final RMSNorm
[B,T,D]

↓

LM Head
Linear(D,V)

↓

Logits
[B,T,V]
```

主要模型配置：

| Parameter | Value |
| --- | ---: |
| vocab_size | 6400 |
| hidden_size / d_model | 768 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 4 |
| head_dim | 96 |
| intermediate_size | 2432 |
| normalization | RMSNorm |
| positional encoding | RoPE |
| attention | GQA |
| FFN | SwiGLU |
| parameters | ~63.91M |

### Shape Flow

以 `B=8, T=768, D=768, V=6400` 为例：

```text
input_ids
[8,768]

↓ Embedding

[8,768,768]

↓ Transformer Block × 8

[8,768,768]

↓ Final RMSNorm

[8,768,768]

↓ LM Head

[8,768,6400]
```

Transformer Block 内部保持 `[B,T,D]`，只有 LM Head 将 `D → V`。

### Attention / GQA

```text
Q: [B,T,8,96]
K: [B,T,4,96]
V: [B,T,4,96]
```

K/V 通过 `repeat_kv` 扩展：

```text
[B,T,4,96]
↓
[B,T,8,96]
```

GQA 让多个 Query Head 共享较少数量的 K/V Head，从而降低 K/V 的存储和推理开销。

### RoPE

RoPE 主要作用在 Q/K 上，通过位置相关旋转将位置信息编码进 Attention。

```text
Q / K
+
Position-dependent Rotation
↓
Position-aware Q / K
↓
Attention Score
```

RoPE 不改变 Tensor Shape，V 一般不使用 RoPE。

### RMSNorm

```text
[B,T,D]
↓
RMSNorm
[B,T,D]
```

RMSNorm 用于稳定 Hidden State 的数值尺度。MiniMind 使用 Pre-Norm，并在最终 LM Head 前再使用一次 RMSNorm。

### SwiGLU

```text
x
├── gate_proj → SiLU
└── up_proj
        ↓
Element-wise Multiply
        ↓
down_proj
        ↓
[B,T,D]
```

对应：

```python
down_proj(
    silu(gate_proj(x)) * up_proj(x)
)
```

中间维度会扩张，最后通过 `down_proj` 回到 `D`，保证 Residual 可以相加。

---

## Dataset

预训练使用 `PretrainDataset`。

原始格式：

```json
{"text": "..."}
```

处理流程：

```text
Raw Text
↓
Tokenizer(add_special_tokens=False)
↓
truncate to max_length - 2
↓
手动添加 BOS
↓
手动添加 EOS
↓
PAD 到 max_length
↓
input_ids [T]
↓
labels = input_ids.clone()
↓
PAD 对应 label → -100
```

Label：

```python
labels = input_ids.clone()
labels[input_ids == tokenizer.pad_token_id] = -100
```

`-100` 会被 CrossEntropyLoss 的 `ignore_index=-100` 忽略，因此 Padding 不参与 Loss。

---

## Training Pipeline

```text
Raw Text
↓
Tokenizer
↓
Token IDs
↓
Dataset
↓
DataLoader
↓
input_ids / labels
[B,T]
↓
MiniMind
↓
logits
[B,T,V]
↓
Causal LM Loss
↓
Backward
↓
Gradient Accumulation
↓
Gradient Clipping
↓
AdamW
↓
Cosine LR Schedule
↓
Checkpoint
```

### Causal LM Loss

Dataset 不手动 Shift Labels。模型内部执行：

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
```

因此：

```text
x0 → predict x1
x1 → predict x2
x2 → predict x3
...
```

Loss：

```python
F.cross_entropy(
    x.view(-1, x.size(-1)),
    y.view(-1),
    ignore_index=-100
)
```

### Gradient Accumulation

固定评测实验：

```text
micro_batch_size = 8
gradient_accumulation_steps = 4
world_size = 1
```

有效 Batch Size：

```text
8 × 4 × 1 = 32
```

因此：

```text
100 micro steps
÷ 4
=
25 optimizer updates
```

### Gradient Clipping

```text
Backward
↓
Unscale Gradients
↓
clip_grad_norm_
↓
optimizer.step()
```

阈值：

```text
grad_clip = 1.0
```

### AdamW

```text
optimizer = AdamW
learning_rate = 5e-4
weight_decay = 0.01
```

### Scheduler

当前 MiniMind Pretrain Script 没有创建独立的 `torch.optim.lr_scheduler` 对象，而是通过 `get_lr(...)` 根据训练进度计算学习率，并写回 Optimizer parameter group。

本周使用 Cosine Decay，无 Warmup。

---

## Configuration

| Parameter | Value |
| --- | ---: |
| vocab_size | 6400 |
| hidden_size | 768 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 4 |
| max_seq_len | 768 |
| micro_batch_size | 8 |
| gradient_accumulation_steps | 4 |
| world_size | 1 |
| effective_batch_size | 32 |
| learning_rate | 5e-4 |
| weight_decay | 0.01 |
| gradient_clip | 1.0 |
| optimizer | AdamW |
| precision | BF16 |
| seed | 42 |

相关配置：

```text
configs/pretrain_smoke.yaml
configs/pretrain_baseline.yaml
configs/fixed_prompt_eval.yaml
```

---

## Run Command

激活环境：

```bash
source /home/promisea/Environments/llm/bin/activate
```

固定 Prompt Evaluation 对应训练命令：

```bash
cd /home/promisea/Projects/From_Zero_To_LLM/Learn_minimind/minimind/trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_fixed_train_8k.jsonl \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --learning_rate 5e-4 \
  --device cuda \
  --dtype bfloat16 \
  --num_workers 8 \
  --seed 42 \
  --log_interval 50 \
  --save_interval 1000 \
  --save_weight pretrain_fixed_eval \
  --snapshot_steps 100,500,1000 \
  --snapshot_dir ../checkpoints/fixed_prompt_eval
```

训练代码 Commit：

```text
79e46399ca7cf4bff3ce321010ee1b6d0377f8b7
```

---

## Experiment Results

### Experiment 1: Pretrain Baseline

| Field | Value |
| --- | --- |
| experiment_id | pretrain_learning_001 |
| model | MiniMind ~63.91M |
| dataset | pretrain_learning_10k.jsonl |
| data_version | local 10k learning subset |
| dataset_size | 10,000 |
| vocab_size | 6400 |
| d_model | 768 |
| num_layers | 8 |
| num_heads | 8 |
| num_kv_heads | 4 |
| max_seq_len | 768 |
| learning_rate | 5e-4 |
| micro_batch_size | 8 |
| gradient_accumulation_steps | 4 |
| effective_batch_size | 32 |
| world_size | 1 |
| optimizer | AdamW |
| precision | BF16 |
| training_steps | 1250 micro steps |
| optimizer_steps | ~313 |
| final observed train loss | ~5.19 |
| checkpoint | pretrain_learning_001_768.pth |
| validation_loss | N/A |
| training_time | not formally recorded |
| exact git_commit | not formally recorded |
| peak_memory | not formally recorded |
| tokens/s | not formally recorded |

---

## Checkpoint / Resume

### Model-only Checkpoint

主要包含：

```text
Model Weights
```

适合：

```text
Inference
Evaluation
Generation
Weight Loading
```

例如：

```text
pretrain_learning_001_768.pth
```

### Resume Checkpoint

用于恢复训练，典型内容：

```text
Model State
Optimizer State
Epoch
Micro Step
World Size
W&B Run ID
GradScaler State
```

已检查的 Resume Metadata：

```text
pretrain_learning_001_768_resume.pth

epoch = 0
micro step = 1250
optimizer step = 312
world_size = 1
```

```text
pretrain_resume_001_768_resume.pth

epoch = 0
micro step = 400
optimizer step = 100
world_size = 1
```

AdamW 内部包含 Momentum、Second Moment、Step Counter，因此真正的 Resume 不能只恢复模型权重，还要恢复 Optimizer State 和训练进度。

---

## Fixed Prompt Evaluation

### Dataset Split

```text
Original = 10,000 samples
seed = 42
Train = 8,000
Validation = 2,000
Overlap = 0
```

由于：

```text
8000 / batch_size 8 = 1000 micro steps
```

选择：

| Stage | Micro Step | Optimizer Step | Progress |
| --- | ---: | ---: | ---: |
| Early | 100 | 25 | 10% |
| Middle | 500 | 125 | 50% |
| Late | 1000 | 250 | 100% |

三个 Checkpoint 来自同一训练轨迹，因此主要变量只有 Training Progress。

### Checkpoints

```text
pretrain_fixed_eval_step0100_768.pth
pretrain_fixed_eval_step0500_768.pth
pretrain_fixed_eval_step1000_768.pth
```

SHA-256：

```text
Early:
e4cfbf9c7ed6f436b55837296bae92c08ed1c0cf1aee7603aef27a51e46f0597

Middle:
410f2b5bd3b2c26ad991c51991dc3bea4eb97986b990e7240b20265a2fda0c98

Late:
da316797e7c41df47e61789fd44bb26eeea91088777d8481c296f05a0e6a3ede
```

Evaluation Code Commit：

```text
518e9a563a073ee95374ca047242ccbf55bb1334
```

### Fixed Prompts

```text
中国的首都是
机器学习是
1 + 1 =
```

Generation Config：

```text
decoding = greedy
do_sample = False
max_new_tokens = 64
temperature = 1.0
top_p = 1.0
top_k = 0
repetition_penalty = 1.0
```

### Loss Comparison

| Checkpoint | Micro Step | Optimizer Step | Fixed Train Eval Loss | Validation Loss |
| --- | ---: | ---: | ---: | ---: |
| Early | 100 | 25 | 7.2222 | 7.2279 |
| Middle | 500 | 125 | 6.0104 | 6.0425 |
| Late | 1000 | 250 | 5.4029 | 5.4747 |

---

## Generation Samples

### Prompt: `中国的首都是`

Early：

```text
，
```

Middle：

```text
我，我在我是我是我，我是我是我，我是我是我，我是我是我。我，我是我，我是我，我是我，我是我，我是我，我是我，我是我，我是我，我是
```

Late：

```text
夏天的诗歌。秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
...
```

### Prompt: `机器学习是`

Early：

```text
，
```

Middle：

```text
我你的。我我在我你是我你是我，我在我在我，我你是我是我，我是我是我，我是我是我...
```

Late：

```text
““““““““““““““““““““““““““““““““...
```

### Prompt: `1 + 1 =`

Early：

```text
，
```

Middle：

```text
主要为空格 / 换行
```

Late：

```text
3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = ...
```

---

## Observations

### 1. Loss 持续下降

```text
Train Eval Loss:
7.2222 → 6.0104 → 5.4029

Validation Loss:
7.2279 → 6.0425 → 5.4747
```

说明模型在 Token-level Next-token Prediction Objective 上持续学习。

### 2. Generation Quality 并没有同步提升

Early 几乎只输出标点；Middle 开始学习常见中文 Token 和局部结构，但重复严重；Late 的文本形式更像自然语言，但仍然存在语义错误、事实错误和 Repetition Loop。

因此：

```text
Lower CrossEntropy Loss
≠
Better Factual Correctness
≠
Better Reasoning
≠
Better Fluency
≠
Better Sequence-level Generation
```

### 3. Token-level Objective 与 Generation Quality 是不同维度

Pretraining 优化：

```text
P(next token | previous tokens)
```

平均 CrossEntropy Loss 并不会直接衡量：

```text
事实正确率
推理能力
Instruction Following
重复问题
长文本连贯性
人类主观质量
```

### 4. 当前实验规模很小

```text
8000 training samples
250 optimizer updates
~63.91M parameters
```

因此生成质量较差是正常的。实验目标是理解 Loss 与 Generation Evaluation 的区别，而不是训练出高质量 Chat Model。

### 5. 当前是 Pretrain Model

模型尚未经过：

```text
SFT
Preference Optimization
DPO
RL
```

因此这些 Prompt 应主要被理解为 Text Continuation Evaluation，而不是 Instruction-following Evaluation。

---

## Problems Encountered

### 1. Checkpoint 被覆盖

原始文件名不带 Step，后续保存会覆盖之前 Checkpoint。

解决：

```text
--snapshot_steps 100,500,1000
```

保存为独立 Step Snapshot。

### 2. 旧 Checkpoint 不能直接作为 Early / Middle / Late

`memprobe_b1/b2/b4/b8` 改变了 Batch Size 和实验条件，不属于同一训练轨迹。

正确方法：

```text
One Training Trajectory
↓
Step 100
Step 500
Step 1000
```

### 3. 原始 Training Script 没有 Validation Loop

使用固定：

```text
8k Train
2k Validation
seed = 42
```

三个 Checkpoint 都在同一 Validation Set 上计算 Loss。

### 4. 单个 Batch Loss 很噪声

例如：

```text
step800  = 5.6154
step850  = 5.7643
step900  = 5.5612
step950  = 5.5982
step1000 = 5.4800
```

因此正式比较使用 Fixed Train Eval Loss，而不是某一个随机 Batch 的 Loss。

### 5. Resume Checkpoint Save Timing

10k Baseline：

```text
1250 micro steps
GA = 4
```

完整 Accumulation Group：

```text
1248 / 4 = 312
```

还剩 2 个 Micro Batch，需要一次 leftover optimizer update。

Resume Checkpoint 在 leftover update 前保存，因此记录的 Optimizer Step 为 312。

这说明 Checkpoint 的语义取决于它在 Training Loop 中的保存位置。

---

## Experiment Table

### Pretrain Baseline

| Field | Value |
| --- | --- |
| experiment_id | pretrain_learning_001 |
| model | MiniMind ~63.91M |
| dataset | pretrain_learning_10k.jsonl |
| data_version | local 10k learning subset |
| dataset_size | 10,000 |
| seed | not formally recorded |
| git_commit | not formally recorded |
| config_path | configs/pretrain_baseline.yaml |
| vocab_size | 6400 |
| d_model | 768 |
| num_layers | 8 |
| num_heads | 8 |
| max_seq_len | 768 |
| learning_rate | 5e-4 |
| micro_batch_size | 8 |
| gradient_accumulation_steps | 4 |
| world_size | 1 |
| training_steps | 1250 micro steps |
| warmup_steps | 0 |
| optimizer | AdamW |
| weight_decay | 0.01 |
| precision | BF16 |
| training_time | not formally recorded |
| peak_memory | not formally recorded |
| tokens/s | not formally recorded |
| train_loss | final observed ~5.19 |
| validation_loss | N/A |
| checkpoint | pretrain_learning_001_768.pth |
| generation_samples | not formal metric |
| notes | first complete MiniMind pretraining baseline |

### Fixed Prompt Evaluation

| Field | Value |
| --- | --- |
| experiment_id | fixed_prompt_eval_001 |
| model | MiniMind ~63.91M |
| dataset | deterministic split of pretrain_learning_10k.jsonl |
| data_version | seed-42 8k/2k split |
| dataset_size | 8,000 train / 2,000 validation |
| seed | 42 |
| training_git_commit | 79e46399ca7cf4bff3ce321010ee1b6d0377f8b7 |
| evaluation_git_commit | 518e9a563a073ee95374ca047242ccbf55bb1334 |
| config_path | configs/fixed_prompt_eval.yaml |
| vocab_size | 6400 |
| d_model | 768 |
| num_layers | 8 |
| num_heads | 8 |
| max_seq_len | 768 |
| learning_rate | 5e-4 |
| micro_batch_size | 8 |
| gradient_accumulation_steps | 4 |
| world_size | 1 |
| training_steps | 1000 micro steps |
| optimizer_steps | 250 |
| warmup_steps | 0 |
| optimizer | AdamW |
| weight_decay | 0.01 |
| precision | BF16 |
| training_time | exact value not formally recorded |
| peak_memory | exact value not formally recorded |
| tokens/s | exact value not formally recorded |
| Early train loss | 7.2222 |
| Early validation loss | 7.2279 |
| Middle train loss | 6.0104 |
| Middle validation loss | 6.0425 |
| Late train loss | 5.4029 |
| Late validation loss | 5.4747 |
| checkpoints | step100 / step500 / step1000 |
| generation_samples | 3 fixed prompts / greedy decoding |
| notes | demonstrates loss-quality mismatch |

---

## Reproducibility Checklist

正式实验至少记录：

```text
experiment_id
model
dataset
data_version
dataset_size
seed

git_commit
config_path
run_command

vocab_size
d_model
num_layers
num_heads
max_seq_len

learning_rate
micro_batch_size
gradient_accumulation_steps
world_size

training_steps
warmup_steps
optimizer
weight_decay
precision

training_time
peak_memory
tokens/s

train_loss
validation_loss

checkpoint
generation_samples

notes
```

正式实验开始前记录：

```bash
git status
git rev-parse HEAD
```

最终保证：

```text
Code
+
Config
+
Dataset Version
+
Checkpoint
+
Result
```

能够相互对应。

---

## Week 4 Final Acceptance

### Model

- [x] 能解释 MiniMind 整体结构
- [x] 能解释 Attention / GQA
- [x] 能解释 RoPE
- [x] 能解释 RMSNorm
- [x] 能解释 SwiGLU
- [x] 能解释 `[B,T] → [B,T,V]`

### Training

- [x] 能解释 Dataset / DataLoader
- [x] 能解释 Causal LM Loss
- [x] 能解释 Gradient Accumulation
- [x] 能解释 Gradient Clipping
- [x] 能解释 AdamW
- [x] 能解释 LR Schedule
- [x] 能解释完整 Training Step

### Checkpoint

- [x] 能保存 Checkpoint
- [x] 能 Load Checkpoint
- [x] 能 Resume Training
- [x] 能解释 Model State
- [x] 能解释 Optimizer State
- [x] 能解释 LR / Scheduler Progress
- [x] 能解释 Resume Metadata

### Experiment

- [x] Baseline 可以重新运行
- [x] 有 Config
- [x] 有 Run Command
- [x] 有 Git Commit
- [x] 有 Data Version
- [x] 有 Train Eval Loss
- [x] 有 Validation Loss
- [x] 有 Fixed Prompt
- [x] 有多个 Checkpoint Generation Samples

### Project Materials

- [x] 完成模型结构图
- [x] 完成 Training Data Flow
- [x] 完成 Baseline Experiment Table
- [x] 完成第一版 README
- [x] 能解释为什么 Loss 下降不等于 Generation Quality 一定提升

---

## Next Step

第四周完成：

```text
MiniMind Pretrain Baseline
+
Checkpoint / Resume
+
Fixed Prompt Evaluation
+
Experiment Record
```

下一阶段进入 Post-training：

```text
Pretrain
↓
SFT
↓
LoRA
↓
Data Governance / Evaluation
↓
Preference Data
↓
DPO
↓
RL / GRPO
```

后续重点从：

```text
“能跑训练代码”
```

逐渐转向：

```text
“能设计、复现、评测、比较并解释 LLM Algorithm Experiment”
```
