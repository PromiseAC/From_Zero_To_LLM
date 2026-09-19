# Week 4 周五：Checkpoint 与 Resume Training

## 一、今日目标

今天重点掌握：

```text
Checkpoint
Resume Training
Model State
Optimizer State
Scheduler State
Training Step
```

目标不是只会“保存一个 `.pth` 文件”，而是理解：

> 真正的 Resume Training，需要尽可能恢复训练现场，而不是只加载模型参数。

---

# 二、普通 Model Weight 与 Resume Checkpoint

## 1. Model Weight

普通模型权重主要保存：

```python
model.state_dict()
```

也就是：

```text
Embedding 参数
Attention 参数
FFN 参数
RMSNorm 参数
LM Head 参数
...
```

它回答的问题是：

> 模型当前学成了什么样？

只加载 Model Weight，可以恢复模型参数，但不能完整恢复训练现场。

## 2. Resume Checkpoint

Resume Checkpoint 更像：

```text
训练现场快照
```

当前 MiniMind 的 Resume Checkpoint 中实际包含：

```text
model
optimizer
epoch
step
world_size
wandb_id
scaler
```

所以：

```text
Model Weight
≠
完整 Resume Checkpoint
```

---

# 三、Checkpoint 中各个 State 的意义

## 1. Model State

Model State 保存模型参数，例如：

```text
model.embed_tokens.weight
```

本次检查：

```text
name:
model.embed_tokens.weight

shape:
(6400, 768)

first 5 values:
[ 0.0353, -0.0399, -0.0119, 0.0422, -0.0468 ]
```

也就是说 Embedding 参数矩阵大小是：

```text
[vocab_size, hidden_size]
=
[6400, 768]
```

## 2. Optimizer State

当前 MiniMind 使用：

```text
AdamW
```

AdamW 不只是使用“当前梯度”，还维护历史状态：

```text
exp_avg
→ 一阶指数移动平均
→ 近似理解为 m_t

exp_avg_sq
→ 梯度平方的二阶指数移动平均
→ 近似理解为 v_t

step
→ optimizer 已经更新了多少次
```

所以如果训练中断后只恢复模型：

```text
Model State 恢复
Optimizer State 清零
```

那么模型虽然站在原来的参数位置，但 AdamW 已经“失忆”。

这不属于完整 Resume。

## 3. Scheduler State

在常见训练框架里，可能存在：

```python
scheduler = ...
```

Scheduler 可能维护：

```text
last_epoch
current_step
当前 LR schedule 位置
```

完整 Resume 通常需要：

```python
scheduler.load_state_dict(...)
```

否则可能出现：

```text
训练已经进行到中途
↓
Resume
↓
Scheduler 却重新从起点开始
↓
Learning Rate 跳变
```

## 4. 当前 MiniMind 的特殊情况

当前 MiniMind Pretrain 并没有独立的：

```python
scheduler = ...
```

而是在每个 micro step 中直接：

```python
get_lr(
    current_step,
    total_steps,
    learning_rate
)
```

计算当前 LR。

因此当前 MiniMind 没有单独的：

```text
scheduler.state_dict()
```

Learning Rate 是否连续，关键依赖：

```text
epoch
step
total_steps
training config
```

所以 Resume 前后的训练配置必须保持一致。

---

# 四、第一次检查 Resume Checkpoint

昨天 Learning Baseline 的 Resume Checkpoint：

```text
../checkpoints/pretrain_learning_001_768_resume.pth
```

实际读取：

```text
Checkpoint keys:
- model
- optimizer
- epoch
- step
- world_size
- wandb_id
- scaler

epoch: 0
step: 1250
world_size: 1
```

其中：

```text
epoch = 0
```

并不是“没有训练”，而是 Python 中第一个 epoch 的索引就是 0。

---

# 五、Optimizer State 实际检查

本次读取：

```text
state entries: 90
param groups: 1
```

注意：

```text
90
```

不是“90 个标量参数”，而是 AdamW 为约 90 个参数张量维护 optimizer state。

一个参数张量本身可能包含大量标量。

例如：

```text
model.embed_tokens.weight
shape = (6400, 768)
```

就包含：

```text
6400 × 768
```

个标量参数。

Optimizer Param Group 实际结果：

```text
lr: 5e-05
betas: (0.9, 0.999)
eps: 1e-08
weight_decay: 0.01
```

---

# 六、Model Parameter 恢复验证

为了证明：

```python
model.load_state_dict(...)
```

真的恢复了参数，本次选：

```text
model.embed_tokens.weight
```

进行对比。

流程：

```text
Checkpoint 中读取参数
↓
创建一个新模型
↓
load_state_dict()
↓
重新读取同名参数
↓
比较
```

实际结果：

```text
checkpoint first 5:
tensor([ 0.0353, -0.0399, -0.0119, 0.0422, -0.0468])

loaded first 5:
tensor([ 0.0353, -0.0399, -0.0119, 0.0422, -0.0468])

max abs diff:
0.0

exactly restored:
True
```

因此可以确认：

```text
Checkpoint Parameter
↓
load_state_dict()
↓
Loaded Parameter

完全一致
```

Model State 恢复验证：

```text
✅ 通过
```

---

# 七、为什么昨天的 Epoch-End Checkpoint 不适合直接验证 LR Resume

昨天 Learning Baseline：

```text
epochs = 1
total_steps = 1250
```

训练结束时：

```text
step = 1250
lr = 5e-5
```

这已经是整个 cosine schedule 的终点。

如果 Resume 时突然改成：

```text
epochs = 2
```

那么：

```text
total_steps
```

会从：

```text
1250
```

变成：

```text
2500
```

此时：

```text
step = 1250
```

就会从“训练终点”变成“训练中点”。

所以正确的 Resume 实验必须：

> 一开始就固定完整训练计划，然后在中途保存并中断。

---

# 八、真正的 Resume Training 实验设计

实验名称：

```text
pretrain_resume_001
```

配置：

```text
dataset:
pretrain_learning_10k.jsonl

epochs:
2

batch_size:
8

gradient_accumulation_steps:
4

hidden_size:
768

num_hidden_layers:
8

max_seq_len:
768

learning_rate:
5e-4

save_interval:
200

log_interval:
10

device:
cuda
```

单个 epoch：

```text
10000 / 8 = 1250 micro steps
```

总训练计划：

```text
2 × 1250
=
2500 micro steps
```

实验流程：

```text
Train
↓
step 200
↓
Save Checkpoint
↓
Stop Program

重新启动

↓

Load Checkpoint
↓
Restore Model
↓
Restore Optimizer
↓
Restore epoch / step
↓
从 step 201 继续
```

---

# 九、Checkpoint 保存中断事故

第一次到：

```text
step 200
```

后立即按了：

```text
Ctrl + C
```

结果 traceback 显示程序正在：

```text
lm_checkpoint(...)
↓
torch.save(...)
```

时被中断。

磁盘上只留下：

```text
pretrain_resume_001_768.pth
132 MB
```

以及：

```text
pretrain_resume_001_768_resume.pth.tmp
45 MB
```

而正式：

```text
pretrain_resume_001_768_resume.pth
```

并不存在。

这说明：

```text
打印 step 200
≠
Checkpoint 已经完全写完
```

`.tmp` 可以理解为：

```text
先写临时文件
↓
全部写成功
↓
再成为正式 checkpoint
```

因此残缺 `.tmp` 不能直接改名使用。

---

# 十、正确保存 step 200 Checkpoint

重新训练后，不在 step 200 日志出现时立即停止，而是等到 step 210 之后再中断。

最终文件：

```text
../checkpoints/pretrain_resume_001_768.pth
132 MB

../checkpoints/pretrain_resume_001_768_resume.pth
619 MB
```

Resume Checkpoint 检查：

```text
epoch: 0
step: 200
lr: 0.000492931211253942
```

Checkpoint 完整保存：

```text
✅
```

---

# 十一、真正 Resume

第二次启动：

```text
--from_resume 1
```

程序输出：

```text
Epoch [1/2]:
跳过前200个step，从step 201开始
```

这直接证明：

```text
Checkpoint:
step = 200

Resume:
从 step 201 继续
```

而不是重新从 step 0 / step 1 开始。

---

# 十二、为什么第一条日志是 step 210

当前：

```text
log_interval = 10
```

所以：

```text
201
202
203
...
209
```

虽然正常执行，但不会打印。

第一条打印日志：

```text
210
```

不代表从 step 210 才开始训练。

真正恢复位置仍然是：

```text
step 201
```

---

# 十三、Learning Rate 连续性验证

Checkpoint 保存时：

```text
step 200
lr = 0.0004929312
```

Resume 后：

```text
step 210 → 0.00049221
step 220 → 0.00049146
step 230 → 0.00049067
step 240 → 0.00048984
step 250 → 0.00048899
step 260 → 0.00048810
step 270 → 0.00048717
step 280 → 0.00048622
step 290 → 0.00048522
step 300 → 0.00048420
```

形成连续下降：

```text
step 200
0.00049293
↓
step 210
0.00049221
↓
step 220
0.00049146
↓
...
```

没有出现：

```text
Resume
↓
LR 跳回 5e-4
```

因此：

```text
Learning Rate 连续
✅
```

---

# 十四、Loss 连续性观察

保存前：

```text
step 200
loss = 6.6552
```

Resume 后：

```text
210 → 6.8314
220 → 6.7716
230 → 6.6538
240 → 6.8395
250 → 6.8652
260 → 6.7176
270 → 6.6356
280 → 6.6904
290 → 6.5938
300 → 6.2833
```

Loss 不要求每一步单调下降。

正确判断标准：

```text
loss 仍然处于合理量级
没有突然爆炸
没有 NaN
训练可以继续
```

本次：

```text
Loss Resume 后正常
✅
```

---

# 十五、Optimizer State 连续性验证

配置：

```text
gradient_accumulation_steps = 4
```

所以：

```text
4 个 micro steps
→ 1 次 optimizer.step()
```

在：

```text
micro step = 200
```

时：

```text
200 / 4
=
50 次 optimizer update
```

Resume 后继续到：

```text
micro step = 400
```

又增加：

```text
200 / 4
=
50 次 optimizer update
```

如果 Optimizer State 被正确恢复：

```text
50 + 50
=
100
```

所以预计：

```text
AdamW step = 100
```

实际结果：

```text
checkpoint micro step:
400

checkpoint lr:
0.0004721690030098693

optimizer parameter key:
0

AdamW step:
tensor(100.)
```

完全符合预测。

因此：

```text
Optimizer State 连续恢复
✅
```

---

# 十六、如果只恢复 Model Weight 会发生什么

假设：

```text
step 200
```

时保存模型。

Resume 时只：

```python
model.load_state_dict(...)
```

然后重新：

```python
optimizer = AdamW(...)
```

那么：

```text
模型参数
→ 恢复

AdamW m
→ 清零

AdamW v
→ 清零

AdamW step
→ 从 0 开始
```

从 step 201 跑到 step 400：

```text
200 micro steps
÷ 4
=
50 optimizer updates
```

此时 AdamW 内部应该只有：

```text
step ≈ 50
```

而本次真实结果是：

```text
step = 100
```

因此直接证明：

> AdamW 并没有重新初始化，而是从 checkpoint 的历史状态继续。

---

# 十七、exp_avg 与 exp_avg_sq

step 400 checkpoint 中：

```text
exp_avg shape:
(6400, 768)

exp_avg_sq shape:
(6400, 768)
```

前几个值：

```text
exp_avg:
[-7.9437e-07,
  1.0065e-06,
 -9.7758e-07]

exp_avg_sq:
[1.2890e-12,
 5.2410e-12,
 7.7470e-13]
```

`exp_avg` 可以近似理解为：

```text
梯度的一阶指数移动平均
m_t
```

`exp_avg_sq` 可以近似理解为：

```text
梯度平方的二阶指数移动平均
v_t
```

AdamW 会结合：

```text
m_t
v_t
optimizer step
learning rate
weight decay
```

决定参数更新。

---

# 十八、为什么 exp_avg Shape 与参数 Shape 一致

当前检查的第一个 parameter：

```text
model.embed_tokens.weight
```

Shape：

```text
(6400, 768)
```

AdamW 对该参数维护：

```text
exp_avg
(6400, 768)

exp_avg_sq
(6400, 768)
```

也就是：

```text
Parameter Tensor
↓
对应一份 m
↓
对应一份 v
```

这也是 Resume Checkpoint 比普通 Model Weight 大很多的重要原因。

---

# 十九、Checkpoint 大小对比

本次：

```text
普通模型权重：
≈ 132 MB

Resume Checkpoint：
≈ 619 MB
```

原因之一就是 Resume Checkpoint 还保存 Optimizer State，而 AdamW 需要维护：

```text
exp_avg
exp_avg_sq
step
```

---

# 二十、Program Stop Position 与 Checkpoint Position 不一定一样

例如：

```text
程序实际已经训练到：
step 217

最近保存：
step 200
```

如果突然崩溃：

```text
Resume
↓
恢复 step 200 checkpoint
```

那么：

```text
201 ~ 217
```

这部分尚未保存的计算会重新执行。

因此：

```text
checkpoint interval
```

本质上是在权衡：

```text
保存频率
vs
I/O 开销
vs
最多允许丢失多少训练进度
```

---

# 二十一、完整 Resume Training 流程

通用流程：

```text
Train
↓
Step N
↓
Save Checkpoint
↓
Stop Program

重新启动

↓

Create Model
↓
Load Model State

Create Optimizer
↓
Load Optimizer State

Create Scheduler（如果有）
↓
Load Scheduler State

Restore Scaler（如果需要）

Restore Epoch / Step
↓
Skip Finished Batches
↓
Continue Training
```

目标：

> 尽可能让重新启动后的训练状态，与程序从未中断时保持一致。

---

# 二十二、当前 MiniMind Resume 流程

当前 MiniMind 可以理解成：

```text
Resume Checkpoint
│
├── model
├── optimizer
├── scaler
├── epoch
├── step
├── world_size
└── wandb_id
```

加载后：

```text
恢复 Model
↓
恢复 Optimizer
↓
恢复 Scaler
↓
恢复 Epoch
↓
恢复 Step
↓
SkipBatchSampler 跳过已训练 batch
↓
从下一个 step 继续
```

当前没有独立：

```text
Scheduler State
```

LR 通过：

```text
get_lr(current_step, total_steps, base_lr)
```

重新计算。

---

# 二十三、今日关键实验链路

```text
检查 checkpoint keys
↓
确认 Model / Optimizer / Step 等存在

检查 model parameter
↓
load_state_dict()
↓
max abs diff = 0
↓
Model State 恢复 ✅

重新设计 Resume 实验
epochs = 2
↓
训练到 step 200
↓
完整保存 checkpoint
↓
停止

重新启动
--from_resume 1
↓
跳过前 200 step
↓
从 201 开始
↓
Training Step 连续 ✅

检查 LR
step 200
0.00049293
↓
step 210
0.00049221
↓
...
step 400
0.00047217
↓
LR 连续 ✅

检查 AdamW
step 400
↓
AdamW step = 100
↓
Optimizer State 连续 ✅
```

---

# 二十四、今日验收

```text
能保存 Checkpoint                         ✅
能加载 Checkpoint                         ✅
Model Parameter 完全恢复                  ✅
max abs diff = 0                          ✅
torch.equal = True                        ✅
能 Resume Training                        ✅
Checkpoint step = 200                     ✅
Resume 从 step 201 开始                   ✅
Training Step 连续                        ✅
Learning Rate 连续                        ✅
Loss Resume 后保持合理量级                ✅
Optimizer State 连续                      ✅
AdamW step = 100                          ✅
exp_avg 恢复                              ✅
exp_avg_sq 恢复                           ✅
能解释 Model State                        ✅
能解释 Optimizer State                    ✅
能解释 Scheduler State                    ✅
能解释为什么只加载 Model Weight 不等于 Resume ✅
```

---

# 二十五、最重要的概念对比

```text
Model Weight
=
模型现在学成什么样
```

而：

```text
Resume Checkpoint
=
整个训练现场快照
```

真正 Resume 追求：

```text
Model
+
Optimizer
+
Scheduler（如果存在）
+
Scaler（如果需要）
+
Epoch
+
Step
+
其他训练状态
```

目标：

```text
程序只是暂停过
而不是拿着旧模型重新开一场训练
```

---

# 二十六、最短复习版

```text
Checkpoint
≠
只保存 model.state_dict()
```

完整 Resume 尽量恢复：

```text
model
optimizer
scheduler（如果有）
scaler
epoch
step
...
```

Model State：

```text
模型参数
```

Optimizer State：

```text
AdamW:
exp_avg
exp_avg_sq
step
```

Scheduler State：

```text
LR schedule 当前进度
```

当前 MiniMind：

```text
没有独立 scheduler
LR = get_lr(current_step, total_steps, lr)
```

本次实验：

```text
checkpoint:
step 200
lr = 0.0004929312
```

Resume：

```text
跳过前 200 step
从 201 开始
```

LR：

```text
200 → 0.00049293
210 → 0.00049221
...
400 → 0.00047217
```

Optimizer：

```text
GA = 4

400 micro steps
→ 100 optimizer updates

实际：
AdamW step = 100
```

参数恢复：

```text
max abs diff = 0
torch.equal = True
```

最终结论：

```text
Model State 恢复 ✅
Optimizer State 恢复 ✅
Training Step 连续 ✅
Learning Rate 连续 ✅
Resume Training 成功 ✅
```

---

# 二十七、当日口头总结

今天学习的是 Checkpoint 和 Resume Training。

普通 Model Weight 主要保存模型参数，回答的是“模型现在学成什么样”；Resume Checkpoint 则更像训练现场快照，除了 Model State，还需要尽量保存 Optimizer State、Epoch、Step、Scaler，以及项目中存在的话 Scheduler State。

本次首先读取 MiniMind 的 Resume Checkpoint，确认其中包含 model、optimizer、epoch、step、world_size、wandb_id 和 scaler。随后选择 `model.embed_tokens.weight` 做参数恢复实验，加载 checkpoint 后参数的最大绝对误差为 0，`torch.equal()` 为 True，说明 Model State 被完全恢复。

为了正确验证 Resume，没有直接使用昨天已经跑到 cosine schedule 终点的 checkpoint，而是重新设计了一个 epochs=2 的训练实验。第一次训练到 step 200 保存 checkpoint 后停止，checkpoint 中记录 `epoch=0`、`step=200`、`lr≈0.00049293`。

重新启动时使用 `--from_resume 1`，程序明确提示跳过前 200 个 step，从 step 201 开始。之后 step 210、220、230 等日志连续出现，Learning Rate 也从 step 200 的约 0.00049293 平滑下降，没有重新跳回初始 5e-4，说明训练进度和 LR schedule 成功衔接。

最后继续训练到 step 400，并检查 AdamW 的内部状态。由于 Gradient Accumulation 为 4，所以 400 个 micro steps 对应 100 次 optimizer update。实际 checkpoint 中 AdamW 的 `step` 正好为 100，而不是 Resume 之后重新计数得到的 50，这直接证明 Optimizer State 也被真正恢复了。同时还能看到 `exp_avg` 和 `exp_avg_sq`，它们分别对应 AdamW 的一阶和二阶历史统计量。

因此，真正的 Resume Training 不是只加载模型权重，而是尽可能恢复整个训练现场，让重新启动后的程序像只是暂停了一次一样继续运行。
