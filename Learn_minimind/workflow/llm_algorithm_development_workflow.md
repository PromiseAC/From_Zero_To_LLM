# LLM 算法学习与远程开发工作流

## 0. 核心目标

以后不再以：

```text
复制命令
→ 运行
→ 看输出
```

作为主要学习方式。

统一改成：

```text
读源码
→ 理解
→ 开 Branch
→ 修改代码
→ 看 Git Diff
→ Commit 固定版本
→ GPU 实验
→ 分析结果
→ 写实验记录
→ Push
→ PR Review
→ Merge
→ 下一个任务
```

目标有两个：

1. 真正学习 LLM / 算法工程，而不是只会把项目跑起来。
2. 尽量模拟真实团队中的远程开发、Git 分支和实验管理方式。

---

## 1. 当前开发架构

```text
MacBook
VS Code / Codex / 浏览器
        │
        │ Remote SSH
        ▼
WSL Ubuntu
/home/promisea/Projects/From_Zero_To_LLM
        │
        ├── 源码
        ├── Git Working Tree
        ├── Python / PyTorch
        ├── Dataset
        ├── Checkpoint
        └── RTX 3060
        │
        │ git push / pull
        ▼
GitHub
PromiseAC/From_Zero_To_LLM
```

职责：

```text
Mac
= 操作界面

Ubuntu
= 真正开发 + 运行代码

GitHub
= 代码版本的中央仓库 / Source of Truth
```

VS Code 虽然运行在 Mac 上，但 Remote SSH 打开的文件实际位于 Ubuntu。

因此：

```text
Mac 上点开文件
↓
实际读取 Ubuntu 文件

Mac 上修改代码
↓
实际修改 Ubuntu Working Tree

VS Code Terminal 运行 Python
↓
实际使用 Ubuntu Python / CUDA / RTX 3060
```

---

## 2. 正式工作目录

主工作区：

```text
/home/promisea/Projects/From_Zero_To_LLM
```

Week 4：

```text
From_Zero_To_LLM/
├── README.md
├── .gitignore
└── week4_minimind/
    ├── README.md
    ├── notes/
    ├── experiments/
    └── minimind/
        ├── model/
        ├── trainer/
        ├── dataset/
        ├── configs/
        ├── eval_llm.py
        └── ...
```

旧目录：

```text
/home/promisea/Projects/minimind
```

只保留为：

```text
MiniMind upstream/reference repo
```

以后不要再作为日常开发目录。

---

## 3. 每个任务怎么开始

先同步最新主分支：

```bash
git switch main
git pull
```

然后新建任务 branch：

```bash
git switch -c week4/fixed-prompt-eval
```

原则：

```text
一个明确任务
≈
一个 Branch
```

例如：

```text
week4/fixed-prompt-eval
week5/sft-baseline
week5/lora-ablation
week6/dpo-baseline
```

不要把所有改动长期堆在 `main`。

---

## 4. 学习顺序：先源码，后命令

遇到新任务时，顺序固定为：

```text
1. 打开相关源码
2. 找入口
3. 找核心函数
4. 理解数据流
5. 理解 Shape / State / Config
6. 修改少量关键代码
7. 最后运行实验
```

例如 Resume Training：

```text
train_pretrain.py
↓
from_resume
↓
checkpoint load
↓
model.load_state_dict
↓
optimizer.load_state_dict
↓
start_epoch / start_step
↓
SkipBatchSampler
```

例如 Fixed Prompt Evaluation：

```text
eval_llm.py
↓
model load
↓
tokenizer
↓
prompt
↓
generate
↓
sampling config
↓
decode
```

核心原则：

> Terminal 是验证代码的工具，不是学习的主体。

---

## 5. 修改代码后先看 Diff

修改后先执行：

```bash
git status
git diff
```

确认：

```text
我到底改了哪些文件？
改了哪些行？
有没有误改？
有没有误加 Dataset / Checkpoint？
```

固定习惯：

```text
修改
→ 看 Diff
→ 再决定是否 Commit
```

---

## 6. 为什么正式实验前要 Commit

比较正式的实验建议：

```text
改代码
↓
改 Config
↓
检查 Diff
↓
Commit
↓
得到固定 Git SHA
↓
再运行实验
```

例如：

```bash
git add week4_minimind/minimind/eval_llm.py
git commit -m "feat: add fixed prompt evaluation"
git rev-parse HEAD
```

实验记录里写：

```text
git_commit: <实际 SHA>
```

这样以后看到：

```text
loss = 5.18
```

可以准确回答：

> 当时到底运行的是哪份代码？

---

## 7. Config 必须版本化

应该进 Git：

```text
源码
configs/*.yaml
README
notes
experiments
小型文本结果
```

Config 建议记录：

```text
experiment_id
source_commit
repo_commit_at_run
dataset_version
model config
batch size
gradient accumulation
learning rate
sequence length
seed
precision
device
save interval
run command
```

`repo_commit_at_run` 在真正运行实验时通过：

```bash
git rev-parse HEAD
```

获取，不要提前猜。

---

## 8. 哪些东西不要进 Git

只留在 Ubuntu：

```text
dataset/*.jsonl
dataset/*.parquet
checkpoints/
out/
runs/
wandb/
*.pth
*.pt
*.ckpt
*.safetensors
*.log
*.tmp
```

原则：

```text
Code / Config / Docs
→ Git

Dataset / Checkpoint / Large Logs
→ Remote Machine
```

但大文件的信息要进入实验记录，例如：

```text
checkpoint:
pretrain_resume_001_768_resume.pth

step:
400

git_commit:
...

loss:
...

generation:
...
```

即：

```text
大文件不进 Git
大文件的元数据进 Git
```

---

## 9. GPU 实验统一流程

正式训练前：

```text
Smoke Test
↓
Memory Probe
↓
Benchmark
↓
Freeze Config
↓
Small Full Run
↓
Full Training
```

### Smoke Test

验证：

```text
Dataset
→ DataLoader
→ Model
→ Loss
→ Backward
→ Optimizer
→ Checkpoint
```

### Memory Probe

测试：

```text
合理 micro batch
峰值显存
OOM
稳定性
```

### Benchmark

记录：

```text
micro steps/s
sequences/s
tokens/s
ETA
```

### Freeze Config

固定：

```text
model config
dataset version
micro batch
GA
learning rate
sequence length
precision
seed
```

然后再跑正式实验。

---

## 10. 实验记录

建议放：

```text
week4_minimind/experiments/
```

每次实验至少记录：

```text
experiment_id
date
git_commit
source_commit
config
dataset_version
device
training_time
peak_memory
throughput
train_loss
validation_loss
checkpoint
generation_samples
notes
```

不要只写：

```text
训练成功
```

目标是以后可以完整复盘。

---

## 11. Commit Message 习惯

推荐：

```text
feat:
新增功能

fix:
修复 Bug

docs:
文档 / 实验记录

chore:
工程配置 / 仓库结构

exp:
实验相关修改
```

例如：

```text
feat: add fixed prompt evaluation
exp: benchmark MiniMind pretraining on RTX 3060
docs: record checkpoint resume experiment
chore: track reproducible training configs
```

一个任务也可以多个 commit：

```text
feat: add fixed prompt evaluation
↓
运行实验
↓
docs: record fixed prompt evaluation results
```

这样 Git History 能区分：

```text
实现了什么
vs
实验得到了什么
```

---

## 12. Push 和 PR

第一次 push branch：

```bash
git push -u origin <branch-name>
```

之后：

```bash
git push
```

然后：

```text
Task Branch
↓
Push
↓
Pull Request
↓
Review
↓
Merge
↓
main
```

PR 重点检查：

```text
Files changed
代码逻辑
实验 Config
README / 实验记录
是否误提交 Dataset
是否误提交 Checkpoint
是否有明显 Bug
```

Merge 后：

```bash
git switch main
git pull
git switch -c <next-task>
```

---

## 13. VS Code 推荐使用方式

左侧 Explorer：

```text
看目录
打开源码
看 configs
看 notes
看 experiments
```

中间 Editor：

```text
读源码
改源码
搜索变量
看函数引用
```

底部 Terminal：

```text
git
python
CUDA
训练
测试
```

形成：

```text
左边看结构
中间看代码
下面跑实验
```

而不是只开 Terminal。

---

## 14. Codex 的角色

Codex 适合：

```text
检查仓库结构
修改工程配置
机械性重构
维护 .gitignore
整理 README
检查 Diff
辅助 Branch / Commit / PR
```

但算法学习核心部分：

```text
Attention
Training Loop
Checkpoint
SFT
DPO
GRPO
```

不要完全交给 Codex。

更好的方式：

```text
自己读
→ 自己理解
→ 自己改关键逻辑
→ Codex 辅助检查
```

---

## 15. ChatGPT 的角色

建议：

```text
ChatGPT：
告诉你打开哪个文件
↓
指出关键函数
↓
解释源码和算法
↓
设计小实验
↓
检查理解
↓
分析结果
↓
Review Git Diff / 实验设计
```

你：

```text
自己看源码
↓
自己修改
↓
自己运行
↓
自己解释结果
```

---

## 16. 统一适用于后续所有 LLM 阶段

无论后面是：

```text
Pretrain
SFT
LoRA
DPO
PPO
GRPO
verl
```

都重复：

```text
读源码
↓
理解算法
↓
设计实验
↓
开 Branch
↓
修改源码 / Config
↓
Git Diff
↓
Commit
↓
GPU Run
↓
Monitor
↓
分析 Loss / Metrics / Generation
↓
Bad Case
↓
记录实验
↓
Push
↓
PR Review
↓
Merge
```

算法会变，工程闭环不变。

---

## 17. 示例：Fixed Prompt Evaluation

```text
main
↓
git pull
↓
创建 week4/fixed-prompt-eval

↓ VS Code

打开 eval_llm.py

↓ 阅读

模型如何加载
Tokenizer 如何调用
generate 如何工作
sampling 参数含义

↓ 修改

固定：
Prompt
max_new_tokens
temperature
top_p
seed

↓ Git

git diff
↓
commit

↓ GPU

Early Checkpoint
Middle Checkpoint
Late Checkpoint

↓ Analysis

Train Loss
Validation Loss
Generation Quality

↓ Docs

generation_results.md

↓ Git

commit
push

↓ GitHub

PR
review
merge
```

---

## 18. 开始任务 Checklist

```text
□ main 是否最新
□ 是否创建独立 branch
□ 是否明确任务目标
□ 是否先读源码
□ 是否确认 config
□ 是否明确实验变量
□ 是否明确控制变量
□ 是否计划记录 metrics
```

---

## 19. 实验前 Checklist

```text
□ git status 是否清楚
□ git diff 是否看过
□ 关键代码是否已 commit
□ git_commit 是否记录
□ config 是否进 Git
□ dataset version 是否记录
□ GPU memory 是否足够
□ ETA 是否估算
□ checkpoint 路径是否正确
```

---

## 20. 实验后 Checklist

```text
□ Loss 是否合理
□ Metrics 是否保存
□ Checkpoint 是否生成
□ 是否记录运行时间
□ 是否记录峰值显存
□ 是否记录吞吐
□ 是否记录 generation sample
□ 是否分析 bad cases
□ experiment record 是否完成
□ 是否提交实验结果
```

---

## 21. PR 前 Checklist

```text
□ git status
□ git diff main...HEAD
□ 没有 dataset
□ 没有 checkpoint
□ 没有 model weights
□ 没有 log
□ config 已提交
□ README / experiment record 已更新
□ commit message 清楚
□ branch 已 push
```

---

## 22. 最短复习版

```text
Mac
→ VS Code Remote SSH
→ Ubuntu Dev Machine
→ RTX 3060

GitHub
= Source of Truth
```

每个任务：

```text
git switch main
git pull
git switch -c task-branch
```

然后：

```text
读源码
→ 理解
→ 修改
→ git diff
→ commit
→ GPU 实验
→ 分析
→ 写实验记录
→ push
→ PR
→ review
→ merge
```

Git 保存：

```text
Code
Config
Docs
Experiment Metadata
```

Git 不保存：

```text
Dataset
Checkpoint
Model Weight
Large Logs
```

核心原则：

```text
不要把“运行项目”当成“学习项目”。

真正的学习流程：

源码
→ 算法理解
→ 修改
→ 实验
→ 分析
→ 复现
→ Review
```

---

## 23. 一句话总结

以后所有 LLM 学习任务尽量按照：

```text
Read
→ Understand
→ Branch
→ Modify
→ Diff
→ Commit
→ Experiment
→ Analyze
→ Document
→ Push
→ PR
→ Review
→ Merge
```

完成。
