# Week 4 Day 1 - MiniMind 项目结构、Tokenizer 与 Inference

## 0. 中文复习总结

周一的目标不是深入研究 Transformer 内部结构，而是先把整个 MiniMind 项目跑通并建立项目地图，明确：

```text
Repo
↓
Tokenizer
↓
Dataset
↓
Model
↓
Training Script
↓
Checkpoint
↓
Inference
```

今天完成的核心事情包括：

```text
1. Clone MiniMind 源码
2. 确认当前 commit
3. 找到 Model / Tokenizer / Dataset / Pretrain / Config / Checkpoint / Inference
4. 测试 Tokenizer encode / decode
5. 查看 vocab_size 和特殊 token
6. 验证普通 encode 不自动加入 BOS / EOS
7. 找到 Chat Template
8. 下载 minimind-3 模型
9. 检查 Apple MPS
10. 成功运行最简单的 Inference
```

当前源码路径：

```text
~/Projects/Learning_codes/Learn_minimind/minimind
```

当前 Git commit：

```text
7a9137d
```

项目核心文件映射：

```text
Model
→ model/model_minimind.py

Tokenizer
→ model/tokenizer.json
→ model/tokenizer_config.json

Dataset
→ dataset/lm_dataset.py

Pretrain
→ trainer/train_pretrain.py

Config
→ model/model_minimind.py
→ MiniMindConfig

Checkpoint
→ trainer/trainer_utils.py
→ lm_checkpoint()

Inference
→ eval_llm.py
```

Tokenizer 的核心理解：

```text
Raw Text
↓
Tokenizer
↓
Token IDs
↓
Model

Model Output IDs
↓
Tokenizer.decode()
↓
Text
```

MiniMind Tokenizer：

```text
vocab_size = 6400
```

特殊 token：

```text
BOS = <|im_start|>   id=1
EOS = <|im_end|>     id=2
PAD = <|endoftext|>  id=0
UNK = <|endoftext|>  id=0
```

普通文本 encode 时：

```text
add_special_tokens=True
```

在当前 tokenizer 中并不会自动把 BOS / EOS 加进去。

因此 MiniMind 的预训练或推理代码需要根据场景手动处理 BOS / EOS。

例如 pretrain inference 中：

```python
inputs = tokenizer.bos_token + prompt
```

MiniMind 的 tokenizer 不是简单的“一个汉字一个 token”。

例如：

```text
中国的首都是北京。
```

会被切成类似：

```text
中国 | 的 | 首 | 都是 | 北京 | 。
```

而：

```text
机器学习是人工智能的重要分支。
```

会出现：

```text
机器学习 | 是 | 人工智能 | 的重要 | 分 | 支 | 。
```

英文也可能被拆成 subword，例如：

```text
Hello MiniMind.
```

可以拆成：

```text
Hello |  M | in | i | M | ind | .
```

所以必须牢记：

```text
Token ≠ 字符
Token ≠ 单词
```

它是 tokenizer 学到的 subword / byte-level 单元。

Inference 跑通后，完整链路是：

```text
Prompt
↓
Tokenizer
↓
input_ids / attention_mask
↓
MiniMind
↓
model.generate()
↓
generated_ids
↓
Tokenizer.decode()
↓
Text
```

今天成功运行的模型：

```text
minimind-3
```

参数量：

```text
63.91M
```

权重文件：

```text
minimind-3/model.safetensors
```

大小约：

```text
122 MB
```

设备：

```text
Apple MPS
```

推理命令：

```bash
python eval_llm.py \
  --load_from ./minimind-3 \
  --device mps \
  --max_new_tokens 64
```

模型虽然成功生成文本，但回答质量并不一定正确。

这说明：

```text
Inference 跑通
≠
模型能力很强
```

周一的主要目标是验证：

```text
Tokenizer
+
Model Loading
+
Generation Pipeline
```

全部可以正常工作。

---

# 1. Repository 基本信息

本地项目：

```text
Learn_minimind/
├── minimind/
└── week4_minimind/
```

MiniMind 源码：

```text
Learn_minimind/minimind
```

当前 Git commit：

```text
7a9137d
```

顶层目录：

```text
minimind/
├── dataset/
├── model/
├── scripts/
├── trainer/
├── eval_llm.py
├── requirements.txt
├── README.md
└── README_en.md
```

---

# 2. 项目地图

周一首先需要建立：

```text
Repo
↓
Tokenizer
↓
Dataset
↓
Model
↓
Training
↓
Checkpoint
↓
Inference
```

当前项目中的对应关系：

```text
Model
→ model/model_minimind.py

Tokenizer
→ model/tokenizer.json
→ model/tokenizer_config.json

Dataset
→ dataset/lm_dataset.py

Pretrain
→ trainer/train_pretrain.py

Config
→ model/model_minimind.py
→ MiniMindConfig

Checkpoint
→ trainer/trainer_utils.py
→ lm_checkpoint()

Inference
→ eval_llm.py
```

---

# 3. Model

核心模型源码：

```text
model/model_minimind.py
```

主要配置类：

```python
class MiniMindConfig(PretrainedConfig):
```

主要模型类：

```python
class MiniMindModel(nn.Module):
```

以及：

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
```

周一只要求找到位置。

具体 Attention、RoPE、RMSNorm、SwiGLU、Block、LM Head 在周二详细学习。

---

# 4. Tokenizer 文件

Tokenizer 相关文件：

```text
model/tokenizer.json
model/tokenizer_config.json
```

加载方式：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "model"
)
```

当前 tokenizer：

```text
vocab_size = 6400
```

---

# 5. Tokenizer 基本流程

Tokenizer 负责：

```text
文本
↓
Tokenization
↓
Token IDs
```

例如：

```text
"中国的首都是北京。"
```

经过 tokenizer：

```text
Raw Text
↓
Tokens
↓
Token IDs
```

模型只接收：

```text
Token IDs
```

而不是直接接收字符串。

反过来：

```text
Token IDs
↓
tokenizer.decode()
↓
Text
```

---

# 6. 中文 Tokenization 示例 1

文本：

```text
中国的首都是北京。
```

Token IDs：

```text
[1405, 296, 1408, 2462, 5412, 302]
```

逐段 decode：

```text
1405 → 中国
296  → 的
1408 → 首
2462 → 都是
5412 → 北京
302  → 。
```

可以理解为：

```text
中国 | 的 | 首 | 都是 | 北京 | 。
```

说明：

```text
一个 token
可以是一个字
也可以是多个字组成的 subword
```

---

# 7. 中文 Tokenization 示例 2

文本：

```text
机器学习是人工智能的重要分支。
```

Token IDs：

```text
[2683, 357, 2225, 2297, 513, 976, 302]
```

对应：

```text
机器学习
是
人工智能
的重要
分
支
。
```

所以：

```text
Token ≠ 单字
```

Tokenizer 会学习高频片段。

---

# 8. 英文 Tokenization 示例

文本：

```text
Hello MiniMind.
```

Token IDs：

```text
[1602, 869, 301, 108, 80, 916, 49]
```

逐段：

```text
1602 → Hello
869  →  M
301  → in
108  → i
80   → M
916  → ind
49   → .
```

大致：

```text
Hello |  M | in | i | M | ind | .
```

说明英文同样是 subword tokenization。

---

# 9. 为什么 tokens 看起来像乱码

在 tokenizer 内部打印 tokens 时，可能看到类似：

```text
ä¸ŃåĽ½
```

这种形式。

这不代表 tokenizer 损坏。

这是 byte-level tokenizer 内部的字节映射表示。

判断 tokenizer 是否正常，最重要的是：

```text
encode
↓
decode
```

是否能恢复原文本。

例如：

```text
中国的首都是北京。
↓ encode
Token IDs
↓ decode
中国的首都是北京。
```

只要 round-trip 正常，就说明 tokenizer 工作正常。

---

# 10. Special Tokens

当前特殊 token：

```text
BOS:
<|im_start|>
id = 1

EOS:
<|im_end|>
id = 2

PAD:
<|endoftext|>
id = 0

UNK:
<|endoftext|>
id = 0
```

需要注意：

```text
PAD 和 UNK
共享 id = 0
```

这是当前 tokenizer 的设计。

---

# 11. BOS / EOS

BOS：

```text
Beginning Of Sequence
```

用于表示：

```text
序列开始
```

EOS：

```text
End Of Sequence
```

用于表示：

```text
序列结束
```

当前 MiniMind：

```text
BOS id = 1
EOS id = 2
```

---

# 12. PAD

PAD：

```text
Padding Token
```

用于把不同长度 sequence 补成相同长度。

例如：

```text
sample A:
[BOS, 10, 20, EOS]

sample B:
[BOS, 11, 12, 13, 14, EOS]
```

为了组成 batch，可以变成：

```text
sample A:
[BOS, 10, 20, EOS, PAD, PAD]

sample B:
[BOS, 11, 12, 13, 14, EOS]
```

这样才能组成统一 Tensor：

```text
[B,T]
```

---

# 13. 普通 encode 是否自动加 BOS / EOS

测试：

```python
tokenizer(
    text,
    add_special_tokens=False
)
```

和：

```python
tokenizer(
    text,
    add_special_tokens=True
)
```

当前 tokenizer 得到的 Token IDs 相同。

因此结论：

```text
普通 encode
不会自动加入 BOS / EOS
```

这点非常重要。

`add_special_tokens=True` 并不意味着当前 tokenizer 一定会自动加：

```text
BOS
EOS
```

实际行为要以 tokenizer 配置和真实输出为准。

---

# 14. 为什么 Pretrain Inference 手动加 BOS

在 `eval_llm.py` 中：

```python
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt
```

说明 pretrain 模式下需要手动：

```text
BOS + prompt
```

因为普通 tokenizer encode 不会自动加入 BOS。

这与周一的 tokenizer 实验完全对应。

---

# 15. Chat Template

Chat Template 位于：

```text
model/tokenizer_config.json
```

里面定义了：

```text
system
user
assistant
tool
thinking
```

等消息如何转换成模型输入格式。

其中会使用：

```text
<|im_start|>
<|im_end|>
```

等特殊 token。

周一只要求：

```text
找到 Chat Template
```

不需要深入理解完整 Jinja 模板。

---

# 16. 普通 Tokenization 和 Chat Template 的区别

普通文本：

```python
tokenizer(text)
```

负责：

```text
Text
↓
Token IDs
```

Chat Template：

```python
tokenizer.apply_chat_template(...)
```

负责先把：

```text
system
user
assistant
```

这样的结构化对话，拼成模型期望的文本格式。

然后再 tokenize。

所以：

```text
普通 encode
≠
apply_chat_template
```

---

# 17. Dataset

Dataset 源码：

```text
dataset/lm_dataset.py
```

周一只需要找到位置。

后面周三会详细学习：

```text
PretrainDataset
SFTDataset
DPODataset
...
```

以及：

```text
Dataset
→ DataLoader
→ Batch
```

的流程。

---

# 18. Pretrain Script

预训练入口：

```text
trainer/train_pretrain.py
```

这是之后真正运行 MiniMind Pretrain Baseline 的脚本。

周一只需要知道：

```text
Pretrain
→ trainer/train_pretrain.py
```

后面会继续研究：

```text
Dataset
DataLoader
Forward
Loss
Backward
Gradient Accumulation
Optimizer
Checkpoint
```

---

# 19. Config

模型配置定义在：

```text
model/model_minimind.py
```

类：

```python
MiniMindConfig
```

它控制：

```text
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
vocab_size
intermediate_size
RoPE
RMSNorm
MoE
...
```

周二会详细阅读。

---

# 20. Checkpoint

Checkpoint 核心函数：

```text
trainer/trainer_utils.py
```

函数：

```python
lm_checkpoint(...)
```

训练脚本：

```text
trainer/train_pretrain.py
```

会调用这个函数保存和恢复训练。

---

# 21. 普通模型权重

MiniMind 会保存类似：

```text
*.pth
```

主要内容：

```text
model.state_dict()
```

用途：

```text
Inference
Evaluation
Load Model Weights
```

可以理解为：

```text
模型现在的参数是什么
```

---

# 22. Resume Checkpoint

Resume 文件：

```text
*_resume.pth
```

包含训练状态，例如：

```text
model
optimizer
epoch
step
world_size
wandb_id
```

如果通过：

```python
**kwargs
```

传入像：

```text
scaler
```

这种拥有 `state_dict()` 的对象，也可以保存进去。

因此 Resume Checkpoint 更接近：

```text
训练现场快照
```

而普通模型权重只是：

```text
模型参数快照
```

---

# 23. 为什么有 .tmp + os.replace

Checkpoint 保存时：

```text
先写临时文件
↓
保存完成
↓
os.replace
↓
替换正式文件
```

例如：

```text
checkpoint.pth.tmp
↓
checkpoint.pth
```

这样比直接覆盖正式 checkpoint 更安全。

如果写文件过程中程序崩溃，可以减少正式 checkpoint 被写坏的风险。

---

# 24. DDP / Compile 模型解包

Checkpoint 保存前可能会看到：

```python
raw_model = model.module \
    if isinstance(model, DistributedDataParallel) \
    else model
```

这是为了：

```text
如果模型被 DDP 包裹
↓
取出真正的 model
```

还会处理：

```text
_orig_mod
```

用于兼容：

```text
torch.compile
```

这是工程细节，周一认识即可。

---

# 25. Inference 入口

主要推理脚本：

```text
eval_llm.py
```

核心流程：

```text
Tokenizer
↓
Model Loading
↓
Prompt
↓
model.generate()
↓
generated_ids
↓
decode
```

---

# 26. 两种模型加载路线

`eval_llm.py` 中存在两种加载思路。

一种是 MiniMind 原生 `.pth` 权重：

```text
MiniMindForCausalLM
+
MiniMindConfig
+
*.pth
```

另一种是 Transformers 格式模型：

```text
AutoModelForCausalLM.from_pretrained(...)
```

我们周一下载的：

```text
minimind-3/
```

属于 Transformers 格式模型。

---

# 27. minimind-3 文件

下载目录：

```text
minimind-3/
```

主要文件：

```text
README.md
README_en.md
config.json
configuration.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
```

模型权重：

```text
model.safetensors
```

本地大小约：

```text
122 MB
```

---

# 28. MPS

Mac 上检查：

```python
torch.backends.mps.is_available()
torch.backends.mps.is_built()
```

实际结果：

```text
MPS available: True
MPS built: True
```

所以这台 Mac 可以使用：

```text
Apple GPU / MPS
```

进行推理。

---

# 29. 最简单 Inference

命令：

```bash
python eval_llm.py \
  --load_from ./minimind-3 \
  --device mps \
  --max_new_tokens 64
```

模型成功加载：

```text
Model Params: 63.91M
```

手动输入：

```text
中国的首都是哪里？
```

模型能够正常生成文本。

生成速度约：

```text
34.84 tokens/s
```

说明：

```text
Model Loading
Tokenizer
MPS
Generation
Decode
```

整条 inference pipeline 已经跑通。

---

# 30. 为什么模型回答错误但 Inference 仍然算成功

模型回答内容存在事实错误。

这并不代表 inference pipeline 出错。

需要区分：

```text
程序是否正常运行
```

和：

```text
模型回答质量是否高
```

两件事。

周一的验收目标是：

```text
能正常加载模型
能正常 tokenize
能正常 generate
能正常 decode
```

而不是要求：

```text
63.91M 小模型
必须像大型 LLM 一样回答准确
```

所以：

```text
Inference Success
≠
Model Quality Good
```

---

# 31. Inference 完整数据流

完整流程：

```text
Prompt
"中国的首都是哪里？"

↓

Tokenizer

↓

input_ids
attention_mask

↓

MiniMind Model

↓

model.generate()

↓

generated_ids

↓

去掉 Prompt 部分

↓

Tokenizer.decode()

↓

Generated Text
```

---

# 32. generated_ids 为什么要去掉 Prompt

生成结果通常包含：

```text
原 Prompt Token IDs
+
新生成 Token IDs
```

所以 decode 前会取：

```python
generated_ids[0][len(inputs["input_ids"][0]):]
```

也就是：

```text
跳过原输入 token
只 decode 新生成内容
```

否则输出中会再次包含用户 Prompt。

---

# 33. Week 1 / Week 2 / Week 3 到 MiniMind 的映射

目前可以看到之前学过的知识开始进入真实项目。

Week 1：

```text
Tensor
Dataset
DataLoader
Training Loop
```

Week 2：

```text
Embedding
Attention
RoPE
RMSNorm
SwiGLU
KV Cache
Transformer Block
```

Week 3：

```text
Next-token Prediction
CrossEntropy
Gradient Accumulation
AdamW
Checkpoint
Generation
```

Week 4：

```text
把这些东西
映射到真实 MiniMind 源码
```

---

# 34. 当日中文口头总结

可以用下面这段话复述周一：

> 周一首先把 MiniMind 项目克隆到本地，并确认当前源码 commit。然后对项目目录进行定位，找到模型源码 `model/model_minimind.py`、Tokenizer 文件、Dataset、预训练脚本、Config、Checkpoint 工具以及 `eval_llm.py` 推理入口，从而建立 MiniMind 项目的整体地图。
>
> 接着重点测试 Tokenizer。MiniMind 当前 vocabulary size 是 6400，BOS 为 `<|im_start|>`、EOS 为 `<|im_end|>`，PAD 和 UNK 都使用 `<|endoftext|>`。通过中文和英文 encode/decode 实验可以看到，MiniMind Tokenizer 使用 subword / byte-level 风格的切分方式，一个 token 不一定对应一个汉字或一个完整单词。Tokenizer 内部打印出来的部分 token 字符串可能看起来像乱码，但只要 encode 后再 decode 能正确恢复文本，就说明 tokenizer 工作正常。
>
> 实验还确认了普通 tokenizer encode 即使设置 `add_special_tokens=True`，当前配置也不会自动给普通文本加入 BOS 和 EOS，因此 MiniMind 在预训练 Dataset 或 pretrain inference 中需要根据具体逻辑手动加入这些特殊 token。Chat Template 也已经定位到 `tokenizer_config.json`，它负责把 system、user、assistant 等对话消息转换为模型期望的格式，但周一只要求找到位置，不深入解析。
>
> 最后下载了 `minimind-3` 模型，确认 `model.safetensors` 权重存在，并检查当前 Mac 的 MPS 可用。通过 `eval_llm.py --load_from ./minimind-3 --device mps` 成功完成一次推理，说明从 Prompt、Tokenizer、Model Loading、`model.generate()`、generated IDs 到 decode 的完整 inference pipeline 已经跑通。虽然小模型生成的事实内容可能不准确，但这和程序是否跑通是两个不同问题。

---

# 35. 最短复习版

如果只剩 1 分钟复习周一，记住：

```text
MiniMind 项目地图

Model
→ model/model_minimind.py

Tokenizer
→ model/tokenizer.json
→ model/tokenizer_config.json

Dataset
→ dataset/lm_dataset.py

Pretrain
→ trainer/train_pretrain.py

Checkpoint
→ trainer/trainer_utils.py
→ lm_checkpoint()

Inference
→ eval_llm.py
```

Tokenizer：

```text
vocab_size = 6400

BOS = 1
EOS = 2
PAD = 0
UNK = 0
```

核心：

```text
Text
↓ Tokenizer
Token IDs
↓ Model
Generated IDs
↓ Decode
Text
```

注意：

```text
普通 encode
不会自动加入 BOS/EOS
```

Inference：

```text
Prompt
↓
Tokenizer
↓
input_ids
↓
MiniMind
↓
generate
↓
generated_ids
↓
decode
↓
Text
```

周一最终完成：

```text
项目结构看懂
+
Tokenizer 跑通
+
Inference 跑通
```
