# Week 4 Day 1 - MiniMind Project Structure

## Environment

- Repo: `jingyaogong/minimind`
- Git Commit: `7a9137d`
- PyTorch: `2.14.0`
- Transformers: `5.17.0`
- Device: Apple MPS
- MiniMind-3 Params: `63.91M`

---

## Repository Structure

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
Model
Main model source:
model/model_minimind.py
Important classes:
MiniMindConfig
MiniMindForCausalLM
Model config is also defined in:
model/model_minimind.py
Tokenizer
Tokenizer files:
model/tokenizer.json
model/tokenizer_config.json
Tokenizer vocabulary size:
6400
Special tokens:
BOS: <|im_start|>   id=1
EOS: <|im_end|>     id=2
PAD: <|endoftext|>  id=0
UNK: <|endoftext|>  id=0
The tokenizer uses subword / byte-level style tokenization rather than simple character-level tokenization.
Example:
中国的首都是北京。

中国 | 的 | 首 | 都是 | 北京 | 。
Token IDs:
[1405, 296, 1408, 2462, 5412, 302]
Another example:
机器学习是人工智能的重要分支。

机器学习 | 是 | 人工智能 | 的重要 | 分 | 支 | 。
Normal encode() does not automatically add BOS/EOS even when:
add_special_tokens=True
Chat Template
Chat template is defined in:
model/tokenizer_config.json
It uses special tokens such as:
<|im_start|>
<|im_end|>
to format system / user / assistant / tool messages.
Detailed Chat Template behavior is not studied yet.
Dataset
Dataset implementation:
dataset/lm_dataset.py
Detailed Dataset logic will be studied on Day 3.
Pretrain
Pretraining entry:
trainer/train_pretrain.py
Detailed training loop will be studied on Day 3 / Day 4.
Checkpoint
Checkpoint utility:
trainer/trainer_utils.py
Main function:
lm_checkpoint(...)
Pretrain calls it from:
trainer/train_pretrain.py
Default checkpoint directory:
../checkpoints
Two kinds of checkpoint are saved.
Model Weight
*.pth
Mainly contains:
model.state_dict()
Suitable for model loading / inference.
Resume Checkpoint
*_resume.pth
Contains training state such as:
model
optimizer
epoch
step
world_size
wandb_id
Additional objects passed through **kwargs, such as scaler, can also be saved through their state_dict().
Inference
Main local inference entry:
eval_llm.py
Pipeline:
Prompt
↓
Tokenizer
↓
input_ids / attention_mask
↓
Model
↓
model.generate()
↓
generated_ids
↓
tokenizer.decode()
↓
Text
For pretrained weights:
inputs = tokenizer.bos_token + prompt
For chat/SFT models:
tokenizer.apply_chat_template(...)
Transformers Model
Downloaded model:
minimind-3/
Main weight:
model.safetensors
Size:
~122 MB
Inference command:
python eval_llm.py \
  --load_from ./minimind-3 \
  --device mps \
  --max_new_tokens 64
Observed:
Model Params: 63.91M
Speed: ~34.84 tokens/s
Example prompt:
中国的首都是哪里？
The inference pipeline ran successfully, but the generated factual answer was incorrect, showing that successful inference does not imply high model capability.
Day 1 Summary
Current MiniMind map:
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