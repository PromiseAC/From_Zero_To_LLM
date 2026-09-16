# Week 4 Day 1 — MiniMind Project Structure

> A map of the MiniMind codebase, tokenizer, training flow, checkpoints, and local inference.

## 1. Environment

| Item | Value |
| --- | --- |
| Repository | `jingyaogong/minimind` |
| Git commit | `7a9137d` |
| PyTorch | `2.14.0` |
| Transformers | `5.17.0` |
| Device | Apple MPS |
| MiniMind-3 parameters | `63.91M` |

## 2. Repository Structure

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

## 3. Model

### Main implementation

- Source: `model/model_minimind.py`
- Important classes:
  - `MiniMindConfig`
  - `MiniMindForCausalLM`
- The model configuration is also defined in `model/model_minimind.py`.

## 4. Tokenizer

### Files and vocabulary

- `model/tokenizer.json`
- `model/tokenizer_config.json`
- Vocabulary size: `6400`

### Special tokens

| Token | Symbol | ID |
| --- | --- | ---: |
| BOS | `<|im_start|>` | 1 |
| EOS | `<|im_end|>` | 2 |
| PAD | `<|endoftext|>` | 0 |
| UNK | `<|endoftext|>` | 0 |

The tokenizer uses subword / byte-level-style tokenization rather than simple character-level tokenization.

### Examples

Input:

```text
中国的首都是北京。
```

Possible tokenization:

```text
中国 | 的 | 首 | 都是 | 北京 | 。
```

Token IDs:

```python
[1405, 296, 1408, 2462, 5412, 302]
```

Another example:

```text
机器学习 | 是 | 人工智能 | 的重要 | 分 | 支 | 。
```

> `encode()` does not automatically add BOS/EOS, even when `add_special_tokens=True`.

## 5. Chat Template

The chat template is defined in `model/tokenizer_config.json`.

It uses special tokens such as:

```text
<|im_start|>
<|im_end|>
```

These tokens format `system`, `user`, `assistant`, and `tool` messages. The detailed chat-template behavior has not been studied yet.

## 6. Dataset

- Implementation: `dataset/lm_dataset.py`
- Detailed dataset logic: to be studied on Day 3.

## 7. Pretraining

- Entry point: `trainer/train_pretrain.py`
- Detailed training loop: to be studied on Day 3 / Day 4.

## 8. Checkpoints

### Checkpoint utility

- Utility: `trainer/trainer_utils.py`
- Main function: `lm_checkpoint(...)`
- Called by: `trainer/train_pretrain.py`
- Default directory: `../checkpoints`

Two kinds of checkpoints are saved.

### Model weights

- Pattern: `*.pth`
- Main content: `model.state_dict()`
- Use case: model loading and inference

### Resume checkpoints

- Pattern: `*_resume.pth`
- Training state may include:
  - `model`
  - `optimizer`
  - `epoch`
  - `step`
  - `world_size`
  - `wandb_id`

Additional objects passed through `**kwargs`, such as `scaler`, can also be saved through their `state_dict()`.

## 9. Local Inference

### Entry point

- Main script: `eval_llm.py`

### Pipeline

```text
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
```

### Input formatting

For pretrained weights:

```python
inputs = tokenizer.bos_token + prompt
```

For chat/SFT models:

```python
tokenizer.apply_chat_template(...)
```

### Downloaded Transformers model

- Directory: `minimind-3/`
- Main weight: `model.safetensors`
- Size: approximately `122 MB`

Run inference with:

```bash
python eval_llm.py \
  --load_from ./minimind-3 \
  --device mps \
  --max_new_tokens 64
```

Observed results:

- Model parameters: `63.91M`
- Speed: approximately `34.84 tokens/s`
- Example prompt: `中国的首都是哪里？`

The inference pipeline ran successfully, but the generated factual answer was incorrect. This shows that successful inference does not necessarily imply high model capability.

## 10. Day 1 Summary

| Component | Main entry / files |
| --- | --- |
| Model | `model/model_minimind.py` |
| Tokenizer | `model/tokenizer.json`, `model/tokenizer_config.json` |
| Dataset | `dataset/lm_dataset.py` |
| Pretraining | `trainer/train_pretrain.py` |
| Configuration | `MiniMindConfig` in `model/model_minimind.py` |
| Checkpoint | `lm_checkpoint()` in `trainer/trainer_utils.py` |
| Inference | `eval_llm.py` |
