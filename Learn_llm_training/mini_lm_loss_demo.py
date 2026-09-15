import sys
from pathlib import Path
import math

import torch
import torch.nn as nn
from transformers import AutoTokenizer


# ============================================================
# 0. 导入第二周写的 MiniTransformer
# ============================================================

transformer_dir = (
    Path(__file__).resolve().parent.parent
    / "Learn_transformer"
)

sys.path.insert(0, str(transformer_dir))

from mini_transformer import MiniTransformer


# ============================================================
# 1. 固定随机种子
# ============================================================

torch.manual_seed(42)


# ============================================================
# 2. 加载 Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

print("vocab size:", tokenizer.vocab_size)


# ============================================================
# 3. 准备原始文本
# ============================================================

text = "我正在学习大语言模型的训练原理"

token_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

print("text:", text)
print("token ids:", token_ids)
print("token count:", len(token_ids))


# ============================================================
# 4. 转成 Tensor
# ============================================================

token_ids = torch.tensor(
    token_ids,
    dtype=torch.long
)


# ============================================================
# 5. 构造 Causal LM input / label
# ============================================================

# token_ids:
# [t0, t1, t2, t3, ..., tn]

# input:
# [t0, t1, t2, ..., t(n-1)]

# label:
# [t1, t2, t3, ..., tn]

input_ids = token_ids[:-1]
labels = token_ids[1:]


# ============================================================
# 6. 增加 Batch 维度
# ============================================================

# [T]
# ↓
# [1, T]

input_ids = input_ids.unsqueeze(0)
labels = labels.unsqueeze(0)

print()
print("input_ids shape:", input_ids.shape)
print("labels shape:", labels.shape)


# ============================================================
# 7. 创建 MiniTransformer
# ============================================================

model = MiniTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=2,
    intermediate_size=256
)


# ============================================================
# 8. 创建 Loss 和 Optimizer
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# ============================================================
# 9. Training Loop
# ============================================================

print()
print("Start training...")
print("-" * 50)

num_steps = 50

for step in range(num_steps):

    # ----------------------------------------
    # 1. 清空上一轮梯度
    # ----------------------------------------

    optimizer.zero_grad()


    # ----------------------------------------
    # 2. Forward
    #
    # input_ids:
    # [B, T]
    #
    # logits:
    # [B, T, V]
    # ----------------------------------------

    logits = model(input_ids)


    # ----------------------------------------
    # 3. 取出 shape
    # ----------------------------------------

    B, T, V = logits.shape


    # ----------------------------------------
    # 4. Flatten
    #
    # logits:
    # [B, T, V]
    # →
    # [B*T, V]
    #
    # labels:
    # [B, T]
    # →
    # [B*T]
    # ----------------------------------------

    logits_flat = logits.reshape(B * T, V)

    labels_flat = labels.reshape(B * T)


    # ----------------------------------------
    # 5. Cross Entropy Loss
    # ----------------------------------------

    loss = loss_fn(
        logits_flat,
        labels_flat
    )


    # ----------------------------------------
    # 6. Backward
    # ----------------------------------------

    loss.backward()


    # ----------------------------------------
    # 7. 更新参数
    # ----------------------------------------

    optimizer.step()


    # ----------------------------------------
    # 8. 每 5 step 打印一次
    # ----------------------------------------

    if step % 5 == 0:

        ppl = math.exp(loss.item())

        print(
            f"step: {step:2d} | "
            f"loss: {loss.item():.4f} | "
            f"ppl: {ppl:.2f}"
        )


print("-" * 50)
print("Training finished.")

# ============================================================
# 10. 检查训练后的预测
# ============================================================

model.eval()

with torch.no_grad():
    logits = model(input_ids)

    # 每个位置选择 logit 最大的 token
    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )

    print()
print("Prediction after training:")

print("predicted ids:", predicted_ids)
print("labels:", labels)

correct = (
    predicted_ids == labels
).sum().item()

total = labels.numel()

accuracy = correct / total

print(
    f"next-token accuracy: "
    f"{correct}/{total} = {accuracy:.2%}"
)

for i in range(labels.shape[1]):

    context = tokenizer.decode(
        input_ids[0, :i + 1]
    )

    pred = tokenizer.decode([
        predicted_ids[0, i].item()
    ])

    target = tokenizer.decode([
        labels[0, i].item()
    ])

    print(
        f"context: {context!r} | "
        f"pred: {pred!r} | "
        f"target: {target!r}"
    )


# ============================================================
# 11. Greedy vs Sampling
# ============================================================

prompt = "我正在学习大语言模型的"

prompt_ids = tokenizer.encode(
    prompt,
    add_special_token=False
)

prompt_ids = torch.tensor(
    prompt_ids,
    dtype=torch.long
).unsqueeze(0)

print()
print("prompt:", prompt)
print("prompt_ids shape:", prompt_ids.shape)

model.eval()

with torch.no_grad():
    logits = model(prompt_ids)
    # 只取最后一个位置
    next_token_logits = logits[:, -1, :]


print("next_token_logits shape:", next_token_logits.shape)

#1. greedy decoding
greedy_id = torch.argmax(
    next_token_logits,
    dim=-1
)

greedy_text = tokenizer.decode(
    greedy_id.tolist()
)

print("greedy:", greedy_text)

#2. sampling
temperatures = [0.2, 1.0, 2.0]

for temperature in temperatures:

    print()
    print("=" * 50)
    print("temperature:", temperature)

    # 调整 logits
    scaled_logits = next_token_logits / temperature

    # 转为概率
    probs = torch.softmax(
        scaled_logits,
        dim=-1
    )

    # 看概率最高的 token
    top_prob, top_id = torch.max(
        probs,
        dim=-1
    )

    print(
        "top token:",
        tokenizer.decode(top_id.tolist())
    )

    print(
        "top probability:",
        top_prob.item()
    )

    print("Sampling 10 times:")

    for i in range(10):

        sampled_id = torch.multinomial(
            probs,
            num_samples=1
        )

        sampled_text = tokenizer.decode(
            sampled_id[0].tolist()
        )

        print(i, repr(sampled_text))

