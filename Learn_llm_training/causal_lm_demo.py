import torch
import torch.nn as nn
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

text = "我正在学习大语言模型的训练原理"

token_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)
print("text:", text)
print("token ids:", token_ids)
print("token count:", len(token_ids))

token_ids = torch.tensor(token_ids)
input_ids = token_ids[:-1]
labels = token_ids[1:]

print("input ids:", input_ids)
print("labels:", labels)

print("input shape:", input_ids.shape)
print("labels shape:", labels.shape)


print("\nNext Token Prediction:")

for i in range(len(input_ids)):
    context = tokenizer.decode(input_ids[:i + 1])
    target = tokenizer.decode([labels[i]])

    print(f"context: {context!r}")
    print(f"target : {target!r}")
    print()