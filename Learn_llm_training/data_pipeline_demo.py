import random
import torch
from transformers import AutoTokenizer


# ============================================================
# 0. Load Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)


# ============================================================
# 1. Raw Text
# ============================================================

texts = [
    "  我喜欢机器学习  ",
    "",
    "   ",
    "大语言模型很重要\n",
    "\tTransformer 很重要\t",
    "机器学习     很有趣",
    "我喜欢机器学习",
    "大语言模型很重要",
    "深度学习",
    "AI",
    "自然语言处理很重要",
    "神经网络可以学习复杂模式",
    "PyTorch 是常用深度学习框架",
    "Tokenizer 将文本转换成 Token ID",
    "Attention 是 Transformer 的核心机制",
    "语言模型可以预测下一个 Token",
]

print("原始数据量:", len(texts))


# ============================================================
# 2. Cleaning
# ============================================================

cleaned_texts = []

for text in texts:

    # 去除首尾空格、\n、\t
    text = text.strip()

    # 删除空字符串
    if not text:
        continue

    # 将连续空白统一成一个空格
    text = " ".join(text.split())

    cleaned_texts.append(text)


print("Cleaning 后:", len(cleaned_texts))


# ============================================================
# 3. Length Filtering
# ============================================================

min_length = 4
max_length = 30

filtered_texts = []

for text in cleaned_texts:

    if len(text) < min_length:
        continue

    if len(text) > max_length:
        continue

    filtered_texts.append(text)


print("Length Filtering 后:", len(filtered_texts))


# ============================================================
# 4. Exact Deduplication
# ============================================================

seen = set()
unique_texts = []

for text in filtered_texts:

    if text in seen:
        continue

    seen.add(text)
    unique_texts.append(text)


print("Exact Dedup 后:", len(unique_texts))

print()

for text in unique_texts:
    print(repr(text))


# ============================================================
# 5. Shuffle
# ============================================================

random.seed(42)

random.shuffle(unique_texts)

print()
print("Shuffle 后:")

for text in unique_texts:
    print(repr(text))


# ============================================================
# 6. Train / Validation / Test Split
# ============================================================

n = len(unique_texts)

train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_texts = unique_texts[:train_end]
val_texts = unique_texts[train_end:val_end]
test_texts = unique_texts[val_end:]


print()
print("Train:", len(train_texts))
print("Validation:", len(val_texts))
print("Test:", len(test_texts))


# ============================================================
# 7. Padding Demo
# ============================================================

print("\n========== Padding Demo ==========")

sequences = [
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25, 26, 27, 28],
    [31, 32, 33],
]

pad_id = 0

max_len = max(len(seq) for seq in sequences)

print("max_len:", max_len)


padded_sequences = []

for seq in sequences:

    padding_length = max_len - len(seq)

    padded_seq = seq + [pad_id] * padding_length

    padded_sequences.append(padded_seq)


batch = torch.tensor(padded_sequences)

print()
print("batch:")
print(batch)

print("batch shape:", batch.shape)


# ============================================================
# 8. Padding Mask
# ============================================================

padding_mask = (batch != pad_id)

print()
print("padding mask:")
print(padding_mask)

print("padding mask shape:", padding_mask.shape)


# ============================================================
# 9. Padding Ratio
# ============================================================

valid_tokens = padding_mask.sum().item()

total_tokens = padding_mask.numel()

padding_tokens = total_tokens - valid_tokens

padding_ratio = padding_tokens / total_tokens


print()
print("valid tokens:", valid_tokens)
print("total tokens:", total_tokens)
print("padding tokens:", padding_tokens)
print("padding ratio:", f"{padding_ratio:.2%}")


# ============================================================
# 10. Causal LM Shift + Loss Mask
# ============================================================

print("\n========== Causal LM Shift + Loss Mask ==========")

# 去掉最后一个 token
input_ids = batch[:, :-1]

# 去掉第一个 token
labels = batch[:, 1:].clone()

# labels 对应位置的 Padding Mask
label_mask = padding_mask[:, 1:]

# PAD 对应的 label 改成 -100
# CrossEntropyLoss 默认 ignore_index=-100
labels[~label_mask] = -100


print("input_ids:")
print(input_ids)

print()

print("labels:")
print(labels)


# ============================================================
# 11. Whole-sequence Packing
# ============================================================

print("\n========== Whole-sequence Packing ==========")

sequences = [
    [11, 12, 13, 14],
    [21, 22, 23],
    [31, 32],
    [41, 42, 43, 44, 45],
]

max_seq_len = 10
pad_id = 0

packed_sequences = []

current = []


for seq in sequences:

    # 当前 block 还能完整放下该 sequence
    if len(current) + len(seq) <= max_seq_len:

        current.extend(seq)

    else:

        # 当前 packed sequence 封包
        packed_sequences.append(current)

        # 新建一个 block
        current = list(seq)


# 最后一条不要忘记
if current:
    packed_sequences.append(current)


print("packing before padding:")

for seq in packed_sequences:
    print(seq)


# ============================================================
# 12. Padding Packed Sequences
# ============================================================

padded_packed_sequences = []

for seq in packed_sequences:

    padding_length = max_seq_len - len(seq)

    padded_seq = seq + [pad_id] * padding_length

    padded_packed_sequences.append(padded_seq)


packed_batch = torch.tensor(padded_packed_sequences)


print()
print("packing after padding:")
print(packed_batch)

print("packed batch shape:", packed_batch.shape)


# ============================================================
# 13. Packing Padding Ratio
# ============================================================

packed_mask = (packed_batch != pad_id)

packed_valid_tokens = packed_mask.sum().item()

packed_total_tokens = packed_mask.numel()

packed_padding_tokens = (
    packed_total_tokens - packed_valid_tokens
)

packed_padding_ratio = (
    packed_padding_tokens / packed_total_tokens
)


print()
print("packed valid tokens:", packed_valid_tokens)
print("packed total tokens:", packed_total_tokens)
print("packed padding tokens:", packed_padding_tokens)
print(
    "packed padding ratio:",
    f"{packed_padding_ratio:.2%}"
)


# ============================================================
# 14. Stream Packing with Toy EOS
# ============================================================

print("\n========== Stream Packing with EOS ==========")

sequences = [
    [11, 12, 13, 14, 15, 16],
    [21, 22, 23, 24, 25, 26, 27],
]

max_seq_len = 10

# 教学用假的 EOS
eos_id = 99

token_stream = []


for seq in sequences:

    token_stream.extend(seq)

    # 每个 sequence 结束插入 EOS
    token_stream.append(eos_id)


print("token stream:")
print(token_stream)


blocks = []

for i in range(
    0,
    len(token_stream),
    max_seq_len
):

    block = token_stream[
        i:i + max_seq_len
    ]

    blocks.append(block)


print("blocks:")

for block in blocks:
    print(block)


# ============================================================
# 15. Real Tokenization
# ============================================================

print("\n========== Real Tokenization ==========")

token_sequences = []


for text in train_texts:

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    token_sequences.append(token_ids)

    print("text:", repr(text))
    print("token ids:", token_ids)
    print("length:", len(token_ids))
    print()


# ============================================================
# 16. Real EOS
# ============================================================

real_eos_id = tokenizer.eos_token_id

print("EOS token:", tokenizer.eos_token)
print("EOS id:", real_eos_id)


# ============================================================
# 17. Real Token Stream
# ============================================================

real_token_stream = []


for token_ids in token_sequences:

    real_token_stream.extend(token_ids)

    # 每个文档后加入真实 EOS
    real_token_stream.append(real_eos_id)


print()
print("real token stream:")
print(real_token_stream)

print(
    "total tokens:",
    len(real_token_stream)
)


# ============================================================
# 18. Chunk Real Token Stream
# ============================================================

max_seq_len = 10

real_blocks = []


for i in range(
    0,
    len(real_token_stream),
    max_seq_len
):

    block = real_token_stream[
        i:i + max_seq_len
    ]

    real_blocks.append(block)


print()
print("real blocks:")


for i, block in enumerate(real_blocks):

    print(f"block {i}:")
    print(block)
    print("length:", len(block))


# ============================================================
# 19. Full Blocks Only
# ============================================================

full_blocks = []

for block in real_blocks:

    if len(block) == max_seq_len:
        full_blocks.append(block)


print()
print("full blocks only:")

for i, block in enumerate(full_blocks):

    print(
        f"block {i}:",
        block
    )

print(
    "number of full blocks:",
    len(full_blocks)
)