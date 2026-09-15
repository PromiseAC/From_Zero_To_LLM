import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import sys
from pathlib import Path

transformer_dir = (
    Path(__file__).resolve().parent.parent
    / "Learn_transformer"
)

sys.path.insert(0, str(transformer_dir))
from mini_transformer import MiniTransformer

seed = 42
torch.manual_seed(seed)

train_text = (
    "我喜欢学习语言模型。"
    "语言模型根据前文预测下一个字。"
    "注意力机制帮助模型理解上下文。"
    "训练模型需要数据和优化器。"
    "我们使用小模型进行预训练实验。"
) * 100

val_text = (
    "语言模型根据上下文预测下一个字。"
    "模型需要数据和注意力。"
) * 20

def clean_text(text):
    return "".join(text.strip().split())

train_text = clean_text(train_text)
val_text = clean_text(val_text)

chars = sorted(set(train_text))

stoi = {
    ch: idx
    for idx, ch in enumerate(chars)
}

unk_id = len(stoi)
stoi["<UNK>"] = unk_id

itos = {
    idx: ch
    for ch, idx in stoi.items()
}

vocab_size = len(stoi)


def encode(text):
    return [
        #找到ch返回对应的int，负责返回unk_id
        stoi.get(ch, unk_id)
        for ch in text
    ]

train_ids = encode(train_text)
val_ids = encode(val_text)

max_seq_len = 32

# chunk = train_ids[:max_seq_len + 1]

# input_ids = chunk[:-1]
# labels = chunk[1:]

class TextDataset(Dataset):
    def __init__(self, token_ids, max_seq_len):
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len

        self.num_sequences = (
            len(token_ids) - 1
        ) // max_seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1

        chunk = self.token_ids[start:end]

        input_ids = torch.tensor(
            chunk[:-1],
            dtype=torch.long
        )
        labels = torch.tensor(
            chunk[1:],
            dtype=torch.long
        )

        return input_ids, labels

#测试
train_dataset = TextDataset(
    train_ids,
    max_seq_len=max_seq_len
)

val_dataset = TextDataset(
    val_ids,
    max_seq_len=max_seq_len
)

micro_batch_size = 8

train_loader = DataLoader(
    train_dataset,
    batch_size = micro_batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=micro_batch_size,
    shuffle=False
)

input_ids, labels = next(iter(train_loader))

vocab_size = len(stoi)

d_model = 32
num_heads = 4
num_layers = 2
intermediate_size = 128

model = MiniTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    intermediate_size=intermediate_size
)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

num_epochs = 10
max_grad_norm = 1.0
total_steps = num_epochs * len(train_loader)
warmup_steps = 20

def lr_lambda(step):

    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = (
        step - warmup_steps
    ) / (
        total_steps - warmup_steps
    )

    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda
)

save_epochs = {0, 3, 9}
global_step = 0

checkpoint_dir = Path(__file__).resolve().parent

for epoch in range(num_epochs):
    model.train()

    total_loss = 0.0
    total_grad_norm = 0.0

    for input_ids, labels in train_loader:

        optimizer.zero_grad()

        logits = model(input_ids)

        B, T, V = logits.shape

        logits_flat = logits.reshape(B * T, V)
        labels_flat = labels.reshape(B * T)

        loss = criterion(
            logits_flat,
            labels_flat
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm
        )

        optimizer.step()
        scheduler.step()
        global_step += 1
        total_loss += loss.item()
        total_grad_norm += grad_norm.item()

    avg_loss = total_loss / len(train_loader)
    avg_grad_norm = (
        total_grad_norm / len(train_loader)
    )
    current_lr = optimizer.param_groups[0]["lr"]


    model.eval()

    val_total_loss = 0.0

    with torch.no_grad():

        for input_ids, labels in val_loader:

            logits = model(input_ids)

            B, T, V = logits.shape

            logits_flat = logits.reshape(B * T, V)
            labels_flat = labels.reshape(B * T)

            val_loss = criterion(
                logits_flat,
                labels_flat
            )

            val_total_loss += val_loss.item()

        val_avg_loss = val_total_loss / len(val_loader)

        print(
            f"epoch={epoch:2d} "
            f"train_loss={avg_loss:.4f} "
            f"val_loss={val_avg_loss:.4f} "
            f"grad_norm={avg_grad_norm:.4f} "
            f"lr={current_lr:.6f}"
        )

    if epoch in save_epochs: 
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),

            "epoch": epoch,
            "global_step": global_step,

            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "intermediate_size": intermediate_size,
            "max_seq_len": max_seq_len,

            "stoi": stoi,
            "itos": itos,
        }

        checkpoint_path = (
            checkpoint_dir
            / f"checkpoint_epoch_{epoch}.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print("checkpoint saved to:", checkpoint_path)



# loaded_checkpoint = torch.load(
#     checkpoint_path,
#     map_location="cpu"
#     )
# print(
#     "has tokenizer:",
#     "stoi" in loaded_checkpoint
#     and "itos" in loaded_checkpoint
# )

# print("stoi size:", len(loaded_checkpoint["stoi"]))
# print("itos size:", len(loaded_checkpoint["itos"]))
# print("checkpoint keys:")
# print(loaded_checkpoint.keys())
# print("epoch:", loaded_checkpoint["epoch"])
# print("global_step:", loaded_checkpoint["global_step"])
# print("vocab_size:", loaded_checkpoint["vocab_size"])
# print("max_seq_len:", loaded_checkpoint["max_seq_len"])

# model_state = loaded_checkpoint["model_state_dict"]
# print(type(model_state))
# print(model_state.keys())

# print(
#     "lm_head weight shape:",
#     model_state["lm_head.weight"].shape
# )


# new_model = MiniTransformer(
#     vocab_size=loaded_checkpoint["vocab_size"],
#     d_model=loaded_checkpoint["d_model"],
#     num_heads=loaded_checkpoint["num_heads"],
#     num_layers=loaded_checkpoint["num_layers"],
#     intermediate_size=loaded_checkpoint["intermediate_size"]
# )
# print(
#     "before loading:",
#     torch.equal(
#         model.lm_head.weight,
#         new_model.lm_head.weight
#     )
# )

# new_model.load_state_dict(
#     loaded_checkpoint["model_state_dict"]
# )
# print(
#     "after loading:",
#     torch.equal(
#         model.lm_head.weight,
#         new_model.lm_head.weight
#     )
# )