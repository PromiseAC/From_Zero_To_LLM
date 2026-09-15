import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
# ============================================================
# 0. 导入第二周写的 MiniTransformer
# ============================================================

transformer_dir = (
    Path(__file__).resolve().parent.parent
    / "Learn_transformer"
)

sys.path.insert(0, str(transformer_dir))

from mini_transformer import MiniTransformer

# tokens = [1, 2, 3, 4, 7]

# input_ids = torch.tensor(tokens[:-1])
# labels = torch.tensor(tokens[1:])

# input_ids = input_ids.unsqueeze(0)
# labels = labels.unsqueeze(0)


sequences = [
    [1, 2, 3, 4, 7],
    [1, 2, 6, 4, 7],
    [5, 2, 3, 4, 7],
    [5, 2, 6, 4, 7],
]


class TinyLMDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.sequences[idx], dtype=torch.long)

        input_ids = tokens[:-1]
        labels = tokens[1:]

        return input_ids, labels

dataset = TinyLMDataset(sequences)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

vocab_size = 8
d_model = 32
num_heads = 4
num_layers = 2
intermediate_size = 128

model = MiniTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    intermediate_size=intermediate_size,
)

for input_ids, labels in dataloader:
    logits = model(input_ids)

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)
    print("logits shape:", logits.shape)

    break

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-3,
)

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0

    for input_ids, labels in dataloader:

        optimizer.zero_grad()

        logits = model(input_ids)

        B, T, V = logits.shape

        logits_flat = logits.reshape(B * T, V)
        labels_flat = labels.reshape(B * T)

        loss = criterion(logits_flat, labels_flat)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    #len(dataloader) 是指batch的个数
    avg_loss = total_loss / len(dataloader)

    if epoch % 10 == 0:
        print(
            f"epoch={epoch:3d} "
            f"loss={avg_loss:.4f}"
        )


