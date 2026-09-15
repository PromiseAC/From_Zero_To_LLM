import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

x = torch.randn(128, 10)
y = torch.randn(128, 1)

model = nn.Linear(10, 1)

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

total_steps = 100
warmup_steps = 10

def get_lr(step):
    if step < warmup_steps:
        return (step+1) / warmup_steps

    progress = (step-warmup_steps) / (total_steps-warmup_steps)

    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=get_lr
)


max_grad_norm = 0.5

lr_history = []

for step in range(total_steps):
    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm
    )
    was_clipped = grad_norm.item() > max_grad_norm

    current_lr = optimizer.param_groups[0]["lr"]
    lr_history.append(current_lr)
    optimizer.step()
    scheduler.step()

    print(
        f"step={step:3d} "
        f"loss={loss.item():.4f} "
        f"lr={current_lr:.6f} "
        f"grad_norm={grad_norm.item():.4f} "
        f"clipped={was_clipped}"
    )

plt.plot(range(total_steps), lr_history)

plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Warmup + Cosine Learning Rate")

plt.show()

