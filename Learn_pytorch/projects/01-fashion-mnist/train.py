import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP


DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pt"

BATCH_SIZE = 64
EPOCHS = 5


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print("保存最佳模型")

    epoch_numbers = range(1, EPOCHS + 1)

    plt.plot(epoch_numbers, train_losses, label="train loss")
    plt.plot(epoch_numbers, val_losses, label="val loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(epoch_numbers)
    plt.legend()
    plt.show()

    plt.plot(epoch_numbers, val_accuracies)

    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.xticks(epoch_numbers)
    plt.show()

    print(f"best val loss: {best_val_loss:.4f}")
    print("模型保存位置:", MODEL_PATH)


if __name__ == "__main__":
    main()
