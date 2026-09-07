import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import datasets, transforms

from model import MLP


DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pt"


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    model = MLP().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.eval()

    transform = transforms.ToTensor()

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=True,
    )

    image, label = test_dataset[0]

    # 模型需要 batch 维
    image = image.unsqueeze(0)

    # 同样要把图片放到模型所在 device
    image = image.to(device)

    with torch.no_grad():
        logits = model(image)
        prediction = torch.argmax(logits, dim=1)

    print("真实标签:", label)
    print("预测标签:", prediction.item())

    # matplotlib 显示时转回 CPU
    image = image.cpu()

    plt.imshow(
        image.squeeze(),
        cmap="gray",
    )

    plt.title(
        f"true: {label}, pred: {prediction.item()}"
    )

    plt.show()


if __name__ == "__main__":
    main()
