# PyTorch Learning

本目录用于记录 PyTorch 入门到完整训练流程的学习过程。内容按照“先理解张量，再理解梯度与优化，最后完成数据集分类”的顺序组织，便于后续继续添加实验。

本目录目前覆盖：

* Tensor 的创建、形状、索引、变形与矩阵运算
* `Dataset`、`DataLoader` 与 batch 数据
* `nn.Module`、线性层与 MLP
* 损失函数、反向传播、梯度和优化器
* 线性回归与 FashionMNIST 图像分类
* 模型保存、加载、验证和单张图片推理

---

## 1. 目录结构

```text
Learn_pytorch/
├── tensor_practice.py
├── 1.py
├── linear_regression.py
├── linear_practice.py
├── autograd_practice.py
├── train_one_epoch.py
├── evaluate_one_epoch.py
├── fashionmnist_dataloader.py
├── min_fashionmnist_dataset.py
├── MLP_practice.py
├── a_simple_complete_train .py
├── load_and_predict.py
├── fashion_mnist/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── README.md
└── README.md
```

`__pycache__/`、`data/`、`model.pt` 等缓存、数据集和模型权重属于运行时生成文件，不应提交到知识库。

---

## 2. 学习路线

```text
Tensor 基础
    ↓
Dataset 与 DataLoader
    ↓
线性层与前向传播
    ↓
Loss、反向传播与梯度
    ↓
优化器更新参数
    ↓
训练/验证循环
    ↓
FashionMNIST MLP 分类
    ↓
保存模型并进行推理
```

---

## 3. Tensor 基础

`tensor_practice.py` 用小规模示例熟悉 PyTorch Tensor：

```python
import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
])
```

重点操作包括：

```text
shape / dtype / device
索引与切片
reshape
unsqueeze / squeeze
逐元素运算
矩阵乘法 @
```

张量形状是后续模型调试的基础。比如给单张图片增加 batch 维度：

```python
image = image.unsqueeze(0)
```

形状会从 `[C, H, W]` 变为 `[B, C, H, W]`。

---

## 4. Dataset 与 DataLoader

### 4.1 TensorDataset

`1.py` 使用 `TensorDataset` 和 `DataLoader` 创建一个小型数据集：

```python
dataset = TensorDataset(x, labels)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)
```

其中：

```text
Dataset       负责保存样本与标签
DataLoader    负责按 batch 读取数据
batch_size    每个 batch 的样本数
shuffle       每轮训练前是否打乱数据
```

### 4.2 FashionMNIST

`fashionmnist_dataloader.py` 使用 torchvision 下载并读取 FashionMNIST：

```python
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True,
)
```

一个 batch 的形状为：

```text
images.shape = [64, 1, 28, 28]
labels.shape = [64]
```

含义是：

```text
64   -> batch size
1    -> 灰度通道
28   -> 图片高度
28   -> 图片宽度
```

`min_fashionmnist_dataset.py` 进一步演示如何取出一张图片并用 matplotlib 显示。

---

## 5. 模型与前向传播

PyTorch 模型通常继承 `nn.Module`，在 `__init__` 中定义层，在 `forward` 中描述数据流：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

FashionMNIST 的数据流为：

```text
[B, 1, 28, 28]
      ↓ Flatten
[B, 784]
      ↓ Linear(784, 256)
[B, 256]
      ↓ ReLU
[B, 256]
      ↓ Linear(256, 10)
[B, 10]  logits
```

`MLP_practice.py` 只取一个 batch，专门检查数据和模型输出的形状。

---

## 6. Loss、反向传播与优化

`autograd_practice.py` 使用三分类线性模型演示完整的一步更新：

```python
model = nn.Linear(2, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
)
```

训练步骤：

```text
前向传播：logits = model(x)
计算损失：loss = criterion(logits, labels)
清空旧梯度：optimizer.zero_grad()
反向传播：loss.backward()
更新参数：optimizer.step()
```

`loss.backward()` 会根据计算图计算参数梯度，梯度保存在 `model.weight.grad` 和 `model.bias.grad`。

---

## 7. 线性回归

`linear_regression.py` 生成带噪声的数据：

```text
y = 3x + 2 + noise
```

并使用 `nn.Linear(1, 1)` 和均方误差损失拟合参数：

```python
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)
```

训练完成后，模型学习到的 weight 应接近 `3`，bias 应接近 `2`。`linear_practice.py` 是同一主题的更长训练版本，用于观察 loss 的变化。

---

## 8. 训练与验证循环

`train_one_epoch.py` 把一轮训练封装成函数：

```python
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
```

`evaluate_one_epoch.py` 演示验证阶段：

```python
model.eval()

with torch.no_grad():
    logits = model(images)
```

训练阶段需要 `model.train()` 和梯度；验证/推理阶段使用 `model.eval()` 与 `torch.no_grad()`，避免保存不必要的计算图。

---

## 9. FashionMNIST 完整实践

完整项目位于 `fashion_mnist/`，包括：

```text
fashion_mnist/
├── model.py       # MLP 模型结构
├── train.py       # 训练、验证和保存最佳权重
├── predict.py     # 加载权重并预测单张图片
└── README.md      # 项目说明与实验记录
```

### 9.1 训练流程

`a_simple_complete_train .py` 是将数据、模型、损失、优化器和训练循环组合在一起的完整示例；`fashion_mnist/train.py` 是整理后的项目版本。

训练核心配置：

```text
模型：MLP
损失：CrossEntropyLoss
优化器：AdamW
batch size：64
epoch：5
设备：优先使用 Apple MPS，否则使用 CPU
```

验证集 loss 变小时保存：

```python
torch.save(
    model.state_dict(),
    MODEL_PATH,
)
```

### 9.2 加载模型与推理

重新推理时必须先创建相同的模型结构，再加载参数：

```python
model = MLP()
state_dict = torch.load(
    "model.pt",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()
```

单张图片需要增加 batch 维度：

```text
[1, 28, 28]
      ↓ unsqueeze(0)
[1, 1, 28, 28]
```

模型输出 `logits` 的形状是 `[1, 10]`，使用最大 logit 对应的类别作为预测结果：

```python
prediction = torch.argmax(logits, dim=1)
```

更完整的运行方式和实验结果见 [`fashion_mnist/README.md`](fashion_mnist/README.md)。

---

## 10. 运行示例

在本目录下执行：

```bash
# 激活已有虚拟环境
source ../.venv/bin/activate

# Tensor 与自动求导基础
python tensor_practice.py
python autograd_practice.py

# Dataset 与 DataLoader
python fashionmnist_dataloader.py
python MLP_practice.py

# 线性回归
python linear_regression.py

# FashionMNIST 完整项目
cd fashion_mnist
python train.py
python predict.py
```

首次运行 FashionMNIST 脚本会下载数据到 `data/`，训练后会生成 `model.pt`。这些文件仅用于本地运行，不纳入知识库提交。

---

## 11. 与 Transformer 学习的衔接

PyTorch 基础为后续手写 Transformer 提供必要的工程与张量能力：

```text
Tensor shape 与矩阵乘法
        ↓
Linear / Embedding
        ↓
Batch 维度与序列维度
        ↓
Attention 中的 Q、K、V
        ↓
Transformer Block
```

Transformer 的独立实现放在同级的 `Learn_transformer/` 目录，两个目录分别记录基础框架和模型结构学习。
