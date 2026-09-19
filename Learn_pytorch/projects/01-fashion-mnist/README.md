# FashionMNIST 图像分类实践

一个基于 PyTorch 实现的 FashionMNIST 图像分类项目。

本项目使用一个简单的多层感知机（MLP）完成 FashionMNIST 十分类任务，包含完整的训练、验证、模型保存、模型加载和单张图片推理流程。

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python predict.py
```

`train.py` 会自动下载 FashionMNIST 数据集，并在项目目录生成最佳模型权重 `model.pt`。完成训练后再运行 `predict.py`。

---

## Project Structure

```text
fashion_mnist/
├── .gitignore
├── model.py
├── train.py
├── predict.py
├── requirements.txt
├── model.pt                  # 训练后生成，不提交
├── data/                     # 自动下载，不提交
└── README.md
```

各文件作用：

- `model.py`：定义 MLP 模型结构
- `train.py`：训练模型、验证模型、保存最佳模型
- `predict.py`：加载训练好的模型并进行单张图片预测
- `requirements.txt`：项目依赖及版本
- `model.pt`：训练后生成，保存验证集表现最好的模型参数
- `data/`：运行时自动下载的 FashionMNIST 数据集
- `.gitignore`：排除数据、缓存和模型权重等生成文件
- `README.md`：项目说明文档

---

## Dataset

本项目使用 FashionMNIST 数据集。

FashionMNIST 是一个服装图片分类数据集，每张图片为：

```text
1 × 28 × 28
```

其中：

```text
1   -> 灰度通道
28  -> 图片高度
28  -> 图片宽度
```

FashionMNIST 一共有 10 个类别：

```text
0 -> T-shirt/top
1 -> Trouser
2 -> Pullover
3 -> Dress
4 -> Coat
5 -> Sandal
6 -> Shirt
7 -> Sneaker
8 -> Bag
9 -> Ankle boot
```

---

## Model

本项目使用一个简单的 MLP：

```text
输入图片
[Batch, 1, 28, 28]

        ↓

Flatten

        ↓

[Batch, 784]

        ↓

Linear(784, 256)

        ↓

ReLU

        ↓

Linear(256, 10)

        ↓

[Batch, 10] logits
```

对应 PyTorch 代码：

```python
import torch.nn as nn


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

---

## Training Configuration

训练配置如下：

```text
Dataset       : FashionMNIST
Batch Size    : 64
Epochs        : 5
Loss Function : CrossEntropyLoss
Optimizer     : AdamW
Learning Rate : 0.001
Weight Decay  : 0.01
Random Seed   : 42
```

优化器：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)
```

损失函数：

```python
criterion = nn.CrossEntropyLoss()
```

---

## Device

程序优先使用 Apple Silicon 的 MPS 加速。

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

模型需要移动到对应设备：

```python
model = MLP().to(device)
```

训练和验证过程中，数据也需要移动到相同设备：

```python
images = images.to(device)
labels = labels.to(device)
```

即：

```text
model  -> mps
images -> mps
labels -> mps
```

模型和数据必须位于相同设备。

---

## Training Process

一个完整的训练 batch 包含以下步骤：

```text
读取一个 batch
      ↓
前向传播
      ↓
计算 loss
      ↓
清空旧梯度
      ↓
反向传播
      ↓
计算梯度
      ↓
更新参数
```

对应代码：

```python
optimizer.zero_grad()

logits = model(images)

loss = criterion(logits, labels)

loss.backward()

optimizer.step()
```

其中：

- `optimizer.zero_grad()`：清空上一轮保存在参数中的梯度
- `loss.backward()`：通过自动微分计算梯度
- `optimizer.step()`：根据梯度更新模型参数

---

## Training Mode

训练前使用：

```python
model.train()
```

将模型切换到训练模式。

完整的单轮训练函数：

```python
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
```

---

## Evaluation

验证阶段不需要更新模型参数。

首先：

```python
model.eval()
```

将模型切换到评估模式。

然后：

```python
with torch.no_grad():
```

关闭 autograd 的计算图记录，从而降低内存开销并减少不必要计算。

验证流程：

```python
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
```

准确率计算：

```text
accuracy = 预测正确的样本数量 / 总样本数量
```

---

## Save Best Model

训练过程中，不直接保存最后一个 epoch 的模型，而是根据验证集的 `val_loss` 保存当前表现最好的模型。

首先：

```python
best_val_loss = float("inf")
```

每轮验证结束后：

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )
```

逻辑如下：

```text
当前 val_loss
      ↓
是否小于 best_val_loss？
      ↓
   是       否
   ↓         ↓
保存模型    不保存
```

这样最终的 `model.pt` 保存的是验证集表现最好的模型，而不是最后一个 epoch 的模型。

---

## state_dict

PyTorch 中：

```python
model.state_dict()
```

主要保存模型参数，可以理解为：

```text
参数名称 -> Tensor
```

例如：

```text
fc1.weight -> Tensor
fc1.bias   -> Tensor
fc2.weight -> Tensor
fc2.bias   -> Tensor
```

保存：

```python
torch.save(
    model.state_dict(),
    "model.pt"
)
```

---

## Load Model

重新加载模型时，需要先重新创建相同的模型结构：

```python
model = MLP()
```

然后读取保存的参数：

```python
state_dict = torch.load(
    "model.pt",
    map_location=device,
    weights_only=True,
)
```

最后将参数加载到模型中：

```python
model.load_state_dict(state_dict)
```

完整流程：

```text
创建模型结构
      ↓
model = MLP()

读取参数文件
      ↓
torch.load(..., weights_only=True)

加载参数
      ↓
load_state_dict()

切换评估模式
      ↓
model.eval()
```

---

## Inference

从 FashionMNIST 测试集中读取一张图片：

```python
image, label = test_dataset[0]
```

此时图片 shape 为：

```text
[1, 28, 28]
```

对应：

```text
[C, H, W]
```

由于模型输入需要 batch 维，因此增加一个维度：

```python
image = image.unsqueeze(0)
```

shape 变化：

```text
[1, 28, 28]

      ↓ unsqueeze(0)

[1, 1, 28, 28]
```

对应：

```text
[B, C, H, W]
```

其中：

```text
B = 1
```

表示当前 batch 中只有一张图片。

推理代码：

```python
with torch.no_grad():
    logits = model(image)
    prediction = torch.argmax(logits, dim=1)
```

输出：

```text
logits shape = [1, 10]
```

表示：

```text
1 个样本
10 个类别
```

通过：

```python
torch.argmax(logits, dim=1)
```

选择 logits 最大的类别作为最终预测结果。

---

## Run Training

进入项目目录：

```bash
cd fashion_mnist
```

运行：

```bash
python train.py
```

训练过程中会输出类似：

```text
device: mps

epoch 1: train_loss=0.5174, val_loss=0.4412, val_acc=0.8398
保存最佳模型

epoch 2: train_loss=0.3821, val_loss=0.3815, val_acc=0.8627
保存最佳模型

epoch 3: train_loss=0.3396, val_loss=0.3668, val_acc=0.8682
保存最佳模型

epoch 4: train_loss=0.3163, val_loss=0.3442, val_acc=0.8757
保存最佳模型

epoch 5: train_loss=0.2950, val_loss=0.3539, val_acc=0.8732

best val loss: 0.3442
```

---

## Run Prediction

运行：

```bash
python predict.py
```

输出示例：

```text
device: mps

真实标签: 9
预测标签: 9
```

表示该样本预测正确。

---

## Training Result

本次训练结果：

```text
Best Validation Loss     : 0.3442
Best Validation Accuracy : 87.57%
Best Epoch               : 4
```

第 5 个 epoch：

```text
train_loss = 0.2950
val_loss   = 0.3539
val_acc    = 0.8732
```

可以观察到：

```text
train loss ↓

但

val loss ↑
val accuracy ↓
```

说明从第 4 个 epoch 到第 5 个 epoch，训练集表现继续改善，但验证集表现出现轻微下降。

这可能是开始出现过拟合的信号，但单独一轮的小幅波动还不能直接确定模型已经明显过拟合。

---

## Underfitting and Overfitting

### 正常训练

```text
train loss ↓
val loss   ↓
val acc    ↑
```

说明模型在训练集和验证集上的表现都在改善。

### 欠拟合

典型表现：

```text
train loss 较高
val loss   较高

train accuracy 较低
val accuracy   较低
```

说明模型连训练数据本身都没有充分学习。

可能原因包括：

- 模型容量太小
- 训练 epoch 太少
- 学习率设置不合适
- 特征表达能力不足

### 过拟合

典型表现：

```text
train loss 持续下降
val loss   开始持续上升
```

或者：

```text
train accuracy 持续提高
val accuracy 开始下降
```

说明模型越来越适应训练数据，但泛化能力开始下降。

需要关注的是训练曲线和验证曲线的整体趋势，而不是某一个 epoch 的小幅波动。

---

## What I Learned

通过本项目完成了以下 PyTorch 基础内容：

- Tensor 与常见 shape
- Broadcasting
- 矩阵乘法
- `nn.Module`
- `forward`
- `nn.Linear`
- ReLU
- Logits
- CrossEntropyLoss
- Autograd
- `loss.backward()`
- `optimizer.step()`
- Dataset
- DataLoader
- Batch / Iteration / Epoch
- SGD
- AdamW
- Weight Decay
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- MPS / CPU device
- `state_dict`
- 模型保存与加载
- 单张图片推理
- 训练 loss 与验证 loss
- 欠拟合与过拟合
- 保存最佳模型
- 基础多文件 PyTorch 项目结构

---

## Future Improvements

后续可以继续尝试：

- 增加训练 epoch
- 使用独立 validation set
- 尝试 CNN
- 对比 SGD 与 AdamW
- 调整 learning rate
- 调整 hidden dimension
- 添加 Dropout
- 添加更完整的模型评估指标
- 保存训练曲线到图片文件
- 加入梯度裁剪
- 学习梯度累积
