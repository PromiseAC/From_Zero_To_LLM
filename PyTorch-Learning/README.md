# PyTorch Learning

PyTorch 学习笔记与独立实践项目。这里记录从张量基础、模型训练到推理部署的学习过程，与其他课程笔记保持并列、互不耦合。

## 内容索引

### 实践项目

| # | 项目 | 核心内容 | 状态 |
|---:|---|---|:---:|
| 01 | [FashionMNIST 图像分类](projects/01-fashion-mnist/README.md) | MLP、训练与验证、MPS、模型保存与推理 | ✅ 完成 |

### 学习笔记

后续概念笔记统一放在 `notes/` 下，并在这里补充索引。

## 目录约定

```text
PyTorch-Learning/
├── README.md
├── notes/                       # 概念与 API 笔记
└── projects/
    └── 01-fashion-mnist/        # 可独立运行的实践项目
```

- 每个项目都使用独立目录，并包含自己的 `README.md` 和依赖文件。
- 数据集、缓存、模型权重等可再生成文件不提交到 Git。
- 项目编号按完成顺序递增，名称使用小写英文和连字符。
- 顶层 README 只负责导航，具体原理、实验结果和运行方式放在项目内部。

