# From Zero to LLM

从零开始理解神经网络，并最终亲手构建一个 LLM。

本仓库记录 [Andrej Karpathy 的 Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 课程学习过程。笔记不追求逐字复述视频，而是围绕「核心问题 → 原理 → 实现 → 易错点 → 复习与练习」整理，方便回顾和继续扩展。

## 学习进度

| # | 主题 | 核心内容 | 状态 | 笔记 |
|---:|---|---|:---:|---|
| 01 | micrograd | 导数、计算图、反向传播、MLP、梯度下降 | ✅ 已学习 | [查看笔记](notes/01-micrograd/README.md) |
| 02 | makemore：Bigram | 字符级语言模型、计数、损失函数 | ⬜ 待学习 | — |
| 03 | makemore：MLP | Embedding、MLP、数据集划分、超参数 | ⬜ 待学习 | — |
| 04 | 激活与梯度 | 初始化、激活分布、BatchNorm | ⬜ 待学习 | — |
| 05 | Backprop Ninja | 手动反向传播、交叉熵、BatchNorm | ⬜ 待学习 | — |
| 06 | WaveNet | 层次化网络、卷积思想 | ⬜ 待学习 | — |
| 07 | GPT | Self-Attention、Transformer、GPT | ⬜ 待学习 | — |
| 08 | Tokenizer | Token、BPE、编码与解码 | ⬜ 待学习 | — |

> 课程仍在更新，以上顺序以课程官网为准。

## 仓库结构

```text
.
├── README.md                    # 总览、路线与进度
├── notes/
│   └── 01-micrograd/
│       └── README.md            # 第一课完整笔记
└── templates/
    └── lesson-note-template.md  # 后续课程笔记模板
```

后续每节课使用独立目录：`notes/<序号>-<主题>/`。笔记正文固定命名为 `README.md`，配套代码、图片和练习分别放入该课目录下的 `code/`、`assets/` 和 `exercises/`（需要时再创建）。这样 GitHub 打开目录即可阅读，也不会让仓库根目录越来越乱。

## 笔记约定

- 用自己的话记录结论，同时保留「为什么」。
- 代码只保留能说明概念的最小示例，完整实验放进对应课程目录。
- 明确区分课程内容、自己的理解和待验证问题。
- 每课结束补充易错点、复习问题与练习，避免笔记变成一次性摘抄。
- 公式统一使用 LaTeX，代码和变量名使用英文，解释使用中文。

新建笔记时复制 [`templates/lesson-note-template.md`](templates/lesson-note-template.md)，并同步更新上方进度表。

## 参考资料

- [Neural Networks: Zero to Hero 课程主页](https://karpathy.ai/zero-to-hero.html)
- [Andrej Karpathy 的 micrograd](https://github.com/karpathy/micrograd)
