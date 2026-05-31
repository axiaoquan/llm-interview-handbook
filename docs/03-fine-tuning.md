# 03 · Fine-tuning 微调

SFT / PEFT / 指令微调相关。

## 📑 本章目录

- [Q01 · LoRA 原理](#q01--lora-原理)
- [Q02 · QLoRA / 4-bit 量化训练](#q02--qlora)
- [Q03 · Adapter / Prefix / P-tuning](#q03--其他-peft-方法)
- [Q04 · 指令微调（Instruction Tuning）](#q04--指令微调)
- [Q05 · 灾难性遗忘 / 过拟合处理](#q05--常见微调问题)
- [Q06 · 全量微调 vs LoRA 选型](#q06--全量微调-vs-lora-选型)

---

## Q01 · LoRA 原理

### 🎯 核心思想

**冻结预训练权重**，只训练**低秩分解矩阵**：

$$
W' = W + \Delta W = W + BA
$$

- $W \in \mathbb{R}^{d \times d}$：原始权重（**冻结**）
- $B \in \mathbb{R}^{d \times r}$：低秩矩阵（**可训练**）
- $A \in \mathbb{R}^{r \times d}$：低秩矩阵（**可训练**）
- $r \ll d$：秩，通常 4~64

### 📊 参数量对比

- 全量微调：$d \times d$
- LoRA：$2dr$
- 当 $r=16, d=4096$ 时，参数量比 **~0.78%**

### 📖 应用位置

通常作用在 **Attention 的 Q, K, V, O 投影矩阵**上。
实验：**同时对 Q 和 V 做 LoRA 效果最佳**。

### 📖 AB 矩阵的初始化（重要！）

| 矩阵 | 初始化 | 原因 |
|---|---|---|
| **降维矩阵 A** | 高斯随机 | 打破对称性。如果 A、B 都为 0，反向传播梯度为 0，无法学习 |
| **升维矩阵 B** | **全 0** | 保证初始 $\Delta W = 0$，模型初始等价于预训练，避免引入随机噪声 |

### 🛠 实现要点

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=32):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)  # 冻结
        for p in self.W.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)  # 高斯
        self.B = nn.Parameter(torch.zeros(out_dim, r))         # 全 0
        self.scaling = alpha / r

    def forward(self, x):
        return self.W(x) + (x @ self.A.T @ self.B.T) * self.scaling
```

### 🪤 面试常见追问

- **Q：scaling 系数 alpha/r 是干嘛的？**
  A：让 LoRA 输出量级独立于 r，调整 r 时不用重新调 lr。常见 alpha=2r。

- **Q：LoRA 推理时增加延迟吗？**
  A：**不增加**——可以把 $BA$ 融合进 $W$（$W' = W + BA$），变成一个矩阵。这是 LoRA 比 Adapter 优秀的关键。

- **Q：LoRA 能学到全量微调的效果吗？**
  A：在小数据集上接近全量；大规模继续预训练（CPT）类任务效果不如全量。

---

## Q02 · QLoRA

### 🎯 一句话

QLoRA = **基础模型量化到 4-bit (NF4)** + 在量化模型上做 LoRA 微调。

### 📖 三大创新

1. **NF4**（Normal Float 4）：针对正态分布权重设计的 4-bit 数据类型
2. **Double Quantization**：连量化常数也量化，进一步压缩
3. **Paged Optimizers**：用 NVIDIA 统一内存管理优化器状态

### 📊 显存对比（7B 模型，单卡）

| 方案 | 显存 |
|---|---|
| 全量微调 | ~80 GB |
| LoRA | ~30 GB |
| **QLoRA** | **~10 GB**（24G 卡可训） |

### 🪤 追问

- **Q：QLoRA 推理时怎么用？**
  A：要么把 LoRA 合并回去（用 fp16 模型）；要么保持 4-bit + LoRA 的组合（dequantize on-the-fly）。

---

## Q03 · 其他 PEFT 方法

### 📖 Adapter

在 Transformer 中插入小型 Adapter 模块：
$$
\text{Attention} \to \text{Adapter}(\downarrow \to \text{NonLinear} \to \uparrow) \to \text{FFN} \to \text{Adapter}(\downarrow \to \text{NonLinear} \to \uparrow)
$$

**缺点**：增加推理延迟（不能合并到原权重）。

### 📖 Prefix Tuning

在每层的 K/V 前面添加可训练的 prefix：
$$
K' = [P_K; K], \quad V' = [P_V; V]
$$

### 📖 Prompt Tuning

在输入 embedding 前添加可训练的 soft prompt：
$$
E' = [P; E]
$$

### 📊 PEFT 方法对比

| 方法 | 参数量 | 推理延迟 | 效果 | 推荐 |
|---|---|---|---|---|
| **LoRA** | 0.1-1% | 无增加 | ⭐⭐⭐⭐ | ✅ |
| **QLoRA** | 0.1-1% | 量化损失 | ⭐⭐⭐ | ✅ |
| Adapter | 1-5% | 有增加 | ⭐⭐⭐ | |
| Prefix | 0.1% | 有增加 | ⭐⭐ | |
| Prompt | <0.1% | 无增加 | ⭐⭐ | |

### 🪤 追问

- **Q：DoRA 是什么？**
  A：Decomposed LoRA，把 $W$ 分解为 magnitude + direction，对 direction 做 LoRA。在小 r 下效果比 LoRA 好。

---

## Q04 · 指令微调

### 🎯 一句话

指令微调属于有监督微调（SFT），让模型**学会对话格式**和**遵循指令**。

### 📖 损失函数

$$
\mathcal{L} = -\sum_{t \in \text{output}} \log P(x_t \mid x_{<t})
$$

只在 **assistant 回复部分**算损失（不算 user 输入和 system prompt）。

### 📖 数据：质量 > 数量

| 来源 | 优点 | 缺点 |
|---|---|---|
| **人工标注** | 质量最高 | 成本高 |
| **GPT 蒸馏** | 量大便宜 | 法律风险 |
| **开源数据集** | 免费 | 质量参差 |
| **Self-Instruct** | 自动化 | 需要种子数据 |

### 📖 对话格式

- **方式一**：每轮独立计算损失
- **方式二**：拼接多轮，**只在 assistant 部分计算损失**（推荐）

```
<|system|> You are a helpful assistant.
<|user|> 你好
<|assistant|> 你好！有什么可以帮你的？  ← 只在这部分算 loss
<|user|> 介绍一下 Transformer
<|assistant|> Transformer 是...        ← 只在这部分算 loss
```

### 🪤 追问

- **Q：SFT 数据多少够？**
  A：LIMA 论文证明 1000 条高质量数据就能让 LLaMA 出对话能力，**质量远比数量重要**。

---

## Q05 · 常见微调问题

### 1) 灾难性遗忘

**症状**：微调后专项任务好了，但通用能力（数学/代码/常识）下降。

**解决**：
- 使用**较小学习率**（1e-5 ~ 5e-5）
- 混入少量预训练数据（**Replay**）
- 使用 **LoRA**（天然缓解，原参数被冻结）

### 2) 过拟合

**症状**：训练 loss 降，验证 loss 升。

**解决**：
- 数据增强
- **早停**（Early Stopping）
- 增大 dropout
- 减小训练 epoch（SFT 通常 1-3 epoch 就够）

---

## Q06 · 全量微调 vs LoRA 选型

| 场景 | 推荐方法 |
|---|---|
| 数据量大（>100K），资源充足 | 全量微调 |
| 数据量中（1K-100K） | LoRA r=16-64 |
| 数据量少（<1K） | LoRA r=4-8 |
| 多任务 / 多租户 | LoRA（可热切换） |
| 显存受限 | QLoRA |
| 大幅修改风格 / 注入新知识 | 全量微调 |
| 注入领域指令格式 | LoRA 即可 |

### 🪤 追问

- **Q：LoRA 多任务怎么部署？**
  A：保留一个基座 + 多个 LoRA adapter 文件，按需加载。一台机器可以同时服务多个领域的模型。

---

[⬅ 回到首页](../README.md)
