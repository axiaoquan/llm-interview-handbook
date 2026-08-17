# 03 · Fine-tuning 微调

SFT / PEFT / 指令微调相关。

## 本章目录

- [Q01 · LoRA 原理](#q01--lora-原理)
- [Q02 · QLoRA / 4-bit 量化训练](#q02--qlora)
- [Q03 · Adapter / Prefix / P-tuning](#q03--其他-peft-方法)
- [Q04 · 指令微调（Instruction Tuning）](#q04--指令微调)
- [Q05 · 灾难性遗忘 / 过拟合处理](#q05--常见微调问题)
- [Q06 · 全量微调 vs LoRA 选型](#q06--全量微调-vs-lora-选型)

---

## Q01 · LoRA 原理

### 核心思想

**冻结预训练权重**，只训练**低秩分解矩阵**：

$$
W' = W + \Delta W = W + BA
$$

- $W \in \mathbb{R}^{d \times d}$：原始权重（**冻结**）
- $B \in \mathbb{R}^{d \times r}$：低秩矩阵（**可训练**）
- $A \in \mathbb{R}^{r \times d}$：低秩矩阵（**可训练**）
- $r \ll d$：秩，通常 4~64

### 为什么低秩更新可能够用？

LoRA 并不是假设预训练权重 $W$ 本身低秩，而是假设**适配任务所需的更新 $\Delta W$ 具有较低的内在维度**。预训练模型已经拥有大部分通用表示，微调往往只需在少数方向上重组能力；用 $BA$ 把更新限制在秩至多为 $r$ 的子空间，相当于用结构性正则换取更少的参数和优化器状态。

$r$ 过小会形成表达瓶颈，过大则逐渐接近全量更新的成本并增加过拟合风险。是否足够取决于任务跨度、数据量、注入层和模型规模，不能只按数据条数机械选择。

### 参数量对比

- 全量微调：$d \times d$
- LoRA：$2dr$
- 当 $r=16, d=4096$ 时，参数量比 **~0.78%**

### 应用位置

通常作用在 Attention 的 Q/K/V/O 投影，也可作用在 FFN。只调 Q/V 是早期常见配置，不是普遍最优；需要更大能力迁移时，覆盖全部线性层通常更有表达力，但训练参数和显存也更高。

### AB 矩阵的初始化（重要！）

| 矩阵 | 初始化 | 原因 |
|---|---|---|
| **降维矩阵 A** | 高斯随机 | 打破对称性。如果 A、B 都为 0，反向传播梯度为 0，无法学习 |
| **升维矩阵 B** | **全 0** | 保证初始 $\Delta W = 0$，模型初始等价于预训练，避免引入随机噪声 |

### 实现要点

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

### 面试常见追问

- **Q：scaling 系数 $\alpha/r$ 是干嘛的？**
  A：它把“低秩分支容量”与“分支整体强度”分开，方便独立调节。不能严格说它保证对所有初始化都与 $r$ 无关；随着 rank 增大，梯度/更新范数仍可能变化。经典 LoRA 用 $\alpha/r$，rsLoRA 等变体使用 $\alpha/\sqrt r$ 来改善较大 rank 下的缩放稳定性。

- **Q：LoRA 推理时增加延迟吗？**
  A：若把 $BA$ 预先融合进 $W$，结构上不增加矩阵乘；若运行时保留可热切换 adapter，则仍有低秩分支计算和调度开销。

- **Q：LoRA 能学到全量微调的效果吗？**
  A：在小数据集上接近全量；大规模继续预训练（CPT）类任务效果不如全量。

---

## Q02 · QLoRA

### 一句话

QLoRA = **基础模型量化到 4-bit (NF4)** + 在量化模型上做 LoRA 微调。

### 三大创新

1. **NF4**（Normal Float 4）：针对正态分布权重设计的 4-bit 数据类型
2. **Double Quantization**：连量化常数也量化，进一步压缩
3. **Paged Optimizers**：用 NVIDIA 统一内存管理优化器状态

NF4 的设计动机不是“4-bit 比 INT4 神奇”，而是预训练权重经标准化后常近似钟形分布。均匀 INT4 把码点等距放置，会把许多码点浪费在低密度区域；NF4 依据正态分布分位数设置非均匀码点，使各区间承载的概率质量更均衡，从而降低平均量化误差。计算时仍需按 block 的 scale 反量化到计算 dtype；4-bit 主要节省权重存储，并不代表所有算子都以原生 4-bit 浮点执行。

### 显存对比（7B 模型，单卡）

| 方案 | 显存 |
|---|---|
| 全量微调 | 最高：参数、梯度和优化器状态都需维护 |
| LoRA | 冻结基座，仍需以较高精度存储基座权重 |
| **QLoRA** | 冻结基座再以 4-bit 存储，主要训练 LoRA 参数 |

具体 GB 数会随序列长度、batch、checkpointing、优化器和量化 group size 大幅变化，面试中应先列组成项再估算，不宜背一个固定显存数字。

### 追问

- **Q：QLoRA 推理时怎么用？**
  A：要么把 LoRA 合并回去（用 fp16 模型）；要么保持 4-bit + LoRA 的组合（dequantize on-the-fly）。

---

## Q03 · 其他 PEFT 方法

### Adapter

在 Transformer 中插入小型 Adapter 模块：

$$
\text{Attention} \to \text{Adapter}(\downarrow \to \text{NonLinear} \to \uparrow) \to \text{FFN} \to \text{Adapter}(\downarrow \to \text{NonLinear} \to \uparrow)
$$

**缺点**：增加推理延迟（不能合并到原权重）。

### Prefix Tuning

在每层的 K/V 前面添加可训练的 prefix：

$$
K' = [P_K; K], \quad V' = [P_V; V]
$$

### Prompt Tuning

在输入 embedding 前添加可训练的 soft prompt：

$$
E' = [P; E]
$$

### PEFT 方法对比

| 方法 | 参数量 | 推理延迟 | 效果 | 推荐 |
|---|---|---|---|---|
| **LoRA** | 0.1-1% | 可 merge；未 merge 时有旁路 | 通常较强 | 常用 |
| **QLoRA** | 0.1-1% | 有反量化/量化 kernel 约束 | 显存效率高 | 常用 |
| Adapter | 1-5% | 增加串行层 | 中等 | 特定场景 |
| Prefix | 约 0.1% | 增加各层 KV 长度 | 中等 | 特定场景 |
| Prompt | <0.1% | 增加输入 token | 依赖模型规模 | 特定场景 |

### 追问

- **Q：DoRA 是什么？**
  A：Decomposed LoRA，把 $W$ 分解为 magnitude + direction，对 direction 做 LoRA。在小 r 下效果比 LoRA 好。

---

## Q04 · 指令微调

### 一句话

指令微调属于有监督微调（SFT），让模型**学会对话格式**和**遵循指令**。

### 损失函数

$$
\mathcal{L} = -\sum_{t \in \text{output}} \log P(x_t \mid x_{1:t-1})
$$

常见对话 SFT 只在 **assistant 回复部分**算损失（不算 user 输入和 system prompt），因为目标是学习“给定上下文后如何回答”。若希望模型学习特殊控制 token、工具调用协议或连续预训练式知识，也可能对更多位置监督；label mask 是训练目标的一部分，不是固定模板。

### 数据：质量 > 数量

| 来源 | 优点 | 缺点 |
|---|---|---|
| **人工标注** | 质量最高 | 成本高 |
| **GPT 蒸馏** | 量大便宜 | 法律风险 |
| **开源数据集** | 免费 | 质量参差 |
| **Self-Instruct** | 自动化 | 需要种子数据 |

### 对话格式

- **方式一**：每轮独立计算损失
- **方式二**：拼接多轮，**只在 assistant 部分计算损失**（推荐）

```
<|system|> You are a helpful assistant.
<|user|> 你好
<|assistant|> 你好！有什么可以帮你的？  ← 只在这部分算 loss
<|user|> 介绍一下 Transformer
<|assistant|> Transformer 是...        ← 只在这部分算 loss
```

### 追问

- **Q：SFT 数据多少够？**
  A：没有通用阈值。少量高质量数据可能足以改变回答格式和风格，但覆盖新任务、长尾指令、多语言与安全边界仍需要足够多样性。LIMA 的结果说明质量和覆盖非常重要，不能外推成“任何模型 1000 条都够”。应看验证集能力覆盖、重复率、loss 分桶和基座能力，而不是只看条数。

---

## Q05 · 常见微调问题

### 1) 灾难性遗忘

**症状**：微调后专项任务好了，但通用能力（数学/代码/常识）下降。

**解决**：
- 使用**较小学习率**（1e-5 ~ 5e-5）
- 混入少量预训练数据（**Replay**）
- 使用 **LoRA**（天然缓解，原参数被冻结）

LoRA 只能减少基座权重直接漂移，不能保证不遗忘：adapter 输出仍可能覆盖原模型行为。应在通用能力回归集上验证，并通过 replay、较小更新幅度、正则或多 adapter 隔离来控制。

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

### 追问

- **Q：LoRA 多任务怎么部署？**
  A：保留一个基座 + 多个 LoRA adapter 文件，按需加载。一台机器可以同时服务多个领域的模型。

---

[⬅ 回到首页](../README.md)
