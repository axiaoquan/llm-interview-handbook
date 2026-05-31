# 01 · Architecture 模型架构

Transformer 及其变体的所有组件 —— 大模型面试**绝对核心**章节。

## 📑 本章目录

- [Q01 · Self-Attention 自注意力](#q01--self-attention-自注意力)
- [Q02 · Multi-Head Attention 多头注意力](#q02--multi-head-attention-多头注意力) ⏳
- [Q03 · MHA / MQA / GQA / MLA 区别](#q03--mha--mqa--gqa--mla-区别) ⏳
- [Q04 · 位置编码（绝对 / 相对 / RoPE / ALiBi）](#q04--位置编码) ⏳
- [Q05 · LayerNorm vs BatchNorm vs RMSNorm（Pre/Post-Norm）](#q05--layernorm-vs-batchnorm-vs-rmsnorm) ⏳
- [Q06 · FFN 前馈网络（SwiGLU / GeLU）](#q06--ffn-前馈网络) ⏳
- [Q07 · Encoder vs Decoder vs Decoder-only 的取舍](#q07--encoder-vs-decoder-vs-decoder-only) ⏳
- [Q08 · MoE 结构与负载均衡](#q08--moe-结构与负载均衡) ⏳

> 标记 ⏳ 表示尚未填充，待补充。

---

## Q01 · Self-Attention 自注意力

> 难度：⭐⭐⭐ · 常见公司：所有大厂必考

### 🎯 一句话答案

Self-Attention 通过 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$ 让序列中每个位置都能**动态聚合**全局信息，
其核心优势是**并行计算 + 全局感受野**，但代价是 $O(n^2 d)$ 的复杂度。

### 📖 详细展开

#### 1. 核心公式

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

输入 $X \in \mathbb{R}^{n \times d_{model}}$，三个线性投影得到 $Q, K, V$；
注意力分数矩阵 $\frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$ 表示每个 query 对所有 key 的相关性，
softmax 归一化后做 V 的加权和。

#### 2. 为什么除以 $\sqrt{d_k}$？

当 $d_k$ 较大时，$Q$ 与 $K$ 的内积方差约为 $d_k$（Xavier 初始化下），
$QK^T$ 的元素会远离 0，softmax 结果接近 one-hot，**梯度趋近于 0** —— 训练崩溃。

除以 $\sqrt{d_k}$ 把方差稳定回 1 附近，softmax 输出分布平缓，梯度可学。

#### 3. 计算复杂度

| 步骤 | 复杂度 |
|---|---|
| $QK^T$ | $O(n^2 d)$ |
| softmax | $O(n^2)$ |
| $\mathrm{attn} \cdot V$ | $O(n^2 d)$ |
| **合计** | $\mathbf{O(n^2 d)}$ |

这就是 long-context 时代 Flash Attention / Linear Attention 等优化的出发点。

#### 4. 实现（PyTorch 50 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_Q(x)   # (batch, seq_len, d_k)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn
```

> ⚠️ 易错点：是 `masked_fill`（不是 `mask_fill`）。这是面试现场写代码常翻车的地方。

### 🪤 面试常见追问

- **Q：为什么 attention 中 Q、K、V 要用三个不同的投影矩阵，不能共享？**
  A：Query 和 Key 表达"我关心什么 / 我能被谁关心"，是**对偶语义**；Value 表达"我能贡献的信息"。
  共享会限制表达能力，实验上效果显著变差。

- **Q：为什么用 softmax 而不是 sigmoid？**
  A：softmax 强制概率和为 1，体现"注意力分配"的零和性；sigmoid 各位置独立，
  会出现"每个位置都很重要 / 都不重要"的退化。

- **Q：mask 为什么要填 $-\infty$ 而不是 0？**
  A：mask 在 softmax **之前**应用。softmax 内部是 $e^x$，$e^0 = 1$ 仍参与归一化；
  $e^{-\infty} = 0$ 才能真正屏蔽。

- **Q：复杂度 $O(n^2 d)$ 中的 $n^2$ 来自哪里？怎么优化？**
  A：来自 $QK^T$ 矩阵。优化方向：**稀疏 attention**（Longformer）、**线性 attention**（Performer）、
  **IO 优化**（Flash Attention）、**State Space Models**（Mamba）。

- **Q：训练时 attention 加 dropout 加在哪？**
  A：标准做法是加在 **softmax 后、乘 V 前** 的 attention weights 上。

### 📚 参考

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

---

## Q02 · Multi-Head Attention 多头注意力

⏳ _待补充。来源笔记位于 `/Users/yaozq/file/work/大模型八股.md` 的"3. Multi-Head Attention"段落。_

---

## Q03 · MHA / MQA / GQA / MLA 区别

⏳ _待补充。重点对比四者在 KV Cache 大小、表达能力、推理速度上的差异。_

---

## Q04 · 位置编码

⏳ _待补充：绝对位置编码 / 相对位置编码 / RoPE / ALiBi。_

---

## Q05 · LayerNorm vs BatchNorm vs RMSNorm

⏳ _待补充：包括 Pre-Norm 与 Post-Norm 的区别。_

---

## Q06 · FFN 前馈网络

⏳ _待补充：SwiGLU / GeLU 等激活函数。_

---

## Q07 · Encoder vs Decoder vs Decoder-only

⏳ _待补充：为什么主流 LLM 都是 Decoder-only。_

---

## Q08 · MoE 结构与负载均衡

⏳ _待补充：路由策略、辅助损失、Sparse Top-k。_

---

[⬅ 回到首页](../README.md)
