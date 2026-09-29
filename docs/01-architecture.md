# 01 · Architecture 模型架构

Transformer 及其变体的所有组件 —— 大模型面试**绝对核心**章节。

## 本章目录

- [Q01 · Transformer 整体架构](#q01--transformer-整体架构)
- [Q02 · Self-Attention 自注意力](#q02--self-attention-自注意力)
- [Q03 · Multi-Head Attention 多头注意力](#q03--multi-head-attention-多头注意力)
- [Q04 · MHA / MQA / GQA / MLA 区别](#q04--mha--mqa--gqa--mla-区别)
- [Q05 · 位置编码（绝对 / RoPE / ALiBi）](#q05--位置编码)
- [Q06 · LayerNorm / RMSNorm / Pre-Post-Norm](#q06--layernorm--rmsnorm--prepost-norm)
- [Q07 · FFN 前馈网络（SwiGLU）](#q07--ffn-前馈网络-swiglu)
- [Q08 · Encoder-Only / Decoder-Only / Encoder-Decoder](#q08--encoder-only--decoder-only--encoder-decoder)
- [Q09 · Causal Mask 因果掩码](#q09--causal-mask-因果掩码)
- [Q10 · Tokenization 分词算法](#q10--tokenization-分词算法)
- [Q11 · MoE 混合专家](#q11--moe-混合专家)

---

## Q01 · Transformer 整体架构

### 一句话答案

Transformer 由 **Encoder + Decoder** 组成（每部分由 $N$ 个相同的 block 堆叠），核心组件是
**Self-Attention + FFN + 残差连接 + LayerNorm**。

### 详细展开

```
              Input Embedding + Position Encoding
                          ↓
┌───────────────────── Encoder ×N ────────────────────┐
│   Multi-Head Self-Attention (双向)                    │
│        ↓ + Residual + LayerNorm                      │
│   FFN                                                │
│        ↓ + Residual + LayerNorm                      │
└──────────────────────────────────────────────────────┘
                          ↓
┌───────────────────── Decoder ×N ────────────────────┐
│   Masked Multi-Head Self-Attention (因果)             │
│        ↓ + Residual + LayerNorm                      │
│   Cross-Attention（Q 来自 decoder，K/V 来自 encoder）  │
│        ↓ + Residual + LayerNorm                      │
│   FFN                                                │
│        ↓ + Residual + LayerNorm                      │
└──────────────────────────────────────────────────────┘
                          ↓
                  Linear → Softmax → Output
```

### 面试常见追问

- **Q：为什么 Transformer 比 RNN 好？**
  A：可以**并行**计算（GPU 友好），训练快；RNN 必须串行（每步依赖上一步）。同时 Self-Attention 提供全局感受野，长距离依赖比 RNN 强得多。
  > 注：Mamba / SSM 等新架构在尝试结合两者优点（保留并行 + 接近线性复杂度）。

- **Q：Transformer 的复杂度瓶颈在哪？**
  A：attention 的 score 与加权求和是 $O(n^2d)$，投影与 FFN 约为 $O(nd^2)$。长序列时二次项突出，短序列而模型很宽时 FFN/投影也可能占主要 FLOPs；推理还常受 KV Cache 带宽而非 FLOPs 限制。

---

## Q02 · Self-Attention 自注意力

> 难度：中等 · 高频考点

### 一句话答案

Self-Attention 通过 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$ 让序列中每个位置都能**动态聚合**全局信息，
其核心优势是**并行计算 + 全局感受野**，但代价是 $O(n^2 d)$ 的复杂度。

### 详细展开

#### 1. 核心公式

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

输入 $X \in \mathbb{R}^{n \times d_{model}}$，三个线性投影得到 $Q, K, V$；
注意力分数矩阵 $\frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$ 表示每个 query 对所有 key 的相关性，
softmax 归一化后做 V 的加权和。

#### 2. 为什么除以 $\sqrt{d_k}$？

设一个 query 和 key 的各维近似独立、零均值，且方差分别为
$\sigma_q^2$、$\sigma_k^2$。点积是 $d_k$ 项乘积之和：

```math
\mathrm{Var}(q^\top k)
=\sum_{j=1}^{d_k}\mathrm{Var}(q_jk_j)
\approx d_k\sigma_q^2\sigma_k^2
```

在归一化和常见初始化下可近似取 $\sigma_q^2\approx\sigma_k^2\approx1$，于是点积的**方差**随 $d_k$ 线性增长，标准差随 $\sqrt{d_k}$ 增长。大幅值 logit 会把 softmax 推入饱和区：最大项概率接近 1，其余项接近 0，Jacobian 中的 $p_i(\delta_{ij}-p_j)$ 也随之变小。

因此除以 $\sqrt{d_k}$，正好消掉标准差的增长，使不同 head dimension 下的 logit 尺度大致不变。

**为什么不是除以 $d_k$？** 因为需要归一化的是标准差，不是方差。除以 $d_k$ 会使方差变成 $1/d_k$，维度越大 logits 反而越接近 0，softmax 趋近均匀，注意力分辨率不足。$\sqrt{d_k}$ 不是数学上唯一可能的常数，而是在“独立、单位方差”假设下自然得到的尺度；若实际分布偏离该假设，也可以配合 QK-Norm 或可学习 temperature。

#### 3. 计算复杂度

| 步骤 | 复杂度 |
|---|---|
| $QK^T$ | $O(n^2 d)$ |
| softmax | $O(n^2)$ |
| $\mathrm{attn} \cdot V$ | $O(n^2 d)$ |
| **合计** | $\mathbf{O(n^2 d)}$ |

#### 4. 实现（PyTorch）

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
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn
```

> 易错点：是 `masked_fill`（带 ed），不是 `mask_fill`。

### 面试常见追问

- **Q：为什么 Q、K、V 要用三个不同的投影矩阵，不能共享？**
  A：注意力要同时学习“匹配空间”和“内容空间”。Q/K 决定路由权重，V 决定被路由的内容；若三者共享，改变匹配特征也会被迫改变输出内容。Q、K 理论上可以共享，但会强制相似度采用同一投影下的对称双线性形式，无法表达 query 与 key 的角色不对称。分开投影不是为了形状，而是减少这种结构约束。

- **Q：为什么用 softmax 而不是 sigmoid？**
  A：softmax 让每个 query 的权重和为 1，输出是 V 的归一化加权和，并对所有 logits 同加常数不变；序列变长时输出尺度也较稳定。sigmoid 让各 key 独立开关，权重和会随激活 key 数变化，需要额外处理尺度。sigmoid attention 并非不可用，只是对应不同归纳偏置。

- **Q：mask 为什么填 $-\infty$ 而不是 0？**
  A：mask 在 softmax **之前**应用。$e^0 = 1$ 仍参与归一化；$e^{-\infty} = 0$ 才能真正屏蔽。

- **Q：Attention 计算的是 $O(n^2)$，怎么优化到长序列？**
  A：Flash Attention（IO 优化）/ 稀疏 attention（Longformer）/ 线性 attention（Performer）/ State Space Models（Mamba）。

- **Q：训练时 attention 加 dropout 加在哪？**
  A：标准做法是加在 **softmax 后、乘 V 前** 的 attention weights 上。

---

## Q03 · Multi-Head Attention 多头注意力

### 公式

$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O
$$

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

设总维度 $d_{model}$、头数 $h$，则 $d_k = d_v = d_{model}/h$，**总参数量与单头相同**。

### 为什么要多头？

1. 每个头拥有独立的投影，可在不同子空间中定义不同的相似度
2. 单个 softmax 每个 query 只产生一套归一化权重；多头允许同一位置同时形成多套分布，再由 $W^O$ 混合
3. 当总维度固定时，投影和主干矩阵乘的参数量、数量级 FLOPs 与一个 $d_{model}$ 维单头相近，但并非“免费”：切头、softmax、kernel 启动和中间张量仍有开销

多头并不保证每个头都学出可解释的“语法头”或“位置头”，头之间也可能冗余。更准确的说法是：它扩大了可表示的注意力分布族，而不是预先规定每个头的功能。

### 实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        # (B, n_heads, N, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V)                 # (B, n_heads, N, d_k)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_O(ctx)
```

### 面试常见追问

- **Q：头数 $h$ 怎么选？**
  A：经验上 $d_k$ 不要太小（保留每个头表达能力），常见 $d_k = 64\text{-}128$。LLaMA-7B 是 $32 \text{ heads}$、$d_k = 128$。

- **Q：多头之间会不会冗余？**
  A：会。论文 *Are Sixteen Heads Really Better than One?* 证明很多头可以剪枝而精度不降，催生了 GQA/MQA。

---

## Q04 · MHA / MQA / GQA / MLA 区别

### 一句话总结

随着模型变大，**KV Cache 成为推理瓶颈**，于是出现了一系列让多 Q 头**共享 K/V 头**的变种。

### 对比表

| 方法 | 描述 | Q 头数 | K/V 头数 | KV Cache | 代表模型 |
|---|---|---|---|---|---|
| **MHA** | 标准多头 | $h$ | $h$ | 大 | GPT-2, BERT |
| **MQA** | 多查询注意力 | $h$ | $1$ | 最小 | PaLM, Falcon |
| **GQA** | 分组查询注意力 | $h$ | $g$（$1 < g < h$） | 中等 | LLaMA-2 70B, LLaMA-3 |
| **MLA** | 多头潜在注意力 | $h$ | 压缩到低维潜在空间 | 取决于压缩维与位置分量 | DeepSeek-V2/V3 |

### MLA 原理（DeepSeek-V2）

采用列向量记号：输入 $h_t$ 经下投影得到共享潜变量 $c_t=W^{DKV}h_t$，各头的内容 key/value 再从 $c_t$ 投影。解耦 RoPE 将内容与位置子空间**拼接**，不是逐元素相加：

```math
k_{t,i}=[k^C_{t,i};k^R_t],\qquad
q_{t,i}=[q^C_{t,i};q^R_{t,i}]
```

其中 $k^R_t=\mathrm{RoPE}(W^{KR}h_t,t)$，既依赖输入内容也依赖位置，不是只由位置编号生成。点积因此分成内容项与位置项：$(q^C)^\top k^C+(q^R)^\top k^R$；缩放维度应是拼接后 Q/K 的总维数。

若内容 key 为 $W^{UK}_i c_t$，则 $(q^C)^\top W^{UK}_i c_t=((W^{UK}_i)^\top q^C)^\top c_t$。推理可将上投影吸收到 query 侧，value 上投影也可与输出投影合并，避免物化完整的历史 KV。若直接在内容 key 上施加随位置变化的旋转，这种固定权重吸收会受到阻碍，因此单独保留位置分支。

缓存包含 **$c_t$ 和旋转后的位置 key $k^R_t$**，每层每 token 的元素数为 $d_c+d_R$，不是只有 latent，也不是固定省某个比例。减少多少应与同一基线的 KV 头数和维度比较。来源：[DeepSeek-V2 第 2.1 节](https://arxiv.org/html/2405.04434v5)。

### 面试常见追问

- **Q：MQA 为什么会掉点？**
  A：共享 K/V 减少了不同 query 头可使用的独立内容子空间，可能损失质量，但下降幅度依赖模型规模、训练方式和任务，并非必然“大幅”。GQA 用多个 KV 组在 cache 与表达能力之间折中。

- **Q：GQA 的分组数怎么选？**
  A：没有固定 $g=h/8$ 规则。KV 头越少，cache 和带宽越省，但共享约束越强；应在目标 batch/上下文下测质量、KV 显存和 decode 吞吐，并满足 Q heads 能整除 KV heads 等实现约束。

---

## Q05 · 位置编码

### 为什么需要位置编码？

不带位置编码、且没有固定位置 mask 的全连接 Self-Attention 对序列置换是**置换等变**的：输入重排，输出相应重排；再做对称 pooling 才成为置换不变。固定 causal mask 本身已包含顺序约束，不能对它直接套用任意置换等变的结论。位置编码进一步提供位置/距离信息，让模型不仅依赖内容匹配与可见范围。

### 三大流派

#### 1. **正弦位置编码**（原始 Transformer）

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

偶数维用 sin、奇数维用 cos，不同维度频率不同（**低维变化快，高维变化慢**）。

**重要性质**：位置 $pos+k$ 的编码可以由位置 $pos$ **线性组合**表示出来 → 模型能从绝对位置推断相对位移。

**优点**：

1. 无需额外参数
2. 可外推到训练未见的长度（理论上）

**缺点**：

1. 本质是**绝对**位置编码
2. 直接加到 embedding，深层后位置信息会被内容稀释
3. 长距离建模不一定最优

#### 2. **RoPE 旋转位置编码**（LLaMA / Qwen / 主流）

**核心思想**：不再把位置加到 token，而是**直接旋转 Q 和 K**：

$$
q_m = R_m q, \quad k_n = R_n k
$$

**关键性质**：

$$
\langle R_m q,\ R_n k \rangle = q^T R_m^T R_n k = q^T R_{n-m} k
$$

→ 位置旋转项只通过 $n-m$ 起作用，但内积仍依赖内容向量 $q,k$，不能说整个 attention 分数只依赖距离。

实际实现：把向量两两分组，每组做 2D 旋转，不同组用不同频率。

**为什么 RoPE 这么受欢迎？**

- 把"加性位置"升级成"几何变换"
- **天然提供相对位置语义**
- 通过频率调整可外推（NTK-aware / YaRN）
- 不增加参数

#### 3. **ALiBi**（BLOOM）

不学位置编码，而是在 attention 分数上加一个**与距离成正比的偏置**：

$$
\text{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|
$$

$m$ 是每个头不同的固定斜率。优点：**外推性极强**。

### 面试常见追问

- **Q：RoPE 怎么做长上下文外推？**
  A：常用 Position Interpolation、NTK-aware scaling、YaRN 等重新映射位置或不同频率；调大 base 只是其中一种参数化。还通常需要长序列继续训练，并同时检查短上下文退化和远距离检索，不能只改配置里的最大长度。

- **Q：Learned absolute position embedding 为什么在生成式 LLM 中变少？**
  A：它为每个训练位置存独立向量，超过表长没有定义，也较难共享相对位移规律；RoPE/相对 bias 更适合变长和外推。但 learned position 仍可用于固定长度模型，不能说已经完全不用。

---

## Q06 · LayerNorm / RMSNorm / Pre/Post-Norm

### LayerNorm 公式

$$
\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

沿 **特征维度**归一化，与 batch 无关。

### 可学习参数为什么是 $\gamma$ 和 $\beta$？

LayerNorm 的均值、方差由当前 token 的特征即时计算，**不是可学习参数**；真正可学习的是逐特征的缩放 $\gamma\in\mathbb{R}^d$ 和偏移 $\beta\in\mathbb{R}^d$。标准化把每个 token 压到统一的一阶、二阶统计，但不同通道未必都应该保持单位尺度、零中心。仿射参数让网络在保留稳定性的同时，重新学习每个通道合适的幅值和基线；必要时甚至能恢复标准化前后续层所需的尺度。

BatchNorm 也是同样的 $\gamma、\beta$ 设计，只是统计轴不同。batch mean/variance 不是独立可学习参数，但训练时仍在计算图内，反向传播必须考虑它们对输入的依赖；running mean/variance 才是以滑动统计更新的 buffer。$\gamma=1、\beta=0$ 仅表示“不额外缩放或平移标准化结果”，**不代表整个归一化层是恒等映射**。固定仿射参数也不能恢复每个样本原来不同的均值和方差。

### 为什么 LN 用有偏方差？BN、RMSNorm 也一样吗？

先区分两个目标：描述当前这组数的离散程度，与用随机样本估计未知总体方差。LN 用前者归一化当前向量，不以总体方差的无偏估计为目标。设统计轴有 $n$ 个元素：

```math
\mu=\frac1n\sum_i x_i,\qquad
v_n=\frac1n\sum_i(x_i-\mu)^2
```

若把 $x_i$ 看作同分布独立样本，则 $E[v_n]=(n-1)\sigma^2/n$；因为用这些样本估计了均值，偏差残差之和为零，损失一个自由度。除以 $n-1$ 可在这些假设下无偏估计总体方差。但特征维并不天然是某个总体的独立同分布抽样；“无偏”也不等于归一化效果更好。

忽略 epsilon 且 $v_n>0$，用 $\sqrt{v_n}$ 归一化后，当前向量的均方偏差为 1；若改用 $v_{n-1}$，则为 $(n-1)/n$。两种计算都能定义，不是另一种无法反传，而是采用不同尺度约定。实际 LN 的仿射前均方偏差为 $v_n/(v_n+\epsilon)$，仿射后也不保证零均值、单位方差。

| 层 / 用途 | 统计量 | 关键区别 |
|---|---|---|
| LN，训练与推理 | 中心方差，除以 n | 都使用当前输入的统计量 |
| PyTorch BN，训练前向 | 中心方差，除以 n | 用于归一化当前 batch |
| PyTorch BN，更新 running_var | 使用除以 n−1 的方差估计更新滑动平均 | 默认推理用历史统计，不等于历史平均始终严格无偏 |
| RMSNorm，训练与推理 | 均方 `mean(x²)` | 不减均值，不能称为“有偏方差” |

RMSNorm 的均方满足 $\frac1n\sum_i x_i^2=v_n+\mu^2$。它控制相对原点的幅值，不只控制围绕均值的波动，因此不需要为“估计均值”作贝塞尔校正。

例：$x=[1,3]$，均值为 2，有偏方差为 1，无偏方差为 2，均方为 5。忽略 epsilon 与仿射，LN 输出 `[-1,1]`，RMSNorm 输出 $[1,3]/\sqrt5$，两者含义不同。常量向量的 LN 标准化输出为零，epsilon 防止除零；$n=1$ 时 LN 也有定义，而除以 $n-1$ 不成立。

这里 n 是**统计轴的元素数**：二维 BN 为 B，图像 BN 通常为 B×H×W；LN 可对指定尾部多维归一化，并非框架只能处理最后一维。具体行为及实现见 [PyTorch LN](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html)、[BN](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm1d.html)、[Norm 手撕](../llm-coding/03-normalization.md)。

### Pre-Norm vs Post-Norm

| | Post-Norm（原始 Transformer） | Pre-Norm（GPT-2 / LLaMA） |
|---|---|---|
| 公式 | $\mathrm{LN}(x + \text{SubLayer}(x))$ | $x + \text{SubLayer}(\mathrm{LN}(x))$ |
| 梯度路径 | 恒等支路也经过 LN Jacobian | 含不经过子层与 LN 的直接恒等项 |
| 训练稳定性 | 深层时通常更依赖初始化、warmup 与残差缩放 | 常更易训练，但不保证任意深度稳定 |
| 实际选择 | 旧架构 | **现代大模型首选** |

### RMSNorm（LLaMA 使用）

LayerNorm 的简化版，去掉均值中心化和偏移 $\beta$：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

- **算子更简单**（少均值中心化和常见的 $\beta$；实际速度取决于 fused kernel）
- 实验证明效果与 LayerNorm 相当

### LayerNorm vs BatchNorm

| 特性 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化维度 | 每通道跨 batch（及空间/时间轴） | 每个样本/token 的 feature 维 |
| 依赖 batch size | 是 | 否 |
| 适用场景 | CV | NLP / Transformer |
| 推理时 | 默认使用 running stats；关闭追踪时仍用 batch stats | 直接计算 |

### 面试常见追问

- **Q：Transformer 为什么通常不用 BatchNorm？**
  A：核心不是“有 padding 就一定不能用”，padding 可以做 masked statistics；真正的问题是 BN 把同一通道在 batch/token 轴上的样本耦合起来，统计量随 batch size、序列长度组成和数据并行分片变化，训练与自回归推理还要切换到 running statistics。LN/RMSNorm 对每个 token 独立计算，训练与推理路径一致，更适合变长序列和小 micro-batch。

- **Q：RMSNorm 比 LayerNorm 好在哪？**
  A：它只控制 RMS 尺度，计算路径更简单，并在许多 LLM 上达到相近效果；具体速度与质量差异依赖实现和架构，不能固定成 10%。

---

## Q07 · FFN 前馈网络（SwiGLU）

### 标准 FFN

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(xW_1 + b_1)W_2 + b_2
$$

两层 MLP，中间维度通常是 $4 d_{model}$。

### SwiGLU（LLaMA / PaLM）

$$
\mathrm{SwiGLU}(x) = (\mathrm{Swish}(xW_1) \odot xW_3)W_2
$$

其中 $\mathrm{Swish}(x) = x \cdot \sigma(\beta x)$，$\odot$ 是逐元素相乘（GLU 门控）。

**优势**：

- 实验：相同计算量下性能更好
- 门控让网络**学习选择性激活**
- 现代大模型标准选择

> **关于参数量**：标准 FFN 中间维度是 $4d$；SwiGLU 由于多了一个矩阵 $W_3$，为了保持总参数量不变，中间维度通常调整为 $\frac{8}{3}d \approx 2.67d$。

### 面试常见追问

- **Q：为什么标准 FFN 常取 $4d$？一定是 4 吗？**
  A：不是定理，而是原始 Transformer 延续下来的容量—计算折中。两层 FFN 参数量约为 $2dd_{ff}$；增大 $d_{ff}$ 会增加逐 token 的非线性特征和记忆容量，也线性增加参数与 FLOPs。$4d$ 是常用基线，具体模型会按总参数预算、门控结构和硬件对齐调整。SwiGLU 有三块矩阵，为保持与 $d_{ff}=4d$ 的普通 FFN 相近的参数量，常取约 $8d/3$，这才是 $8/3$ 的来源。

- **Q：FFN 在 Transformer 里起什么作用？**
  A：Self-Attention 提供**位置间的信息混合**，FFN 提供**位置内的特征变换**。两者互补，缺一不可。

---

## Q08 · Encoder-Only / Decoder-Only / Encoder-Decoder

### 三大架构

| 架构 | 注意力 | 代表模型 | 适用任务 |
|---|---|---|---|
| **Encoder-Only** | 双向 | BERT, RoBERTa | 分类、NER、理解 |
| **Decoder-Only** | 单向（因果） | GPT, LLaMA, DeepSeek | 生成、对话 |
| **Encoder-Decoder** | 双向 + 因果 | T5, BART, mT5 | 翻译、摘要 |

### 为什么许多生成式 LLM 选择 Decoder-Only？

1. **统一的 next-token prediction** 范式简单且强大
2. **Scaling Law** 对 Decoder-Only 最友好
3. GPT 系列验证了仅 Decoder 就能做好几乎所有任务
4. **In-Context Learning** 在 Decoder 中涌现
5. **推理效率高**：KV Cache 天然适配因果注意力
6. **训练目标统一**：因果语言建模可在序列中几乎每个非首 token 上产生监督；BERT 式 MLM 通常只在被选中位置计算主要预测损失，但“15%”是具体训练配置，不是所有 Encoder 的固有限制

这不是“Decoder-Only 在所有任务都更优”的证明。Encoder-only 仍适合双向表示和高吞吐理解，Encoder-Decoder 在输入输出职责不同、需要完整编码源序列的任务中也有优势；Decoder-Only 的核心吸引力是用同一接口和训练目标统一多种生成任务。

---

## Q09 · Causal Mask 因果掩码

### 作用

在 Decoder 中，每个 token 只能看到自己和之前的 token，靠 **下三角掩码**实现。

掩码加在 **softmax 之前**：

$$
\text{attn} = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \mathrm{Mask}\right)
$$

其中 Mask 上三角部分填 $-\infty$（被屏蔽位置），下三角填 $0$。

```python
seq_len = x.size(1)
mask = torch.tril(torch.ones(seq_len, seq_len))   # 下三角全 1
scores = scores.masked_fill(mask == 0, float("-inf"))
```

### 追问

- **Q：训练和推理时 Causal Mask 的区别？**
  A：训练时一次性算完整 mask；推理时配合 KV Cache，每步只算当前 query 对所有 key 的注意力（mask 自然成立）。

---

## Q10 · Tokenization 分词算法

### 主流方法对比

SentencePiece 是可以承载 BPE 或 Unigram 的**分词框架/实现方式**，并不是与 BPE、WordPiece 同层级的单一合并算法：

| 特性 | BPE | WordPiece | SentencePiece（框架） |
|---|---|---|---|
| 合并策略 | 最高频率对 | 最大似然增益 | BPE/Unigram 在句子片段上 |
| 预分词 | 需要 | 需要 | 不需要（直接处理原始文本） |
| 空格处理 | 保留 | `##` 标记续接 | `▁` 标记开头 |
| 使用模型 | GPT, LLaMA | BERT | T5, LLaMA, ChatGLM |

### BPE vs Unigram

| 特性 | BPE | Unigram |
|---|---|---|
| 方向 | **自底向上**（合并） | **自顶向下**（裁剪） |
| 初始化 | 字符 | 大词表 |
| 过程 | 逐步合并 | 逐步删除 |
| 分词 | 确定性 | 概率性（可采样） |

### 中文 LLM 分词的特殊挑战

**挑战**：

- 没有天然空格分隔
- 词粒度不确定（"中华人民共和国"是一个词还是多个？）
- 字符集巨大

**解决方案**：

1. **字级别**：每个汉字一个 token（BERT-Chinese）
2. **SentencePiece**：直接在字节/字符序列上训练 BPE/Unigram
3. **混合方案**：先用 jieba 中文分词，再 BPE

**现代做法**（LLaMA / ChatGLM）：

- SentencePiece BPE 模式
- 增大中文语料比例
- 扩展词表加入中文 token（Chinese-LLaMA-Alpaca）

### 追问

- **Q：BPE 一个 token 等于几个汉字？**
  A：没有固定换算。结果取决于预分词、字节映射、词表和训练语料；常见汉字可能单独成 token，高频词组也可能合成一个 token，未覆盖字符则可能回退为多个 byte token。面试中应以具体 tokenizer 实测，不能把“1 字 1 token”当作保证。

---

## Q11 · MoE 混合专家

### 核心思想

把 FFN 替换成**多个并行的"专家"FFN**，每个 token 由一个 **router** 路由到 **Top-k 个**专家（通常 k=1 或 2）。

```
            ┌─→ Expert 1 ─┐
input ──► Router ─→ Expert 2 ─┼─→ 加权求和
            └─→ Expert N ─┘
```

### 优势

- **参数量大但激活量小**：万亿级模型推理时只激活几十亿参数
- 不同专家可学不同模式

### 关键挑战

- **负载均衡**：避免所有 token 都去同一个专家 → 加 **load balancing loss**
- **训练不稳定**：router 决策是离散的
- **通信开销**：专家分布在不同 GPU 上，all-to-all 通信

### 为什么稀疏激活能扩参数而不等比例扩 FLOPs？

若有 $E$ 个专家、每个 token 只选 $k$ 个，FFN 的存储参数量随 $E$ 增长，而单 token 的专家计算主要随 $k$ 增长。这实现了“总容量大、激活计算小”。但显存、参数加载、router、token dispatch 和跨卡 all-to-all 不会消失，因此 MoE 不是把稠密模型的成本简单乘上 $k/E$。

负载均衡损失也不是越强越好：太弱会造成热点、溢出和设备空闲；太强会迫使 router 为均衡而牺牲语义路由。工程上还要同时看 expert utilization、capacity factor、dropped-token rate 和通信占比。

### 代表模型

- **Switch Transformer**：早期纯 Top-1 MoE
- **Mixtral 8x7B**：8 专家 Top-2，激活 ~13B 推理
- **DeepSeek-V2/V3**：细粒度专家 + 共享专家
- **Qwen3.5**：MoE + 线性注意力混合

### 追问

- **Q：MoE 显存怎么算？**
  A：训练时所有专家都得在显存里（**总参数量**）；推理时也都得在（除非 expert offloading），但只算激活的那 k 个，所以 FLOPs 小。

---

[⬅ 回到首页](../README.md)
