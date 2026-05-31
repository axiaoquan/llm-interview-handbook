# 01 · Architecture 模型架构

Transformer 及其变体的所有组件 —— 大模型面试**绝对核心**章节。

## 📑 本章目录

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

### 🎯 一句话答案

Transformer 由 **Encoder + Decoder** 组成（每部分由 $N$ 个相同的 block 堆叠），核心组件是
**Self-Attention + FFN + 残差连接 + LayerNorm**。

### 📖 详细展开

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

### 🪤 面试常见追问

- **Q：为什么 Transformer 比 RNN 好？**
  A：可以**并行**计算（GPU 友好），训练快；RNN 必须串行（每步依赖上一步）。同时 Self-Attention 提供全局感受野，长距离依赖比 RNN 强得多。
  > 注：Mamba / SSM 等新架构在尝试结合两者优点（保留并行 + 接近线性复杂度）。

- **Q：Transformer 的复杂度瓶颈在哪？**
  A：Self-Attention 是 $O(n^2 d)$，长序列下成为瓶颈，催生了 Flash Attention / 线性注意力 / 滑动窗口等优化。

---

## Q02 · Self-Attention 自注意力

> 难度：⭐⭐⭐ · 常见公司：所有大厂必考

### 🎯 一句话答案

Self-Attention 通过 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$ 让序列中每个位置都能**动态聚合**全局信息，
其核心优势是**并行计算 + 全局感受野**，但代价是 $O(n^2 d)$ 的复杂度。

### 📖 详细展开

#### 1. 核心公式

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
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

> ⚠️ 易错点：是 `masked_fill`（带 ed），不是 `mask_fill`。

### 🪤 面试常见追问

- **Q：为什么 Q、K、V 要用三个不同的投影矩阵，不能共享？**
  A：Q/K 表达"我关心什么 / 我能被谁关心"是**对偶语义**，V 是"我能贡献的信息"。共享会限制表达能力。

- **Q：为什么用 softmax 而不是 sigmoid？**
  A：softmax 强制概率和为 1，体现"注意力分配"的零和性；sigmoid 各位置独立，会出现"全都重要"的退化。

- **Q：mask 为什么填 $-\infty$ 而不是 0？**
  A：mask 在 softmax **之前**应用。$e^0 = 1$ 仍参与归一化；$e^{-\infty} = 0$ 才能真正屏蔽。

- **Q：Attention 计算的是 $O(n^2)$，怎么优化到长序列？**
  A：Flash Attention（IO 优化）/ 稀疏 attention（Longformer）/ 线性 attention（Performer）/ State Space Models（Mamba）。

- **Q：训练时 attention 加 dropout 加在哪？**
  A：标准做法是加在 **softmax 后、乘 V 前** 的 attention weights 上。

---

## Q03 · Multi-Head Attention 多头注意力

### 🎯 公式

$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O
$$

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

设总维度 $d_{model}$、头数 $h$，则 $d_k = d_v = d_{model}/h$，**总参数量与单头相同**。

### 📖 为什么要多头？

1. 不同的头代表**不同的注意力模式**（语法、语义、位置等）
2. 模型可以在不同位置同时关注**不同子空间**的信息
3. 计算量与单头相同，但**表达能力更强**

### 🛠 实现

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

### 🪤 面试常见追问

- **Q：头数 $h$ 怎么选？**
  A：经验上 $d_k$ 不要太小（保留每个头表达能力），常见 $d_k = 64\text{-}128$。LLaMA-7B 是 $32 \text{ heads}$、$d_k = 128$。

- **Q：多头之间会不会冗余？**
  A：会。论文 *Are Sixteen Heads Really Better than One?* 证明很多头可以剪枝而精度不降，催生了 GQA/MQA。

---

## Q04 · MHA / MQA / GQA / MLA 区别

### 🎯 一句话总结

随着模型变大，**KV Cache 成为推理瓶颈**，于是出现了一系列让多 Q 头**共享 K/V 头**的变种。

### 📊 对比表

| 方法 | 描述 | Q 头数 | K/V 头数 | KV Cache | 代表模型 |
|---|---|---|---|---|---|
| **MHA** | 标准多头 | $h$ | $h$ | 大 | GPT-2, BERT |
| **MQA** | 多查询注意力 | $h$ | $1$ | 最小 | PaLM, Falcon |
| **GQA** | 分组查询注意力 | $h$ | $g$（$1<g<h$） | 中等 | LLaMA-2 70B, LLaMA-3 |
| **MLA** | 多头潜在注意力 | $h$ | 压缩到低维潜在空间 | 最小 | DeepSeek-V2/V3 |

### 📖 MLA 原理（DeepSeek-V2）

- 把 KV 压缩到低维潜在空间 $c_t = X W_{DKV}$
- 推理时只缓存低维 $c_t$，需要时用解耦矩阵投影回来
- KV Cache 减少 **~93.3%**

**解耦的相对位置编码**：MLA 把 $K$ 分解为

$$
K_i = K_i^{\text{content}} + K_i^{\text{position}}
$$

$K_i^{\text{content}}$ 来自潜在向量 $c$ 投影，$K_i^{\text{position}}$ 直接由位置编码生成，避免对压缩向量做复杂旋转。

### 🪤 面试常见追问

- **Q：MQA 为什么会掉点？**
  A：所有 Q 头共享一个 K/V，表达能力大幅下降。GQA 是 MHA 和 MQA 之间的折中（既省 cache 又保留差异性）。

- **Q：GQA 的分组数怎么选？**
  A：经验上 $g = h/8$ 是一个不错的平衡点（LLaMA-2 70B：64 头 Q、8 头 K/V）。

---

## Q05 · 位置编码

### 🎯 为什么需要位置编码？

Self-Attention 是**置换不变**的——打乱 token 顺序结果不变。所以必须显式注入位置信息。

### 📖 三大流派

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

#### 2. **RoPE 旋转位置编码**（LLaMA / Qwen / 主流）⭐⭐⭐

**核心思想**：不再把位置加到 token，而是**直接旋转 Q 和 K**：

$$
q_m = R_m q, \quad k_n = R_n k
$$

**关键性质**：

$$
\langle R_m q,\ R_n k \rangle = q^T R_m^T R_n k = q^T R_{n-m} k
$$

→ 内积**只和 $n - m$ 有关**，天然带相对位置信息。

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

### 🪤 面试常见追问

- **Q：RoPE 怎么做长上下文外推？**
  A：把 RoPE 的 base（默认 10000）调大，或者用 **NTK-aware Scaling / YaRN**，对低频维度做调整保持精度。

- **Q：为什么大家不用 learned position embedding 了？**
  A：训练长度限定后无法外推；RoPE 性能更好。

---

## Q06 · LayerNorm / RMSNorm / Pre/Post-Norm

### 🎯 LayerNorm 公式

$$
\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

沿 **特征维度**归一化，与 batch 无关。

### 📖 Pre-Norm vs Post-Norm

| | Post-Norm（原始 Transformer） | Pre-Norm（GPT-2 / LLaMA） |
|---|---|---|
| 公式 | $x + \mathrm{LN}(\text{SubLayer}(x))$ | $x + \text{SubLayer}(\mathrm{LN}(x))$ |
| 表达能力 | 理论更强 | 略弱 |
| 训练稳定性 | 差，需要 warmup | **好**（残差路径无变换） |
| 实际选择 | 旧架构 | **现代大模型首选** |

### 📖 RMSNorm（LLaMA 使用）

LayerNorm 的简化版，去掉均值中心化和偏移 $\beta$：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

- **计算更快**（少一次减均值和加 $\beta$）
- 实验证明效果与 LayerNorm 相当

### 📖 LayerNorm vs BatchNorm

| 特性 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化维度 | Batch 维 | Feature 维 |
| 依赖 batch size | ✅ | ❌ |
| 适用场景 | CV | NLP / Transformer |
| 推理时 | 需要 running stats | 直接计算 |

### 🪤 面试常见追问

- **Q：Transformer 为什么不用 BatchNorm？**
  A：Batch 内序列长度不同（padding），统计量不稳；LayerNorm 沿 feature 归一化天然不受影响。

- **Q：RMSNorm 比 LayerNorm 好在哪？**
  A：少一次操作 → 速度快 ~10%；实证效果相当；LLaMA 系列默认用它。

---

## Q07 · FFN 前馈网络（SwiGLU）

### 📖 标准 FFN

$$
\mathrm{FFN}(x) = \mathrm{ReLU}(xW_1 + b_1)W_2 + b_2
$$

两层 MLP，中间维度通常是 $4 d_{model}$。

### 📖 SwiGLU（LLaMA / PaLM）

$$
\mathrm{SwiGLU}(x) = (\mathrm{Swish}(xW_1) \odot xW_3)W_2
$$

其中 $\mathrm{Swish}(x) = x \cdot \sigma(\beta x)$，$\odot$ 是逐元素相乘（GLU 门控）。

**优势**：

- 实验：相同计算量下性能更好
- 门控让网络**学习选择性激活**
- 现代大模型标准选择

> **关于参数量**：标准 FFN 中间维度是 $4d$；SwiGLU 由于多了一个矩阵 $W_3$，为了保持总参数量不变，中间维度通常调整为 $\frac{8}{3}d \approx 2.67d$。

### 🪤 面试常见追问

- **Q：为什么 FFN 中间维度是 $4d$？**
  A：经验最优。理论解释：FFN 提供**特征维度上的非线性扩展**，需要足够大的中间维度才能 hold 住表达能力。

- **Q：FFN 在 Transformer 里起什么作用？**
  A：Self-Attention 提供**位置间的信息混合**，FFN 提供**位置内的特征变换**。两者互补，缺一不可。

---

## Q08 · Encoder-Only / Decoder-Only / Encoder-Decoder

### 📊 三大架构

| 架构 | 注意力 | 代表模型 | 适用任务 |
|---|---|---|---|
| **Encoder-Only** | 双向 | BERT, RoBERTa | 分类、NER、理解 |
| **Decoder-Only** | 单向（因果） | GPT, LLaMA, DeepSeek | 生成、对话 |
| **Encoder-Decoder** | 双向 + 因果 | T5, BART, mT5 | 翻译、摘要 |

### 🪤 为什么现在的 LLM 都是 Decoder-Only？

1. **统一的 next-token prediction** 范式简单且强大
2. **Scaling Law** 对 Decoder-Only 最友好
3. GPT 系列验证了仅 Decoder 就能做好几乎所有任务
4. **In-Context Learning** 在 Decoder 中涌现
5. **推理效率高**：KV Cache 天然适配因果注意力
6. **数据利用率**：每个 token 都是训练信号（双向模型只能用 mask 的 15%）

---

## Q09 · Causal Mask 因果掩码

### 🎯 作用

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

### 🪤 追问

- **Q：训练和推理时 Causal Mask 的区别？**
  A：训练时一次性算完整 mask；推理时配合 KV Cache，每步只算当前 query 对所有 key 的注意力（mask 自然成立）。

---

## Q10 · Tokenization 分词算法

### 📊 主流方法对比

| 特性 | BPE | WordPiece | SentencePiece |
|---|---|---|---|
| 合并策略 | 最高频率对 | 最大似然增益 | BPE/Unigram 在句子片段上 |
| 预分词 | 需要 | 需要 | 不需要（直接处理原始文本） |
| 空格处理 | 保留 | `##` 标记续接 | `▁` 标记开头 |
| 使用模型 | GPT, LLaMA | BERT | T5, LLaMA, ChatGLM |

### 📖 BPE vs Unigram

| 特性 | BPE | Unigram |
|---|---|---|
| 方向 | **自底向上**（合并） | **自顶向下**（裁剪） |
| 初始化 | 字符 | 大词表 |
| 过程 | 逐步合并 | 逐步删除 |
| 分词 | 确定性 | 概率性（可采样） |

### 📖 中文 LLM 分词的特殊挑战

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

### 🪤 追问

- **Q：BPE 一个词元（token）= 几个汉字？**
  A：英文 LLM 的中文 tokenization 经常 1 个汉字对应 2-3 个 token（按字节切）；中文优化模型基本 1 字 1 token。

---

## Q11 · MoE 混合专家

### 🎯 核心思想

把 FFN 替换成**多个并行的"专家"FFN**，每个 token 由一个 **router** 路由到 **Top-k 个**专家（通常 k=1 或 2）。

```
            ┌─→ Expert 1 ─┐
input ──► Router ─→ Expert 2 ─┼─→ 加权求和
            └─→ Expert N ─┘
```

### 📖 优势

- **参数量大但激活量小**：万亿级模型推理时只激活几十亿参数
- 不同专家可学不同模式

### 📖 关键挑战

- **负载均衡**：避免所有 token 都去同一个专家 → 加 **load balancing loss**
- **训练不稳定**：router 决策是离散的
- **通信开销**：专家分布在不同 GPU 上，all-to-all 通信

### 📖 代表模型

- **Switch Transformer**：早期纯 Top-1 MoE
- **Mixtral 8x7B**：8 专家 Top-2，激活 ~13B 推理
- **DeepSeek-V2/V3**：细粒度专家 + 共享专家
- **Qwen3.5**：MoE + 线性注意力混合

### 🪤 追问

- **Q：MoE 显存怎么算？**
  A：训练时所有专家都得在显存里（**总参数量**）；推理时也都得在（除非 expert offloading），但只算激活的那 k 个，所以 FLOPs 小。

---

[⬅ 回到首页](../README.md)
