# 02 · Position Encoding 手撕

位置编码三大主流方案：Sinusoidal（原 Transformer）、RoPE（LLaMA/Qwen）、ALiBi（BLOOM）。

## 📑 本章目录

- [Q01 · Sinusoidal Position Encoding](#q01--sinusoidal-position-encoding)
- [Q02 · RoPE (Rotary Position Embedding)](#q02--rope-rotary-position-embedding)
- [Q03 · ALiBi (Attention with Linear Biases)](#q03--alibi-attention-with-linear-biases)

---

## Q01 · Sinusoidal Position Encoding

### 🎯 目标

原 Transformer 论文里的固定位置编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

### 🛠 代码

```python
import torch
import math

def sinusoidal_pe(max_len: int, d_model: int):
    """返回 [max_len, d_model]"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()    # [L, 1]
    # div_term = 10000^(2i/d) 的倒数，等价于 exp(-(2i/d) * log(10000))
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )                                                            # [d/2]
    pe[:, 0::2] = torch.sin(position * div_term)                 # 偶数维 sin
    pe[:, 1::2] = torch.cos(position * div_term)                 # 奇数维 cos
    return pe

# 用法：直接加到 token embedding 上
x = token_emb + sinusoidal_pe(L, d_model)[:L].to(x.device)
```

### 🪤 易错点

- **数值稳定**：直接算 `10000^(2i/d)` 大 d 时溢出，用 `exp(log)` 形式
- **不同维度频率不同**：低维变化快、高维变化慢，让模型同时感知短距离和长距离
- **加 vs 拼**：原论文是**直接加**到 embedding 上（共享 d_model），不是 concat

---

## Q02 · RoPE (Rotary Position Embedding)

### 🎯 目标

把位置 m 的 query/key 通过**旋转矩阵**编码：

$$
\tilde{q}_m = q_m \cdot \cos(m\theta) + \text{rotate\\_half}(q_m) \cdot \sin(m\theta)
$$

旋转后，attention 的内积 $\tilde{q}_m^T \tilde{k}_n$ **只跟相对位置 (m-n) 有关**。

### 🛠 代码

```python
import torch

def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0):
    """
    预计算 cos/sin 缓存。
    dim: 每个头的维度 d_k（必须偶数）
    返回: cos, sin 都是 [max_len, dim]
    """
    # θ_i = 1 / base^(2i/d)，i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))   # [d/2]
    t = torch.arange(max_len).float()                                    # [L]
    freqs = torch.outer(t, inv_freq)                                     # [L, d/2]
    # 复制成 [L, d]，让 (cos, cos, cos, ...) 和 (sin, sin, sin, ...) 长度匹配
    emb = torch.cat([freqs, freqs], dim=-1)                              # [L, d]
    return emb.cos(), emb.sin()


def rotate_half(x):
    """把 x 后一半拿出来取负，跟前一半交换：[a, b] → [-b, a]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin, position_ids=None):
    """
    q, k: [B, h, L, d]
    cos, sin: [max_L, d]
    position_ids: [B, L] 可选（推理 KV Cache 时新 token 用真实位置）
    """
    if position_ids is None:
        cos_ = cos[: q.size(-2)]                  # [L, d]
        sin_ = sin[: q.size(-2)]
    else:
        cos_ = cos[position_ids]                  # [B, L, d]
        sin_ = sin[position_ids]
    # 广播到 head 维：[..., L, d] → [B, 1, L, d]
    cos_ = cos_.unsqueeze(-3) if cos_.dim() == 3 else cos_.unsqueeze(0).unsqueeze(0)
    sin_ = sin_.unsqueeze(-3) if sin_.dim() == 3 else sin_.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos_) + (rotate_half(q) * sin_)
    k_embed = (k * cos_) + (rotate_half(k) * sin_)
    return q_embed, k_embed


# === 用法 ===
# 在每层 attention 里，对 Q 和 K（不对 V！）应用：
# cos, sin = precompute_rope_freqs(d_k, max_len)
# q, k = apply_rope(q, k, cos, sin)
# 然后正常算 attention
```

### 🪤 易错点

1. **只对 Q、K 做 RoPE，不对 V**——V 不参与位置敏感的内积
2. **`rotate_half` 不是 element-wise 反**：是把后半段移到前面并取负
3. **推理用 KV Cache 时**：新 token 的 cos/sin 要取**真实位置** `L_past + i`，不能从 0 重新算
4. **外推能力**：直接超过训练长度会崩，需要 NTK-aware / YaRN 调整 base
5. **base 选择**：默认 10000，长上下文模型用 500000 或 1000000（LLaMA-3.1）

### 📊 形状追踪

```
q:                          [B, h, L, d_k]
cos[:L]:                    [L, d_k]   →  unsqueeze →  [1, 1, L, d_k]
q * cos:                    [B, h, L, d_k]   ← broadcast
rotate_half(q) * sin:       [B, h, L, d_k]
q_embed:                    [B, h, L, d_k]
```

---

## Q03 · ALiBi (Attention with Linear Biases)

### 🎯 目标

不改 Q、K，直接在 attention 分数上**加一个线性偏置**：距离越远扣分越多。

$$
\mathrm{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|
$$

每个头用不同的斜率 $m$（几何级数）。

### 🛠 代码

```python
def get_alibi_slopes(n_heads: int):
    """Geometric series for ALiBi slopes."""
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start * start ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    # 不是 2 的幂时，混合两组
    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    return (
        get_slopes_power_of_2(closest_pow2)
        + get_slopes_power_of_2(2 * closest_pow2)[0::2][:n_heads - closest_pow2]
    )


def alibi_bias(n_heads: int, L: int, device='cpu'):
    """
    返回 [n_heads, L, L] 的偏置矩阵
    bias[h, i, j] = -slope[h] * |i - j|
    """
    slopes = torch.tensor(get_alibi_slopes(n_heads), device=device)        # [h]
    pos = torch.arange(L, device=device)
    rel = (pos[None, :] - pos[:, None]).abs().float()                      # [L, L]
    bias = -slopes.view(-1, 1, 1) * rel.unsqueeze(0)                       # [h, L, L]
    return bias


# === 用在 attention 里 ===
# scores: [B, h, L, L]
scores = scores + alibi_bias(n_heads, L, device=scores.device)
# 然后 softmax / mask 等正常流程
```

### 🪤 易错点

- ALiBi **不需要任何位置 embedding 加在输入上**，完全靠 attention bias
- **外推性**：训练 2K 上下文，推理 16K 也基本不退化（这是 ALiBi 的核心卖点）
- **斜率的几何级数**：`m_h = 2^(-8h/H)`，不同头有不同感知尺度

### 📊 三种位置编码对比

| 维度 | Sinusoidal | RoPE | ALiBi |
|---|---|---|---|
| 加在哪里 | token emb | Q/K | attention scores |
| 是否需要训练 | 否 | 否（只有缓存） | 否 |
| 外推能力 | 差 | 中（需 YaRN） | 强 |
| 计算开销 | 极低 | 低 | 极低 |
| 主流应用 | 原 Transformer | LLaMA/Qwen/Mistral | BLOOM/MPT |

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
