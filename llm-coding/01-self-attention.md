# 手撕 · Self-Attention from scratch

> 难度：⭐⭐ · 预计时间：10 分钟

面试现场最常被点名手撕的代码之一。能在白板/编辑器上 10 分钟内不查资料写出来即合格。

## 完整可运行版本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    单头 self-attention.
    输入: x (batch, seq_len, d_model)
    输出: out (batch, seq_len, d_v), attn (batch, seq_len, seq_len)
    """
    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        Q = self.W_Q(x)                                   # (B, N, d_k)
        K = self.W_K(x)
        V = self.W_V(x)

        # (B, N, N)，缩放点积
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            # mask 形状广播到 scores: (B, N, N) 或 (1, N, N)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)                  # (B, N, N)
        out = torch.matmul(attn, V)                       # (B, N, d_v)
        return out, attn


# ===== 自测 =====
if __name__ == "__main__":
    torch.manual_seed(0)
    model = SelfAttention(d_model=64, d_k=32, d_v=32)
    x = torch.randn(2, 10, 64)                            # (B=2, N=10, d=64)
    out, attn = model(x)
    print("out:", out.shape)                              # (2, 10, 32)
    print("attn:", attn.shape)                            # (2, 10, 10)
    print("attn 行和应为 1:", attn.sum(dim=-1)[0, 0].item())
```

## 易错点 / 面试官最爱抓的细节

| # | 错误 | 正确 |
|---|---|---|
| 1 | `mask_fill` | `masked_fill`（带 `ed`） |
| 2 | 缩放写成 `/d_k`（除以 dim） | 应是 `/sqrt(d_k)` |
| 3 | mask 填 `0` | 应填 `float('-inf')`（softmax 前） |
| 4 | softmax 写成 `dim=0` 或 `dim=1` | 应 `dim=-1`（沿 key 维度归一化） |
| 5 | 忘了 `K.transpose(-2, -1)` | `Q @ K^T`，K 必须转置 |
| 6 | bias 默认 True | 通常 `bias=False`（标准实现） |

## 进阶：加 dropout / 改成 Causal

```python
# Causal mask (用于 decoder)
seq_len = x.size(1)
causal = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
out, _ = model(x, mask=causal)
```

```python
# 在 attn 上加 dropout（标准位置）
attn = F.softmax(scores, dim=-1)
attn = F.dropout(attn, p=0.1, training=self.training)  # ← 加在这里
out = torch.matmul(attn, V)
```

## 相关

- 原理篇：[Q01 · Self-Attention 自注意力](../docs/01-architecture/Q01-self-attention.md)
- 多头版本：[手撕 Multi-Head Attention](02-multi-head-attention.md)
