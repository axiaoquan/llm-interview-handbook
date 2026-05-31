# 03 · Normalization 手撕

LayerNorm / RMSNorm / BatchNorm，手撕 + 对比。

## 📑 本章目录

- [Q01 · LayerNorm](#q01--layernorm)
- [Q02 · RMSNorm](#q02--rmsnorm)
- [Q03 · BatchNorm（对比）](#q03--batchnorm对比)
- [Q04 · Pre-Norm vs Post-Norm](#q04--pre-norm-vs-post-norm)

---

## Q01 · LayerNorm

### 🎯 目标

对**每个样本的最后一维（特征维）**做归一化：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

### 🛠 代码

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # 缩放
        self.beta = nn.Parameter(torch.zeros(d_model))   # 偏移
        self.eps = eps

    def forward(self, x):
        """x: [..., d_model]"""
        mean = x.mean(dim=-1, keepdim=True)                          # [..., 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)            # [..., 1]
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta
```

### 🪤 易错点

1. **`unbiased=False`**：用有偏方差（除以 n 而不是 n-1），跟 PyTorch 官方实现一致
2. **`keepdim=True`**：保留维度方便广播，否则形状对不上
3. **gamma 初始化为 1，beta 初始化为 0**：初始时 LN 等价恒等映射
4. **dim=-1**：永远在最后一维（特征维），不要写 `dim=0`

---

## Q02 · RMSNorm

### 🎯 目标

LayerNorm 的简化版：去掉减均值、去掉 beta，只用 RMS 归一化。**LLaMA 全系用的就是这个**。

$$
y = \frac{x}{\mathrm{RMS}(x)} \cdot \gamma, \quad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}
$$

### 🛠 代码

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        """x: [..., d_model]"""
        # rsqrt = 1 / sqrt，PyTorch 内置更快更稳
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_hat = x * torch.rsqrt(rms + self.eps)
        return x_hat * self.gamma
```

### 🪤 易错点

- **没有 beta**：只学一个 gamma，参数量减半
- **不减均值**：直接除 RMS。论文实证：均值这一步对效果影响小，但占 1/3 计算量
- **`rsqrt` 比 `1/sqrt` 更快**：PyTorch 底层 fused kernel
- **混合精度**：归一化里要把 x 转 float32 算（x = x.float()），算完再转回 bf16，否则会数值不稳

```python
def forward(self, x):
    dtype = x.dtype
    x = x.float()                                           # ← 必须先升精度
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(rms + self.eps)
    return (x_hat * self.gamma).to(dtype)                   # ← 算完转回去
```

---

## Q03 · BatchNorm（对比）

### 🎯 目标

对**每个特征**在 batch 维度上归一化（CV 常用，NLP 不用）：

$$
y_j = \frac{x_j - \mu_j^{(\text{batch})}}{\sqrt{\sigma_j^{(\text{batch})2} + \epsilon}} \cdot \gamma_j + \beta_j
$$

### 🛠 代码

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # 推理时用的 running statistics（不参与梯度）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        """x: [B, C]"""
        if self.training:
            mean = x.mean(dim=0)                            # [C]
            var = x.var(dim=0, unbiased=False)              # [C]
            # 更新 running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta
```

### 📊 三种 Norm 对比

| 维度 | LayerNorm | RMSNorm | BatchNorm |
|---|---|---|---|
| 归一化轴 | 特征维（每个样本独立） | 特征维 | batch 维（每个特征独立） |
| 训练/推理一致 | ✅ | ✅ | ❌（推理用 running stats） |
| 依赖 batch size | ❌ | ❌ | ✅（小 batch 退化） |
| 参数量 | 2d（γ + β） | d（只 γ） | 2d（γ + β + running buffers） |
| LLM 用法 | 早期（GPT-2） | 主流（LLaMA/Qwen） | 不用 |

### 🪤 NLP 为什么不用 BatchNorm

- **变长序列**：不同样本长度不同，batch 维度统计不稳定
- **batch 间相关性**：训练/推理 batch 分布差异大，running stats 失效
- **长 padding**：pad token 会污染统计量

---

## Q04 · Pre-Norm vs Post-Norm

### 🎯 区别

```python
# Post-Norm（原 Transformer，2017）
def post_norm_block(x):
    x = LN(x + Attention(x))
    x = LN(x + FFN(x))
    return x

# Pre-Norm（GPT-2 之后主流）
def pre_norm_block(x):
    x = x + Attention(LN(x))
    x = x + FFN(LN(x))
    return x
```

### 📊 对比

| 维度 | Post-Norm | Pre-Norm |
|---|---|---|
| 训练稳定性 | 差（深层易梯度消失） | 好 |
| 收敛速度 | 慢（需要 warmup） | 快 |
| 最终精度 | 略高（如果训得动） | 略低 |
| 现代 LLM | ❌ | ✅ 全用 |

### 🪤 易错点

- **为什么 Pre-Norm 更稳**：残差路径上的 norm 不会缩小信号，梯度能畅通流到底层
- **Post-Norm 必须 warmup**：不然初期梯度爆炸 / 消失
- **Sandwich-Norm**：Pre-Norm + 输出再加一个 LN（部分大模型用，更稳）

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
