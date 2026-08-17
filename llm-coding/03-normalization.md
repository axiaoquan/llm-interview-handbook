# 03 · Normalization 手撕

LayerNorm / RMSNorm / BatchNorm，手撕 + 对比。

## 本章目录

- [Q01 · LayerNorm](#q01--layernorm)
- [Q02 · RMSNorm](#q02--rmsnorm)
- [Q03 · BatchNorm（对比）](#q03--batchnorm对比)
- [Q04 · Pre-Norm vs Post-Norm](#q04--pre-norm-vs-post-norm)

---

## Q01 · LayerNorm

### 目标

对**每个样本的最后一维（特征维）**做归一化：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

### 代码

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

### 易错点

1. **`unbiased=False`**：用有偏方差（除以 n 而不是 n-1），跟 PyTorch 官方实现一致
2. **`keepdim=True`**：保留维度方便广播，否则形状对不上
3. **gamma 初始化为 1，beta 初始化为 0**：初始时不再额外缩放或平移标准化结果；LN 仍会减均值、除标准差，因此并不是恒等映射
4. **dim=-1**：永远在最后一维（特征维），不要写 `dim=0`

### $\gamma、\beta$ 为什么可学习？

- 均值和方差来自当前样本，是统计量，不是模型参数
- $\gamma$ 和 $\beta$ 是逐特征的仿射参数，形状通常为 `[d_model]`
- 标准化统一了每个 token 的整体尺度，但不同通道未必都应保持单位方差、零中心；$\gamma、\beta$ 让后续网络重新选择各通道的幅值和基线
- 它们不“撤销”标准化，因为被标准化前的样本均值和方差已经丢失；它们恢复的是可学习的全局通道尺度，而不是每个样本原来的尺度

---

## Q02 · RMSNorm

### 目标

LayerNorm 的简化版：去掉减均值、常见实现也去掉 beta，只用 RMS 归一化。LLaMA 等许多现代 LLM 使用它。

$$
y = \frac{x}{\mathrm{RMS}(x)} \cdot \gamma, \quad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}
$$

### 代码

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

### 易错点

- **没有 beta**：只学一个 gamma，参数量减半
- **不减均值**：直接除 RMS，省去均值中心化；实际加速取决于是否有 fused kernel 和内存访问，不能固定说省某个百分比
- **`rsqrt` 比 `1/sqrt` 更快**：PyTorch 底层 fused kernel
- **混合精度**：平方和/均值等归约通常用 FP32 累积更稳；手写实现可显式升精度，生产 fused kernel 也可能内部完成，因此不是所有代码都必须先 `x.float()`

```python
def forward(self, x):
    dtype = x.dtype
    x = x.float()                                           # 手写版显式用 FP32 归约
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(rms + self.eps)
    return (x_hat * self.gamma).to(dtype)                   # ← 算完转回去
```

RMSNorm 保留输入方向并只控制向量的 RMS 尺度，具有重缩放不变性；它没有显式平移不变性。省去 $\beta$ 是常见架构选择而非数学强制，有些实现也可以额外加入 bias。

---

## Q03 · BatchNorm（对比）

### 目标

对**每个特征**在 batch 维度上归一化（CV 常用，序列模型较少使用）：

$$
y_j = \frac{x_j - \mu_j^{(\text{batch})}}{\sqrt{\sigma_j^{(\text{batch})2} + \epsilon}} \cdot \gamma_j + \beta_j
$$

### 代码

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

### 哪些量可学习，哪些不可学习？

| 量 | 是否梯度学习 | 作用 |
|---|---|---|
| $\gamma_j$ | 是 | 恢复或抑制第 $j$ 个通道的幅值 |
| $\beta_j$ | 是 | 调整第 $j$ 个通道的基线 |
| batch mean / variance | 否 | 训练当前 batch 的即时统计量 |
| running mean / variance | 否，属于 buffer | 指数滑动更新，推理时替代 batch 统计量 |

为什么标准化后还要 $\gamma、\beta$？若强制每层输出永远为零均值、单位方差，会限制后续层需要的表示尺度；例如 $\gamma_j=0$ 可以关闭某个通道，较大的 $\gamma_j$ 可以放大有用特征。仿射参数让 BN 获得稳定统计的同时不丢掉逐通道重参数化能力。

上面的实现用于讲清机制，和 PyTorch 仍有一个细节差异：PyTorch 训练前向归一化使用有偏方差，而写入 `running_var` 时使用无偏估计；生产代码应直接使用 `nn.BatchNorm1d`。

### 三种 Norm 对比

| 维度 | LayerNorm | RMSNorm | BatchNorm |
|---|---|---|---|
| 归一化轴 | 特征维（每个样本独立） | 特征维 | 每通道跨 batch；3D BN1d 还跨长度轴 |
| 训练/推理一致 | 是 | 是 | 否（推理用 running stats） |
| 依赖 batch size | 否 | 否 | 是（小 batch 统计噪声大） |
| 可学习参数量 | 2d（γ + β） | d（只 γ） | 2d（γ + β）；running stats 是 buffer，不计入参数 |
| LLM 用法 | 广泛 | 许多现代架构采用 | 很少用于主干 |

### NLP 为什么通常不用 BatchNorm

- BN 会把不同样本、甚至不同 token 的同一通道耦合起来，统计量依赖 micro-batch 大小和序列组成；分布式切 batch 后还可能需要同步统计
- 变长序列和 padding 可以用 mask 处理，但会让有效样本数随位置变化，统计与实现都更复杂
- 训练使用 batch stats、推理使用 running stats；自回归 decode 的 batch/长度分布与训练差异尤其明显
- LN/RMSNorm 对每个 token 独立，训练与推理路径一致，所以更自然。结论是“通常不选 BN”，不是“序列上绝对不能使用 BN”

---

## Q04 · Pre-Norm vs Post-Norm

### 区别

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

### 对比

| 维度 | Post-Norm | Pre-Norm |
|---|---|---|
| 训练稳定性 | 差（深层易梯度消失） | 好 |
| 收敛速度 | 慢（需要 warmup） | 快 |
| 最终精度 | 略高（如果训得动） | 略低 |
| 现代 LLM | 少见 | 主流，但并非唯一选择 |

### 易错点

- **为什么 Pre-Norm 更稳**：存在不经过子层和 norm 的恒等残差支路，反向梯度包含直接项，深层连乘更不容易衰减
- **Post-Norm 通常更依赖 warmup、初始化或残差缩放**：不是逻辑上“必须”，而是深层训练更敏感
- **Sandwich-Norm**：Pre-Norm + 输出再加一个 LN（部分大模型用，更稳）

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
