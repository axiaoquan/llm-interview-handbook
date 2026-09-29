# 01 · Attention 手撕

Attention 全家桶，从最基础到生产级。

## 本章目录

- [Q01 · Scaled Dot-Product Attention](#q01--scaled-dot-product-attention)
- [Q02 · Multi-Head Attention](#q02--multi-head-attention)
- [Q03 · Causal (Masked) Attention](#q03--causal-masked-attention)
- [Q04 · Grouped-Query Attention (GQA)](#q04--grouped-query-attention-gqa)
- [Q05 · KV Cache 推理加速](#q05--kv-cache-推理加速)

---

## Q01 · Scaled Dot-Product Attention

### 目标
实现最核心的 attention：$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

### 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, training=False):
    """
    Q, K, V: [B, ..., L, d]
    mask: 可广播到 [B, ..., L_q, L_k]；bool True 保留，float 直接加到分数。
          浮点屏蔽必须用 -inf，不能把浮点 0/1 当布尔 mask。
    全遮挡 query 的输出约定为 0；Q/K/V 及未遮挡分数需有限。
    """
    d_k = Q.size(-1)
    # [B, ..., L_q, L_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float('-inf'))
        elif mask.is_floating_point():
            scores = scores + mask
        else:
            raise TypeError("mask must be bool or floating additive bias")

    # 先替换全 -inf 行，避免 softmax 生成 NaN，再将其权重置零。
    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
    safe_scores = scores.masked_fill(empty, 0)
    attn = F.softmax(safe_scores, dim=-1).masked_fill(empty, 0)
    attn = F.dropout(attn, p=dropout_p, training=training)

    out = torch.matmul(attn, V)                    # [B, ..., L_q, d]
    return out, attn
```

### 易错点

- **为什么除 $\sqrt{d_k}$**：若 $q_j,k_j$ 独立、零均值、单位方差，则 $q^\top k$ 的方差是 $d_k$，标准差是 $\sqrt{d_k}$。除以标准差后 logit 的方差回到约 1，避免 softmax 随 head dimension 增大而饱和
- **为什么不是除以 $d_k$**：缩放随机变量 $z$ 时，$\mathrm{Var}(z/c)=\mathrm{Var}(z)/c^2$。取 $c=\sqrt{d_k}$ 才得到单位方差；取 $c=d_k$ 会让方差降到 $1/d_k$，注意力越来越接近均匀。这个结论依赖近似分布假设，QK-Norm 或可学习 temperature 可以处理实际尺度偏离
- **`masked_fill` 不是 `mask_fill`**（少一个 d）
- **mask 用 `-inf` 还是 `0`**：填 `-inf`，softmax 后变 0；直接乘 0 会导致归一化错位
- **softmax 的 `dim`**：必须是 `-1`（key 维度），归一化"每个 query 对所有 key 的注意力分布"
- **整行都被 mask**：普通 softmax 会得到 NaN；本例显式返回零权重和零输出。padding query 还应从下游 loss 中排除，零 attention 不意味着整个残差块输出为零
- **dropout 的模式**：函数不能读取外部模块的 `eval()`；调用者必须传 `training=self.training`。官方 SDPA 则需在 eval 时显式传 `dropout_p=0.0`

---

## Q02 · Multi-Head Attention

### 目标
把 d_model 切成 h 个头并行算 attention，再 concat 回 d_model。

### 代码

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, mask=None):
        B, L_q, _ = x_q.shape
        L_k = x_k.size(1)

        # 1) 线性变换 + 切头：[B, L, d_model] → [B, h, L, d_k]
        Q = self.W_q(x_q).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_k).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x_v).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)

        # 2) 计算 attention：[B, h, L_q, d_k]
        # 复用 Q01，统一 mask、全遮挡行与 dropout 语义。
        out, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout.p, training=self.training
        )

        # 3) 合头 + 输出投影：[B, L_q, d_model]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        return self.W_o(out)
```

### 形状变化（必背！）

```
x_q:   [B, L_q, d_model]
W_q(x_q):                                 [B, L_q, d_model]
.view(B, L_q, h, d_k):                    [B, L_q, h, d_k]
.transpose(1, 2):                         [B, h, L_q, d_k]   ← 头维度提到第二维

scores = Q @ K^T:                         [B, h, L_q, L_k]
attn = softmax(scores) @ V:               [B, h, L_q, d_k]

.transpose(1, 2):                         [B, L_q, h, d_k]
.contiguous().view(B, L_q, d_model):      [B, L_q, d_model]  ← 必须 contiguous 否则 view 报错
```

### 易错点

1. **`.transpose` 后必须 `.contiguous()` 才能 `.view`**——否则报错，因为 transpose 不改 storage
2. **mask 的形状要能广播**：`[B, 1, L_q, L_k]` → 复制到所有头
3. **bias 不是原则问题**：不同架构有的使用、有的不使用；面试实现可设 `bias=False`，但不要把它说成 Attention 的数学要求
4. **W_o 的作用**：把各头拼接后的子空间重新混合回 residual stream；理论上可设计无 $W_o$ 的变体，但那会限制头间线性组合，并非标准 MHA

### 为什么多头比一个宽头更有表达力？

一个 softmax 对每个 query 只产生一套对 key 的归一化权重。多头先在不同 Q/K 投影空间内产生多套权重，再分别聚合 V，因而同一 token 可以同时保留多种路由模式。总投影维度固定时，主矩阵参数量与宽单头同阶；代价是更多 softmax、重排和 kernel 开销。多头只提供这种表示能力，**不保证**每个头都会自动对应可解释的语法或位置功能。

---

## Q03 · Causal (Masked) Attention

### 目标
LLM decoder 必须的：第 i 个位置只能看到 0..i，不能偷看未来。

### 代码（生成 mask）

```python
def causal_mask(L: int, device='cpu'):
    """
    返回 [L, L] 的下三角 mask，True 表示保留。
    mask[i, j] = True iff j <= i
    """
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

# 用法：在 attention 里
mask = causal_mask(L)                       # [L, L]
scores = scores.masked_fill(~mask, float('-inf'))
```

### 高效实现：用 PyTorch 内置

```python
# PyTorch 2.0+ 内置了 causal flag
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
# 根据设备、dtype、shape 等选择 Flash/高效/数学后端，不保证总用 Flash。
```

### 易错点

- 单 token decode 仅当所有 key 都是有效的过去/当前 token 时可省 causal mask；padding 仍需屏蔽。多 token prefill/chunked decode 必须保留因果约束；不等长 Q/K 不能盲目套用普通左上对齐三角 mask
- **不要把 padding mask 跟 causal mask 搞混**：padding mask 是把 `<pad>` 位置屏蔽，causal mask 是屏蔽未来；两个要**逻辑与**结合

```python
# 同时做 causal + padding mask
causal = torch.tril(torch.ones(L, L, dtype=torch.bool))   # [L, L]
pad = (input_ids != pad_id).unsqueeze(1).unsqueeze(2)     # [B, 1, 1, L]
combined = causal & pad                                    # [B, 1, L, L] 通过广播
```

---

## Q04 · Grouped-Query Attention (GQA)

### 目标
多个 Q 头共享一组 KV 头（介于 MHA 和 MQA 之间），省 KV Cache 显存。

### 代码

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        assert d_model % n_q_heads == 0, "d_model must be divisible by n_q_heads"
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads     # 每个 KV 头被几个 Q 头共享
        self.d_k = d_model // n_q_heads

        self.W_q = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)   # ← K 头变少
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_q_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.n_q_heads,  self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 关键：把 KV 头复制 n_rep 次以对齐 Q 头数（repeat_interleave on head dim）
        K = K.repeat_interleave(self.n_rep, dim=1)   # [B, n_q, L, d_k]
        V = V.repeat_interleave(self.n_rep, dim=1)

        out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)
```

### 易错点

- **`repeat_interleave` vs `repeat`**：前者是"AAABBB"，后者是"ABCABC"。GQA 要前者（保持组内 Q 头连续）
- **KV Cache 显存节省**：从 `n_q_heads × L × d_k` 降到 `n_kv_heads × L × d_k`，比例 = `n_kv_heads / n_q_heads`
- LLaMA-2-70B：n_q=64, n_kv=8 → KV Cache 省 **8×**

---

## Q05 · KV Cache 推理加速

### 目标
推理时只对**新 token** 计算 Q，复用之前的 K、V，避免重复算 prefix。

### 代码

```python
class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None, key_padding_mask=None):
        """
        x: [B, L_new, d_model]
            训练 / prefill 阶段：L_new = 完整序列长度
            decode 阶段：       L_new = 1
        kv_cache: dict 或 None
            { 'K': [B, h, L_past, d_k], 'V': [B, h, L_past, d_k] }
        key_padding_mask: bool [B, L_past + L_new]，True 表示有效 key。
        本例不含位置编码；若加 RoPE，须用 cache offset 旋转新 Q/K。
        """
        B, L_new, _ = x.shape
        if kv_cache is not None and (('K' in kv_cache) != ('V' in kv_cache)):
            raise ValueError("cache must contain both K and V")

        Q = self.W_q(x).view(B, L_new, self.n_heads, self.d_k).transpose(1, 2)
        K_new = self.W_k(x).view(B, L_new, self.n_heads, self.d_k).transpose(1, 2)
        V_new = self.W_v(x).view(B, L_new, self.n_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None and 'K' in kv_cache:
            # 沿 L 维拼接旧 KV 和新 KV
            K = torch.cat([kv_cache['K'], K_new], dim=2)   # [B, h, L_past + L_new, d_k]
            V = torch.cat([kv_cache['V'], V_new], dim=2)
        else:
            K, V = K_new, V_new

        # 更新 cache（in-place）
        if kv_cache is not None:
            kv_cache['K'] = K
            kv_cache['V'] = V

        L_total = K.size(2)
        L_past = L_total - L_new
        query_pos = L_past + torch.arange(L_new, device=x.device)
        key_pos = torch.arange(L_total, device=x.device)
        allowed = key_pos[None, :] <= query_pos[:, None]  # [L_new, L_total]
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (B, L_total):
                raise ValueError("key_padding_mask must be bool [B, L_total]")
            allowed = allowed & key_padding_mask[:, None, None, :]
        out, _ = scaled_dot_product_attention(Q, K, V, mask=allowed)

        out = out.transpose(1, 2).contiguous().view(B, L_new, -1)
        return self.W_o(out), kv_cache
```

### 推理循环示例

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=50, eos_id=None, pad_id=None):
    """教学接口：model(ids, kv_cache) 返回 (logits, cache)。prompt 等长且无 padding。"""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if max_new_tokens == 0:
        return prompt_ids
    if prompt_ids.size(1) == 0:
        raise ValueError("prompt must be nonempty")
    kv_cache = {}
    finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device)
    fill_id = eos_id if pad_id is None else pad_id
    out_ids = [prompt_ids]
    current = prompt_ids
    for _ in range(max_new_tokens):
        # 首次 prefill，之后仅传新 token；每层 cache 应由 model 管理。
        logits, kv_cache = model(current, kv_cache=kv_cache)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        if eos_id is not None:
            next_id = torch.where(finished[:, None], fill_id, next_id)
            finished |= next_id.squeeze(-1).eq(eos_id)
        out_ids.append(next_id)
        if eos_id is not None and finished.all():
            break
        current = next_id
    return torch.cat(out_ids, dim=1)
```

### 易错点

- **位置编码要小心**：用 RoPE 时，新 token 的位置 = `L_past + i`，不能从 0 重新编
- **显存增长**：KV Cache 大小 = `2 × n_layers × n_kv_heads × L × d_k × dtype`，长上下文很恐怖（GQA / MQA / PagedAttention 就是为了解决这个）
- **batch 的结束状态**：本例保留 finished 行并填 pad/EOS，逻辑正确但仍有无效计算；生产系统可压缩活跃 batch，配套重排 cache。有 padding 的 prompt 还需完整 attention mask 与 position_ids，不能直接使用本例生成接口

生成前调用 `model.eval()`；`no_grad()` 只关闭梯度，不会关闭 dropout。KV cache 是推理状态，不应跨独立样本或训练 batch 复用。底层接口参考 [PyTorch SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)。

### 性能对比

```
没有 KV Cache：每一步重算整个 prefix 的投影、FFN 和 attention
有 KV Cache：旧 token 的 K/V 不再重算；单步 attention 仍需让新 query 读取长度 L 的 cache

因此 attention 部分从每步 O(L²d) 降为 O(Ld)，但端到端还包含 O(d²) 的投影/FFN、显存读取和调度开销。实际加速取决于模型宽度、batch、上下文长度和硬件，不能直接把理论降阶说成“长度为 1000 就加速 1000 倍”。
```

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
