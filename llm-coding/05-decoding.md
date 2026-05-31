# 05 · Decoding 手撕

LLM 推理时的采样策略——决定生成质量、多样性、稳定性。

## 📑 本章目录

- [Q01 · Greedy / Argmax Decoding](#q01--greedy--argmax-decoding)
- [Q02 · Top-k Sampling](#q02--top-k-sampling)
- [Q03 · Top-p (Nucleus) Sampling](#q03--top-p-nucleus-sampling)
- [Q04 · Temperature Scaling](#q04--temperature-scaling)
- [Q05 · Beam Search](#q05--beam-search)
- [Q06 · Repetition Penalty](#q06--repetition-penalty)

---

## Q01 · Greedy / Argmax Decoding

### 🎯 目标

每步选 logits 最大的 token。最简单但最容易陷入重复。

### 🛠 代码

```python
import torch

@torch.no_grad()
def greedy_decode(model, input_ids, max_new_tokens=50, eos_id=None):
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]                  # [B, V]
        next_id = logits.argmax(dim=-1, keepdim=True)        # [B, 1]
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return input_ids
```

### 🪤 易错点

- **重复退化**：贪婪解码极易陷入 "the the the the"，**生产环境很少单独用**
- 适合：翻译、摘要等"答案唯一"的任务
- 不适合：开放对话、创作

---

## Q02 · Top-k Sampling

### 🎯 目标

只在 logits 排名前 k 的 token 里采样，剔除长尾噪声。

### 🛠 代码

```python
def top_k_logits(logits: torch.Tensor, k: int):
    """返回处理后的 logits（非 top-k 位置置 -inf）"""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k=k, dim=-1)              # [B, k]
    threshold = values[:, -1:]                                # [B, 1]，第 k 大的值
    return torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)


def sample_top_k(logits: torch.Tensor, k: int = 50):
    """从 top-k 中按概率采样"""
    logits = top_k_logits(logits, k)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)            # [B, 1]
```

### 🪤 易错点

- **k 选大了等于不过滤；选小了等于贪婪**：常用 k=40 ~ 50
- **置 -inf 比直接置 0 更稳**：softmax 后是真正的 0，不会污染分布
- **`torch.multinomial` 需要正概率**：上一步已经过 softmax 所以安全

---

## Q03 · Top-p (Nucleus) Sampling

### 🎯 目标

动态选择最小集合，使其累积概率 ≥ p。比 top-k 更自适应。

### 🛠 代码

```python
def top_p_logits(logits: torch.Tensor, p: float = 0.9):
    """剔除累积概率 > p 之外的 token"""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)              # [B, V]

    # 标记需要移除的位置：累积概率超过 p 的（最少保留 1 个）
    mask = cumulative > p
    mask[:, 1:] = mask[:, :-1].clone()                          # 右移一位，保留第一个超过的 token
    mask[:, 0] = False                                          # 第一个永远保留

    sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))

    # scatter 回原顺序
    logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def sample_top_p(logits: torch.Tensor, p: float = 0.9):
    logits = top_p_logits(logits, p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 🪤 易错点

1. **mask 右移一位**：保留"第一个让累积超过 p 的 token"，否则会少选一个
2. **mask[:, 0] = False**：第一个 token 必须保留（即使它本身概率 > p）
3. **scatter 回原顺序**：sort 后位置乱了，要用 scatter 把 -inf 放回原 logits 的对应位置

### 📊 Top-k vs Top-p 对比

```
logits 概率分布：[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]

top_k=2  → 只保留前 2：[0.6, 0.2]
top_p=0.85 → 保留累积 ≥ 0.85 的最小集：[0.6, 0.2, 0.1] = 0.9
```

**结论**：top-p 在分布尖锐时少选（更确定），分布平坦时多选（更多样），更智能。

---

## Q04 · Temperature Scaling

### 🎯 目标

控制分布的"陡峭度"：T 越大越平、越随机；T 越小越尖、越确定。

$$
P_i = \\frac{\\exp(\\text{logit}_i / T)}{\\sum_j \\exp(\\text{logit}_j / T)}
$$

### 🛠 代码

```python
def apply_temperature(logits: torch.Tensor, temperature: float):
    if temperature <= 0:
        # T=0 等价 greedy
        return torch.zeros_like(logits).scatter_(
            -1, logits.argmax(dim=-1, keepdim=True), 1.0
        ).log()    # 这种实现稍丑，实际生产里 T=0 走 greedy 分支即可
    return logits / temperature


# 完整采样函数（组合所有策略）
@torch.no_grad()
def sample(model, input_ids, max_new_tokens=50,
           temperature=1.0, top_k=50, top_p=0.9, eos_id=None):
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]
        # 1) 温度
        logits = logits / temperature
        # 2) top-k
        if top_k > 0:
            logits = top_k_logits(logits, top_k)
        # 3) top-p
        if top_p < 1.0:
            logits = top_p_logits(logits, top_p)
        # 4) 采样
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return input_ids
```

### 🪤 经验值

| 任务 | 推荐 |
|---|---|
| 代码生成 | T=0.2, top_p=0.95 |
| 创意写作 | T=0.9, top_p=0.95 |
| 严肃问答 | T=0.3, top_k=40 |
| 大开脑洞 | T=1.2, top_p=0.99 |

---

## Q05 · Beam Search

### 🎯 目标

每步保留 logp 累积最高的 k 条候选路径，最后输出最高分序列。**确定性强，多样性差**。

### 🛠 代码

```python
@torch.no_grad()
def beam_search(model, input_ids, beam_size=4, max_new_tokens=50, eos_id=None):
    """
    input_ids: [1, L] (batch=1)
    返回: 最佳序列 [1, L+T]
    """
    B, L = input_ids.shape
    assert B == 1, "只支持 batch=1，多 batch 实现更复杂"

    # 初始化 beam：[(seq, log_prob, finished)]
    beams = [(input_ids, 0.0, False)]
    finished_beams = []

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score, done in beams:
            if done:
                candidates.append((seq, score, True))
                continue
            logits = model(seq)[:, -1, :]              # [1, V]
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)   # [V]
            # 取 top beam_size 个候选
            top_log_probs, top_ids = log_probs.topk(beam_size)
            for lp, tid in zip(top_log_probs, top_ids):
                new_seq = torch.cat([seq, tid.view(1, 1)], dim=-1)
                new_score = score + lp.item()
                is_done = (eos_id is not None and tid.item() == eos_id)
                candidates.append((new_seq, new_score, is_done))

        # 选 top beam_size 个（length normalization 可选）
        candidates.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
        beams = candidates[:beam_size]

        # 全部 beam 都完成则提前停止
        if all(b[2] for b in beams):
            break

    # 返回最佳序列
    beams.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
    return beams[0][0]
```

### 🪤 易错点

1. **长度归一化**：直接累积 logp 会偏向短序列（每个 logp 都是负数），所以除以长度
2. **EOS 处理**：beam 命中 EOS 后要"冻结"，不再扩展，但保留它参与最终排名
3. **不能跟 sampling 混用**：beam search 是确定性的，跟温度、top-p 互斥
4. **显存爆炸**：每步要并发跑 beam_size 个序列的前向，相当于 batch_size × beam_size

### 📊 Beam Search vs Sampling

| | Beam Search | Sampling |
|---|---|---|
| 确定性 | 高 | 低 |
| 多样性 | 差 | 好 |
| 适合 | 翻译、摘要 | 对话、创作 |
| 重复倾向 | 严重（解决方案：no_repeat_ngram） | 一般 |

---

## Q06 · Repetition Penalty

### 🎯 目标

惩罚已经生成过的 token，避免"the the the"。

### 🛠 代码

```python
def repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.2):
    """
    已经出现过的 token 的 logit 除以 penalty（>0）或乘以 penalty（<0）
    HuggingFace 的实现：正 logit 除以 penalty，负 logit 乘以 penalty（让它更负）
    """
    score = torch.gather(logits, 1, input_ids)          # [B, L_history]
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits


# 用法：在采样前应用
@torch.no_grad()
def sample_with_rep_penalty(model, input_ids, ..., rep_penalty=1.2):
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1:, :]
        logits = repetition_penalty(logits.squeeze(1), input_ids, rep_penalty)
        # ... 后续 top-k / top-p / sample
```

### 🪤 替代方案：no_repeat_ngram

```python
def no_repeat_ngram_logits(logits, input_ids, ngram_size=3):
    """禁止生成已经出现过的 ngram"""
    B = input_ids.size(0)
    for i in range(B):
        seq = input_ids[i].tolist()
        # 收集所有出现过的 (n-1)-gram → next-token 映射
        banned = set()
        for j in range(len(seq) - ngram_size + 1):
            prefix = tuple(seq[j:j + ngram_size - 1])
            if prefix == tuple(seq[-(ngram_size - 1):]):
                banned.add(seq[j + ngram_size - 1])
        # 把这些 banned 的 token 屏蔽
        for tok in banned:
            logits[i, tok] = float('-inf')
    return logits
```

### 🪤 选型建议

| 场景 | 用什么 |
|---|---|
| 通用 | rep_penalty=1.05 ~ 1.2 |
| 重复严重 | rep_penalty=1.3 + no_repeat_ngram=3 |
| 代码生成 | 不用！因为代码里关键字必然重复 |

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
