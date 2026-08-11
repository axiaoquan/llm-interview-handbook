# 07 · Loss & RL 手撕

LLM 训练 / 对齐 中的核心损失函数。

## 本章目录

- [Q01 · Cross-Entropy Loss（next-token prediction）](#q01--cross-entropy-lossnext-token-prediction)
- [Q02 · Label Smoothing](#q02--label-smoothing)
- [Q03 · DPO Loss](#q03--dpo-loss)
- [Q04 · PPO Clipped Loss](#q04--ppo-clipped-loss)
- [Q05 · GRPO Loss](#q05--grpo)
- [Q06 · Reward Model Loss](#q06--reward-model-loss)
- [Q07 · 多奖励解耦归一化](#q07--多奖励解耦归一化)

---

## Q01 · Cross-Entropy Loss（next-token prediction）

### 目标

LLM 训练的核心 loss：每个位置预测下一个 token。

### 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """
    logits: [B, L, V]   ← 模型输出
    labels: [B, L]      ← input_ids（next-token 自己错位生成 target）
    """
    # 关键：错位（shift）
    # 第 i 个位置的 logits 预测的是第 i+1 个 token
    shift_logits = logits[:, :-1, :].contiguous()                # [B, L-1, V]
    shift_labels = labels[:, 1:].contiguous()                     # [B, L-1]

    # 拉平后算 CE
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),             # [B*(L-1), V]
        shift_labels.view(-1),                                    # [B*(L-1)]
        ignore_index=ignore_index
    )
    return loss
```

### 易错点（**面试常考**）

1. **shift 错位**：必须是 `logits[:-1] vs labels[1:]`
2. **ignore_index = -100**：把 padding / prompt 部分的 label 设成 -100，CE 会跳过
3. **`contiguous()` 必须**：否则 `.view` 报错
4. **指令微调的 mask label**：训练 SFT 时只计算 response 部分的 loss，prompt 部分 label 全设 -100

```python
# SFT 标签构造
def make_sft_labels(input_ids, prompt_length):
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100   # 不算 prompt 的 loss
    return labels
```

---

## Q02 · Label Smoothing

### 目标

把 one-hot 标签变成"接近 one-hot 但不极端"，正则化模型避免过自信。

$$
P_{\text{smoothed}}(y_i) =
\begin{cases}
1-\epsilon, & y_i=y,\\
\epsilon/(V-1), & y_i\ne y.
\end{cases}
$$

### 代码

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: [N, V]
        target: [N]
        """
        log_probs = F.log_softmax(logits, dim=-1)        # [N, V]
        mask = target != self.ignore_index
        safe_target = target.masked_fill(~mask, 0)       # 避免 scatter(-100) 越界
        if not mask.any():
            return logits.sum() * 0.0                    # 保留计算图，返回可反传的 0

        # 构造 smoothed 分布
        with torch.no_grad():
            smoothed = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            # 在正确类别填 1 - smoothing
            smoothed.scatter_(1, safe_target.unsqueeze(1), 1.0 - self.smoothing)
            smoothed[~mask] = 0

        # 算 KL 等价的形式
        loss = -(smoothed * log_probs).sum(dim=-1)
        return loss[mask].mean()
```

### 易错点

- **smoothing=0.1 是常用值**：再大模型容易欠拟合
- **代码中的 vocab_size - 1**：因为正确类别已经分配了 1-ε，剩下 ε 平均分给其他 V-1 类
- **不能直接 `scatter_(-100)`**：先把 ignore 位置临时替换成合法 id，再用 mask 去掉
- **PyTorch 的 `F.cross_entropy` 自带 label_smoothing 参数**（PyTorch ≥1.10）：

```python
loss = F.cross_entropy(logits, target, label_smoothing=0.1)
```

---

## Q03 · DPO Loss

### 目标

直接优化偏好学习，跳过 reward model 和 PPO 的复杂 RL 训练。

$$
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

### 代码

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # [B] 当前 model 在 chosen 序列上的 log prob 总和
    policy_rejected_logps: torch.Tensor,  # [B]
    ref_chosen_logps: torch.Tensor,       # [B] 冻结的 reference model 的 log prob
    ref_rejected_logps: torch.Tensor,     # [B]
    beta: float = 0.1,
):
    """
    返回 (loss, chosen_rewards, rejected_rewards)
    rewards 用于监控（不参与 loss 计算）
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps    # [B]
    ref_logratios = ref_chosen_logps - ref_rejected_logps          # [B]

    logits = beta * (pi_logratios - ref_logratios)                 # [B]

    loss = -F.logsigmoid(logits).mean()
    # 这两个只用于监控；理想情况是 chosen_rewards > rejected_rewards
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards


def get_seq_logps(model, input_ids, labels, ignore_index=-100):
    """计算一个序列的 log prob 总和"""
    logits = model(input_ids).logits[:, :-1]                       # [B, L-1, V]
    labels = labels[:, 1:]                                         # [B, L-1]
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != ignore_index
    safe_labels = labels.masked_fill(~mask, 0)                       # 避免 gather(-100) 越界
    # gather 出每个位置真实 token 的 log prob
    per_token_logp = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    # mask 掉 ignore_index（一般是 prompt 部分）
    return (per_token_logp * mask).sum(dim=-1)                      # [B]
```

### 易错点（**面试高频**）

1. **为什么要 reference model**：约束当前 policy 不要离 SFT 模型太远（否则会忘记基础能力）
2. **`logsigmoid` 而不是 `sigmoid + log`**：数值稳定（直接 log(sigmoid(x)) 在 x 很负时溢出）
3. **β 越大 → 更接近 reference / 越保守**；β 越小 → 自由度高但训练不稳定
4. **chosen 和 rejected 必须对应同一个 prompt**；实现时可以拼成一个 batch 做单次 forward，再拆回两组以提高吞吐
5. **`ignore_index` 不能直接传给 `gather`**：先替换成合法 token id，再 mask 掉

---

## Q04 · PPO Clipped Loss

### 目标

RLHF 的 actor 损失，限制更新幅度避免训练崩溃。

$$
\mathcal{L}^{CLIP} = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

其中 $r_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$。

### 代码

```python
def ppo_loss(
    new_log_probs: torch.Tensor,   # [B, L]
    old_log_probs: torch.Tensor,   # [B, L]   ← detached, 来自 rollout 时
    advantages: torch.Tensor,      # [B, L]
    mask: torch.Tensor,            # [B, L]   ← response 部分为 1
    clip_eps: float = 0.2,
):
    """计算 PPO Actor Loss"""
    log_ratio = new_log_probs - old_log_probs        # [B, L]
    ratio = log_ratio.exp()

    # 两条路径：clip 和 not clip
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2)            # [B, L]

    # 只在 response 位置算 loss
    policy_loss = (policy_loss * mask).sum() / mask.sum().clamp_min(1)
    return policy_loss


def value_loss(values, returns, old_values, mask, clip_eps=0.2):
    """Value head 的 clipped MSE loss"""
    values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    losses = (values - returns) ** 2
    losses_clipped = (values_clipped - returns) ** 2
    loss = 0.5 * torch.max(losses, losses_clipped)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def kl_penalty(new_log_probs, ref_log_probs, mask):
    """k3 estimator：在 policy 样本上估计 KL(policy || reference)。"""
    log_ratio = ref_log_probs - new_log_probs
    per_token_kl = log_ratio.exp() - log_ratio - 1
    return (per_token_kl * mask).sum() / mask.sum().clamp_min(1)


# 总 loss（系数仅作结构示意，实际需要调参）
total_loss = actor_loss + value_coef * critic_loss + kl_coef * kl_loss
```

### 易错点

1. **`old_log_probs` 必须 detach**：rollout 阶段算的，不能参与梯度
2. **`min(surr1, surr2)` 的方向**：注意是取**更悲观**的那个，避免乐观更新
3. **Advantage 计算**：通常用 GAE（Generalized Advantage Estimation）
4. **mask 必须**：prompt 部分不算 loss
5. **KL 估计方向**：若样本来自当前 policy，k3 中应使用 `log_ratio = log p_ref - log p_policy`

---

## Q05 · GRPO

### 目标

去掉 value model，直接用同一 prompt 的多个 rollout 奖励计算组相对 advantage。

### 代码

```python
def grpo_loss(
    new_log_probs: torch.Tensor,   # [B*G, L]   B 个 prompt，每个采样 G 个 response
    old_log_probs: torch.Tensor,   # [B*G, L]
    rewards: torch.Tensor,         # [B*G]      每个 response 的 reward
    mask: torch.Tensor,            # [B*G, L]
    group_size: int,               # G
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
    ref_log_probs: torch.Tensor = None,
):
    """
    GRPO 关键：advantage = (r - mean(r in group)) / std(r in group)
    不需要 value model！
    """
    B_total = rewards.size(0)
    if B_total % group_size != 0:
        raise ValueError("batch size must be divisible by group_size")
    B = B_total // group_size

    # 1) 按 group 计算 advantage
    rewards_grouped = rewards.view(B, group_size)                  # [B, G]
    mean = rewards_grouped.mean(dim=-1, keepdim=True)              # [B, 1]
    std = rewards_grouped.std(dim=-1, keepdim=True, unbiased=False)
    std = std.clamp_min(1e-4)
    advantages = ((rewards_grouped - mean) / std).view(-1)         # [B*G]
    advantages = advantages.unsqueeze(-1)                          # [B*G, 1]，广播到序列

    # 2) Clipped policy loss（跟 PPO 一样）
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2) * mask
    policy_loss = policy_loss.sum() / mask.sum().clamp_min(1)

    # 3) KL penalty（直接加在 loss 里，不像 PPO 在 reward 里减）
    if ref_log_probs is not None:
        log_ratio = ref_log_probs - new_log_probs
        kl = log_ratio.exp() - log_ratio - 1
        kl_loss = (kl * mask).sum() / mask.sum().clamp_min(1)
        return policy_loss + kl_coef * kl_loss

    return policy_loss
```

### GRPO 关键创新

1. **Group-relative advantage**：用组内 reward 统计量代替 value baseline，不训练 Critic
2. **KL estimate**：$e^x-x-1\ge 0$，单样本非负且数值稳定；这里 $x=\log\pi_{ref}-\log\pi_\theta$
3. **zero-variance group**：一组全同分时 advantage 全为 0，必须监控这类组的比例

### PPO vs DPO vs GRPO

| 维度 | PPO | DPO | GRPO |
|---|---|---|---|
| 需要 reward model | ✅ 或规则奖励 | ❌（直接用偏好对） | 可用 RM 或规则奖励 |
| 需要 value model | ✅ | ❌ | ❌ |
| 训练稳定 | 中（要调参） | 较好 | 依赖组内奖励方差 |
| 数据要求 | prompt + 标量奖励 | 偏好对 | prompt + 同组多次采样 |
| 适合 | 通用对齐 | SFT 后微调 | 推理类（数学、代码） |
| 代表 | InstructGPT | Zephyr 等 | DeepSeekMath / DeepSeek-R1 |

---

## Q06 · Reward Model Loss

### 目标

训练标量 Reward Model，使 chosen 回答分数高于 rejected 回答。

$$
\mathcal{L}_{RM}=-\log\sigma(r_{chosen}-r_{rejected})
$$

### 代码

```python
def reward_model_loss(
    chosen_scores: torch.Tensor,    # [B]
    rejected_scores: torch.Tensor,  # [B]
    margin: torch.Tensor = None,     # [B]，可选：偏好强度
):
    score_diff = chosen_scores - rejected_scores
    if margin is not None:
        score_diff = score_diff - margin

    loss = -F.logsigmoid(score_diff).mean()
    accuracy = (chosen_scores > rejected_scores).float().mean()
    return loss, accuracy
```

### 易错点

1. RM 学的是**相对分差**，所有分数同时平移不改变损失。
2. `margin > 0` 表示不仅要排对，还要求分差至少达到 margin。
3. 不能只看 pairwise accuracy；还要看分差分布、校准、不同长度分桶和 OOD 偏好准确率。
4. chosen / rejected 的 padding、模板和截断策略要一致，避免模型学习伪特征。

---

## Q07 · 多奖励解耦归一化

### 目标

对每个奖励维度分别做组内标准化，再按权重合并，避免“先求总分”丢失多维奖励差异。

### 代码

```python
def decoupled_group_advantages(
    rewards: torch.Tensor,  # [B, G, K]：prompt、rollout、reward 维度
    weights: torch.Tensor,  # [K]
    eps: float = 1e-4,
):
    if rewards.ndim != 3:
        raise ValueError("rewards must have shape [B, G, K]")
    if rewards.size(-1) != weights.numel():
        raise ValueError("weights must match the reward dimension")
    weights = weights.to(device=rewards.device, dtype=rewards.dtype)

    # 每个 prompt 内，对每个 reward 维度分别归一化
    mean = rewards.mean(dim=1, keepdim=True)                        # [B, 1, K]
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    normalized = (rewards - mean) / std.clamp_min(eps)             # [B, G, K]

    # 合并各维 advantage；weights 通常预先归一化
    advantages = (normalized * weights.view(1, 1, -1)).sum(-1)     # [B, G]
    return advantages
```

### 易错点

1. 某一奖励维度在组内全相同时，归一化后该维贡献为 0；这比制造虚假方差更合理。
2. 各维先归一化会消除原始尺度含义，若某个奖励的绝对差值很重要，需要单独建模。
3. 权重改变不一定等于优化优先级改变；还要监控各维饱和率和对最终梯度的贡献。
4. 核心正确性与次级格式目标有严格优先级时，条件化奖励或约束优化可能比简单加权更合适。

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
