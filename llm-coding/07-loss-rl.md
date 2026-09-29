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

    if not (shift_labels != ignore_index).any():
        # 教学约定：无有效 token 返回可反传的零，不让 mean(empty) 产生 NaN。
        return logits.reshape(-1)[:0].sum()
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

`ignore_index` 影响归约分母：默认 mean 只除以有效 token 数。分布式训练若每张卡有效 token 数不同，先算各卡 mean 再平均会让各卡权重相同而不是各 token 权重相同；严谨实现应 all-reduce loss sum 和 valid-token count 后再相除。

更准确地区分两件事：报告指标可以对 detached loss sum/count 做 all-reduce；训练时默认 DDP 会对梯度取卡间平均，因此若总有效数为 N、world size 为 W，每卡应反传 `local_loss_sum * W / N` 才得到全局 token mean 梯度。不能把任意不支持 autograd 的通信调用直接放到 loss 中就认为梯度正确。

### 从公式手写稳定 CE

下面是单位置批量 CE，未做 LM shift；只在有效行计算，避免 padding logits 中的非法值污染归约。数学推导见 [训练原理](../docs/02-training.md#q01--损失函数)。

```python
def cross_entropy_from_logits(logits, target, ignore_index=-100):
    """logits [N,V]，target [N]；只对有效位置求均值。"""
    valid = target != ignore_index
    if not valid.any():
        return logits.reshape(-1)[:0].sum()
    z = logits[valid]
    if z.dtype in (torch.float16, torch.bfloat16):
        z = z.float()
    shifted = z - z.max(dim=-1, keepdim=True).values
    log_probs = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    return -log_probs.gather(1, target[valid, None]).mean()
```

测试不仅比较 loss，还比较对 logits 的梯度：one-hot 情况应为 `(softmax(z)-one_hot(y))/N_valid`。同时测试大幅值 logits、忽略标签、全 ignore 和序列长度为 1 的 LM 输入。全 ignore 的零值约定不同于某些框架默认 mean 行为，训练管线还应记录这种 batch，防止数据异常被零 loss 隐藏。

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
q_i=(1-\epsilon)\mathbb{1}[i=y]+\frac{\epsilon}{V}
$$

### 代码

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        if vocab_size < 1 or not 0 <= smoothing <= 1:
            raise ValueError("invalid vocab_size or smoothing")
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: [N, V]
        target: [N]
        """
        if logits.size(-1) != self.vocab_size:
            raise ValueError("logits size does not match vocab_size")
        mask = target != self.ignore_index
        if not mask.any():
            return logits.reshape(-1)[:0].sum()
        z = logits[mask]
        if z.dtype in (torch.float16, torch.bfloat16):
            z = z.float()
        log_probs = F.log_softmax(z, dim=-1)
        nll = -log_probs.gather(1, target[mask, None]).squeeze(1)
        uniform_ce = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform_ce).mean()
```

### 易错点

- **smoothing=0.1 只是分类任务常见起点**：大词表语言模型未必使用；均匀分给所有错误 token 也忽略了“多个合理下一个词”的语义结构
- **两种定义不要混写**：本例按全类别均匀混合，与 PyTorch 内置参数一致；“正确类 1−ε，其余 ε/(V−1)”是另一种定义，相同 ε 时不等价。V=3、ε=0.1 时两者分别为 `[0.9333,0.0333,0.0333]` 和 `[0.9,0.05,0.05]`
- **不能直接 gather/scatter 非法标签**：本例先筛选有效行；另一种方式是先把 ignore 标签替换成合法 ID，再在归约中去掉
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

1. **为什么要 reference model**：DPO 比较 policy 相对 reference 对 chosen/rejected 的改变量，reference 提供隐式奖励的基准；在原始 KL 正则推导中，它对应不偏离 SFT policy 的锚点
2. **`logsigmoid` 而不是 `sigmoid + log`**：数值稳定（直接 log(sigmoid(x)) 在 x 很负时溢出）
3. **$\beta$ 的两层含义**：在理论目标中它是 KL 正则强度，越大偏离 reference 的代价越高；在实现中它同时缩放分类 logit 和梯度，和学习率、数据 margin 相互作用，不能只凭一次训练的 policy KL 机械判断大小
4. **chosen 和 rejected 必须对应同一个 prompt**；实现时可以拼成一个 batch 做单次 forward，再拆回两组以提高吞吐
5. **`ignore_index` 不能直接传给 `gather`**：先替换成合法 token id，再 mask 掉
6. **序列 log-prob 有长度效应**：总和会让长回答累积更多负值；chosen/rejected 长度分布若不平衡，模型可能学习长度捷径。是否做长度归一化会改变目标，不能静默修改，应通过长度配对和分桶指标先诊断

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
    """k3 样本项；分布/梯度适用条件见 Q05，不自动校正 old-policy 采样。"""
    valid = mask.bool()
    if not valid.any():
        return new_log_probs.reshape(-1)[:0].sum()
    log_ratio = ref_log_probs.detach()[valid] - new_log_probs[valid]
    per_token_kl = torch.expm1(log_ratio) - log_ratio
    if not torch.isfinite(per_token_kl).all():
        raise FloatingPointError("KL overflow: inspect log-prob range")
    return per_token_kl.mean()


# 总 loss（系数仅作结构示意，实际需要调参）
total_loss = actor_loss + value_coef * critic_loss + kl_coef * kl_loss
```

### 易错点

1. **`old_log_probs` 必须 detach**：rollout 阶段算的，不能参与梯度
2. **`min(surr1, surr2)` 的方向**：注意是取**更悲观**的那个，避免乐观更新
3. **Advantage 计算**：通常用 GAE（Generalized Advantage Estimation）
4. **mask 必须**：prompt 部分不算 loss
5. **KL 估计方向**：若样本来自当前 policy，k3 中应使用 `log_ratio = log p_ref - log p_policy`

Clipping 不是硬 KL 约束。它只在样本 advantage 的方向上截断继续提高 surrogate objective 的收益：$A_t>0$ 主要限制 ratio 过大，$A_t<0$ 主要限制 ratio 过小；另一个方向仍可能继续产生梯度。多 epoch、参数共享和样本外动作都可能让真实 KL 变大，所以还要监控 approx KL、clip fraction，并在超阈值时早停或调低学习率。

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
    reduction: str = "sequence_mean",
):
    """
    GRPO 关键：advantage = (r - mean(r in group)) / std(r in group)
    不需要 value model！
    """
    B_total = rewards.size(0)
    if group_size < 2 or B_total == 0 or B_total % group_size != 0:
        raise ValueError("nonempty batch must be divisible by group_size >= 2")
    if new_log_probs.shape != old_log_probs.shape or mask.shape != new_log_probs.shape:
        raise ValueError("log-prob and mask shapes must agree")
    if new_log_probs.size(0) != B_total or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("expected binary mask and one reward per response")
    valid = mask.bool()
    lengths = valid.sum(dim=-1)
    if (lengths == 0).any():
        raise ValueError("each response must have at least one valid token")
    if reduction not in ("sequence_mean", "token_mean"):
        raise ValueError("unknown reduction")
    def reduce_tokens(values):
        values = values.masked_fill(~valid, 0)
        if reduction == "sequence_mean":
            return (values.sum(dim=-1) / lengths).mean()
        return values.sum() / lengths.sum()
    B = B_total // group_size

    # 1) 按 group 计算 advantage
    rewards_grouped = rewards.detach().reshape(B, group_size)     # [B, G]
    mean = rewards_grouped.mean(dim=-1, keepdim=True)              # [B, 1]
    std = rewards_grouped.std(dim=-1, keepdim=True, unbiased=False)
    std = std.clamp_min(1e-4)
    advantages = ((rewards_grouped - mean) / std).view(-1)         # [B*G]
    advantages = advantages.unsqueeze(-1)                          # [B*G, 1]，广播到序列

    # 2) Clipped policy loss（跟 PPO 一样）
    log_ratio = (new_log_probs - old_log_probs.detach()).masked_fill(~valid, 0)
    ratio = log_ratio.exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = reduce_tokens(-torch.min(surr1, surr2))

    # 3) KL penalty（直接加在 loss 里，不像 PPO 在 reward 里减）
    if ref_log_probs is not None:
        if ref_log_probs.shape != new_log_probs.shape:
            raise ValueError("reference shape must agree")
        log_ratio = (ref_log_probs.detach() - new_log_probs).masked_fill(~valid, 0)
        kl = torch.expm1(log_ratio) - log_ratio
        if not torch.isfinite(kl).all():
            raise FloatingPointError("KL overflow: inspect log-prob range and policy drift")
        policy_loss = policy_loss + kl_coef * reduce_tokens(kl)

    if not torch.isfinite(policy_loss):
        raise FloatingPointError("nonfinite loss: inspect ratios and rewards")
    return policy_loss
```

### GRPO 关键创新

1. **Group-relative advantage**：用组内 reward 统计量代替 value baseline，不训练 Critic
2. **KL 项**：$e^x-x-1\ge 0$ 是实数数学性质，不保证浮点稳定；这里 $x=\log\pi_{ref}-\log\pi_\theta$。`expm1(x)-x` 可减轻接近零的消减误差，大正 x 仍可能溢出，应监控策略漂移和 log-prob 范围
3. **zero-variance group**：一组全同分时 advantage 全为 0，必须监控这类组的比例

组均值作为 baseline 不改变同组样本的相对排序，并能降低共同的 prompt 难度造成的方差；除以组内标准差进一步统一不同 prompt 的奖励尺度。但它也会丢掉“这个 prompt 整组都比另一个 prompt 好”的绝对信息。组大小太小，均值/方差估计噪声大；组大小增大，rollout 成本又线性上升。全对/全错组没有相对信号时，应从采样难度和奖励分辨率解决，不能靠给分母加 epsilon 制造梯度。

### 回答等权与 token 等权，不是同一个目标

默认 `sequence_mean`：每条回答先按有效长度平均，再对回答平均，对应原始 GRPO 的回答归一化口径。`token_mean`：把本地 batch 所有有效 token 放在一起平均，长回答权重更大。两条回答长度分别为 1 和 3、各自每 token loss 为 2 和 4 时，两种结果为 `(2+4)/2=3` 与 `(2+12)/4=3.5`。均长时才一致。

这里 `token_mean` 只是本地聚合方式，不是完整 DAPO 算法；跨卡/梯度累积还需全局有效 token 分母。详见 [TRL 聚合说明](https://huggingface.co/docs/trl/grpo_trainer) 与 [后训练原理](../docs/04-alignment.md)。

### KL 数值估计与训练梯度要分开

在固定上下文、样本来自当前 policy 且满足支持集条件时，上面的 k3 对当前 policy 到 reference 的 KL 数值无偏。但本例数据通常由 old policy 采样；多次更新后分布不同，不能仍宣称该样本平均无偏。代码展示常用的样本级 KL 正则，未实现精确的分布校正。即便数值估计无偏，对固定采样 token 直接 autograd，也不自动得到含采样分布变化的完整 KL 梯度；若推导要求精确目标，应明确重要性权重、是否 detach 以及状态访问分布。

### PPO vs DPO vs GRPO

| 维度 | PPO | DPO | GRPO |
|---|---|---|---|
| 需要 reward model | 可用 RM 或规则奖励 | 否（直接用偏好对） | 可用 RM 或规则奖励 |
| 需要 value model | 是 | 否 | 否 |
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

Bradley–Terry loss 只识别分差：$r(y)$ 整体加常数不变，若没有额外约束，绝对“0 分”没有语义。pairwise accuracy 也不衡量分差是否校准；当 RM 被 policy 优化到训练分布之外时，即使 held-out pair accuracy 高，错误排序仍可能被策略放大。

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
