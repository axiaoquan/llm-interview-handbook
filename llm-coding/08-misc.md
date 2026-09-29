# 08 · Misc 手撕

杂项：MoE 路由、Flash Attention 简化版、其他工程小手撕。

## 本章目录

- [Q01 · MoE Top-k Routing](#q01--moe-top-k-routing)
- [Q02 · MoE 负载均衡 Loss](#q02--moe-负载均衡-loss)
- [Q03 · Flash Attention 简化版](#q03--flash-attention-简化版)
- [Q04 · Tied Embedding](#q04--tied-embedding)
- [Q05 · 梯度累积 + 梯度检查点](#q05--梯度累积--梯度检查点)

---

## Q01 · MoE Top-k Routing

### 目标

把 FFN 替换成 N 个专家，每个 token 路由到 top-k 个专家。

### 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """x: [B, L, d_model]"""
        B, L, D = x.shape
        x_flat = x.view(-1, D)                           # [B*L, D]
        N = x_flat.size(0)

        # 1) 门控：每个 token 选 top-k 个专家
        gate_logits = self.gate(x_flat)                   # [N, n_experts]
        topk_logits, topk_idx = gate_logits.topk(self.top_k, dim=-1)  # [N, k]
        topk_weights = F.softmax(topk_logits, dim=-1)     # [N, k]，归一化的权重

        # 2) 对每个专家，找出路由到它的 token，分别计算
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            # 哪些 token、在 top-k 的哪个位置选了这个专家
            mask = (topk_idx == e)                        # [N, k]
            if not mask.any():
                continue
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            # 取出这些 token 跑专家
            expert_out = self.experts[e](x_flat[token_idx])     # [n_routed, D]
            # 乘以对应的 gate weight 后累加回去
            weight = topk_weights[token_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weight)

        return out.view(B, L, D)
```

### 易错点

1. **router 口径要说清楚**：先 top-k 再 softmax，与全量 softmax 后取相同 top-k 再归一化，在无额外变换、无 detach、选中集合固定时，输出与梯度数学等价；有限精度可能造成误差。真正不同的是不再归一化、改变 gate 函数、截断梯度或加入其他损失
2. **`index_add_` in-place**：累加，不是赋值
3. **GPU 利用率**：朴素实现的 `for e in n_experts` 串行；生产用 grouped GEMM 一次算完
4. **专家容量限制**：实际系统常限制每个专家接收的 token 数（capacity factor）；溢出 token 可丢弃专家分支、重路由或走共享/残差路径，策略取决于实现，不能默认都直通

Top-k 的离散选择使未入选专家通常拿不到主任务梯度，容易形成“早期热门专家得到更多训练、因而更热门”的正反馈。负载均衡 loss、router noise、容量控制和 shared experts 都是在解决这个闭环，但约束过强又会牺牲语义专门化。

为什么等价？对选中集合 S，全量 softmax 的公共分母会在再次归一化时约掉，最终都是 $e^{z_i}/\sum_{j\in S}e^{z_j}$。只要没有并列导致选中集合变化，梯度也一致。特别地，本例 `top_k=1` 时唯一权重恒为 1，主任务无法通过该权重训练 router；这不是所有 top-1 MoE 都有的问题，保留未再归一化的全量概率、辅助损失等设计会改变梯度路径。

---

## Q02 · MoE 负载均衡 Loss

### 目标

防止"赢家通吃"——所有 token 都路由到同一个专家。

### 代码

```python
def aux_loss(gate_logits: torch.Tensor, topk_idx: torch.Tensor, n_experts: int):
    """
    Switch Transformer 的辅助 loss
    gate_logits: [N, E]  门控的原始分数
    topk_idx:    [N, k]  实际路由到的专家
    """
    N = gate_logits.size(0)
    # f_e: 路由到专家 e 的 token 比例
    one_hot = F.one_hot(topk_idx, num_classes=n_experts).float()  # [N, k, E]
    f = one_hot.sum(dim=[0, 1]) / (N * topk_idx.size(1))           # [E]，和为 1
    # P_e: 门控对专家 e 的平均分配概率
    P = F.softmax(gate_logits, dim=-1).mean(dim=0)                # [E]
    # loss = E * sum(f_e * P_e)，乘上 E 让数量级合理
    return n_experts * (f * P).sum()


# 用法
total_loss = ce_loss + 0.01 * aux_loss(gate_logits, topk_idx, n_experts)
```

### 易错点

- **辅助 loss 系数**：太大 → 路由混乱效果差；太小 → 失去均衡作用。常用 0.01
- **DeepSeek-V2 的 device-level + expert-level**：分两层均衡，跨设备均衡通信开销
- 对 top-k 路由，$f_e$ 应除以总 assignment 数 $N\times k$；若沿用 Switch Transformer 的 Top-1 公式却只除以 $N$，loss 会额外放大 $k$ 倍。还应分别监控硬路由比例 $f_e$ 和软概率 $P_e$，只看平均 loss 可能掩盖单个热点专家

---

## Q03 · Flash Attention 简化版

### 目标

完整 Flash Attention 用了 GPU SRAM 优化、tiling、CUDA kernel，超出"手撕"范围。我们写一个**展示思想**的版本：

```python
def flash_attention_simplified(Q, K, V, block_size=64):
    """
    展示 Flash Attention 的核心思想：
      1. 分块 K, V
      2. online softmax：流式更新 max 和 sum，避免存中间矩阵
      3. 累加输出
    """
    B, h, L, D = Q.shape
    O = torch.zeros_like(Q)
    L_q = torch.zeros(B, h, L, 1, device=Q.device)        # log-sum-exp 累加器
    M = torch.full((B, h, L, 1), float('-inf'), device=Q.device)  # 最大值

    for j in range(0, L, block_size):
        K_j = K[:, :, j:j+block_size]                      # [B, h, b, D]
        V_j = V[:, :, j:j+block_size]
        # S_ij = Q @ K_j.T / sqrt(D)
        S = torch.matmul(Q, K_j.transpose(-2, -1)) / (D ** 0.5)   # [B, h, L, b]

        # online softmax 更新
        M_new = torch.maximum(M, S.max(dim=-1, keepdim=True).values)
        P = (S - M_new).exp()                              # [B, h, L, b]
        L_new = L_q * (M - M_new).exp() + P.sum(dim=-1, keepdim=True)

        # 输出累加
        O = O * (M - M_new).exp() + torch.matmul(P, V_j)

        M = M_new
        L_q = L_new

    return O / L_q
```

### 易错点

- **online softmax 是核心**：流式更新最大值，每次 rescale 之前的累加结果
- **真实 Flash Attention 在 SRAM 里做**：上面只是数学等价的版本，还没体现"省显存"的效果（需要 CUDA kernel）
- **生产用法**：直接 `F.scaled_dot_product_attention(Q, K, V, is_causal=True)`，PyTorch 2+ 内置 Flash Attention 后端

---

## Q04 · Tied Embedding

### 目标

把输入 token embedding 和输出 LM head 的权重绑定（共享），减少参数。

### 代码

```python
class TiedLM(nn.Module):
    def __init__(self, vocab_size, d_model, transformer):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = transformer  # 调用者传入 [..., D] -> [..., D] 的因果模块
        # LM head 不创建新参数，复用 embed 的权重
        # 不需要单独的 self.lm_head

    def forward(self, input_ids):
        x = self.embed(input_ids)                        # [B, L, D]
        x = self.transformer(x)                          # [B, L, D]
        # 用 embed.weight.T 当 LM head
        logits = x @ self.embed.weight.T                 # [B, L, V]
        return logits
```

### 易错点

- **优点**：节省 vocab_size × d_model 个参数，并让“读入 token 的语义空间”和“预测 token 的分类空间”共享几何结构
- **约束**：输入 embedding 与输出 classifier 被迫共享同一矩阵，可能限制两者各自最优表示；效果可能提升也可能下降，不存在固定 0.5 PPL 结论
- **HuggingFace 的实现**：许多模型配置提供 `tie_word_embeddings`；默认值取决于具体架构

这是依赖外部 block 的组件，不是完整语言模型。传 `nn.Identity()` 可以验证 embedding/head 共享与输出 shape，但没有上下文建模能力；真实自回归训练需传带因果约束的 Transformer。

---

## Q05 · 梯度累积 + 梯度检查点

### 梯度累积：模拟大 batch

```python
def train_with_grad_accum(model, dataloader, optimizer, accum_steps=4):
    optimizer.zero_grad()
    pending = 0
    for step, batch in enumerate(dataloader):
        loss = model(batch)
        # 关键：除以累积步数，让总的梯度等于大 batch 的等效梯度
        loss = loss / accum_steps
        loss.backward()
        pending += 1

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            pending = 0

    # dataloader 长度不是 accum_steps 整数倍时，不能静默丢掉最后几步梯度。
    # 前面每步除以 accum_steps，这里按实际 pending 数把最后窗口校正回来。
    if pending:
        correction = accum_steps / pending
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(correction)
        optimizer.step()
        optimizer.zero_grad()
```

### 梯度检查点：用算力换显存

```python
import torch.utils.checkpoint as checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # 训练时不存中间激活，反向时重算；推理时正常跑
        if self.training:
            return checkpoint.checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)
```

### 易错点

1. **梯度累积**：loss 必须 `/= accum_steps`，否则等于 `accum_steps` 倍学习率
2. **梯度检查点**：不保存被 checkpoint 区域的大部分中间激活，反向时重算；省多少显存、慢多少取决于切分粒度和算子，不能背固定百分比
3. **`use_reentrant=False`**：PyTorch 新版推荐，避免一些边界 bug

只有每个 micro-batch 样本/token 权重相同且模型中无跨 batch 统计时，梯度累积才近似等价于一个大 batch。变长序列若每步先做 token mean 再平均，会让短序列和长序列权重失真；更严谨的做法是累计 loss sum 和有效 token 数后统一归一化。BatchNorm 的统计量、dropout 随机性和优化器更新频率也会让两者不完全等价。

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
