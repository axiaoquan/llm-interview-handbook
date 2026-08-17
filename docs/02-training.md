# 02 · Training 训练

预训练 / 监督训练相关的优化、稳定性、分布式策略。

## 本章目录

- [Q01 · 损失函数（CE / MSE / Label Smoothing）](#q01--损失函数)
- [Q02 · 优化器（SGD / Adam / AdamW / Lion）](#q02--优化器)
- [Q03 · 学习率调度（Warmup + Cosine / WSD）](#q03--学习率调度)
- [Q04 · 梯度问题（消失 / 爆炸 / NaN）](#q04--梯度问题)
- [Q05 · 混合精度训练（FP16 / BF16）](#q05--混合精度训练)
- [Q06 · 分布式训练（DP / DDP / FSDP / DeepSpeed ZeRO）](#q06--分布式训练)
- [Q07 · 并行策略（TP / PP / SP / EP）](#q07--并行策略)

---

## Q01 · 损失函数

### 1) 分类问题：交叉熵损失（CE Loss）

**二分类（BCE）**：

$$
L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

**多分类（CCE）** —— LLM next-token prediction 用的就是这个：

$$
L = -\sum_i y_i \log \hat{y}_i
$$

### 2) 回归问题

| 损失 | 公式 | 特点 |
|---|---|---|
| **MSE / L2** | $\frac{1}{n}\sum (y - \hat{y})^2$ | 对 outlier 敏感 |
| **MAE / L1** | $\frac{1}{n}\sum |y - \hat{y}|$ | 对 outlier 鲁棒 |
| **Huber** | 小误差用 L2，大误差用 L1 | 兼顾两者 |

### 3) Label Smoothing

把硬 one-hot 标签软化成 $(1-\epsilon, \epsilon/(K-1), \ldots)$，**抑制过度置信**并给非目标类别提供梯度。$\epsilon=0.1$ 是分类任务常见起点，但因果语言模型并非默认都使用：词表极大且标签本身存在多解时，它可能改善校准，也可能削弱对正确 token 的峰值概率、影响生成质量。应把它当超参数而不是 LLM 固定配置。

### 追问

- **Q：为什么 LLM 用交叉熵不用 MSE？**
  A：CE 是分布间 KL 散度的等价形式，匹配概率分布的语义；MSE 假设高斯，不适合离散 token。

- **Q：分类用 MSE 会怎么样？**
  A：和 sigmoid/softmax 配对时容易梯度消失；CE 与之配对梯度形式简洁（$\hat{y} - y$）。

- **Q：为什么 CE 与 softmax 的梯度特别合适？**
  A：对 logit $z_j$ 求导，softmax 的交叉项与 $-\log p_y$ 正好抵消，得到 $\partial L/\partial z_j=p_j-\mathbb{1}[j=y]$。错误且自信时梯度仍大；MSE 还会额外乘 softmax Jacobian，在饱和区更容易把梯度压小。

---

## Q02 · 优化器

### 主流优化器

| 优化器 | 特点 | 适用 |
|---|---|---|
| **SGD** | 基础，无动量 | 简单任务 |
| **SGD + Momentum** | 加动量加速 | CV 经典 |
| **Adam** | 一阶 + 二阶矩估计，自适应学习率 | 通用 |
| **AdamW** | Adam + **解耦权重衰减** | **LLM 默认** |
| **Lion** | 只维护一阶动量并用 sign 更新 | 相比 Adam 省去二阶矩状态 |

### AdamW 关键点

若把 L2 项加入 Adam 的原始梯度，$\lambda\theta$ 也会经过动量和二阶矩预条件，不同参数受到的“衰减”取决于历史梯度尺度。AdamW 将收缩步骤从梯度预条件中拆开，使每步参数收缩近似为 $\theta\leftarrow(1-\eta\lambda)\theta$，这就是“解耦”，而不只是代码里多写一个 $\lambda\theta$。

```text
# AdamW 更新（核心）
m = β1 * m + (1 - β1) * g
v = β2 * v + (1 - β2) * g²
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
θ = (1 - lr * λ) * θ                                # 独立衰减
θ = θ - lr * m_hat / (sqrt(v_hat) + ε)             # Adam 更新
```

### 优化器显存开销

每个参数（FP32）需要存：
- 参数本身：4 bytes
- 梯度：4 bytes
- Adam 一阶矩 m：4 bytes
- Adam 二阶矩 v：4 bytes
- = **每参数 16 bytes**

→ 7B 模型按这个口径总计约 **112 GB**；其中 Adam 的 $m,v$ 两份优化器状态是 8 bytes/param，约 **56 GB**。实际还要算 activation、临时 buffer、通信 bucket 和显存碎片；参数/梯度是否保留 FP32 副本取决于具体混合精度实现。

### 追问

- **Q：β1 / β2 默认值？**
  A：β1=0.9（一阶动量），β2=0.999（二阶动量），但 LLM 训练**β2 常设为 0.95**（更适应大梯度变化）。

- **Q：Lion 比 AdamW 好在哪？**
  A：只需要存一阶动量 → 显存省一半。某些场景效果更好，但调参比 AdamW 敏感。

---

## Q03 · 学习率调度

### 1) Warmup + Cosine Decay（最常用）

- **Warmup**：从 0 线性增加到 peak lr
- **Cosine decay**：按余弦曲线下降到 ~10% peak lr

### 2) WSD（Warmup-Stable-Decay）

DeepSeek 等使用：

- **Warmup** 阶段线性增加
- **Stable** 阶段保持恒定（占大部分时间）
- **Decay** 阶段快速下降

优势：**stable 阶段可以随时分叉**做不同实验，不浪费前期算力。

### 为什么需要 Warmup？

1. 初期残差流、归一化层和各层梯度尺度尚未协调，直接使用峰值学习率容易让某些层更新量相对权重过大
2. Adam 虽有 bias correction，但 $v_t$ 仍由很少的 batch 估计，方差大；warmup 让预条件器先积累统计量
3. 大 batch 训练常使用更高峰值学习率，warmup 可避免最初几步的离散更新破坏表示

Warmup 不是为了“让随机参数先学会一点”这么简单，也不能解决错误初始化、坏数据或过高峰值学习率。比例没有固定答案：应结合 token 数、batch、优化器和 loss spike 监控确定。

---

## Q04 · 梯度问题

### 1) 梯度消失

**症状**：深层梯度趋近 0，浅层学不到。

**解决方案**：

- **残差连接**（Residual Connection）—— 最重要
- **Pre-Norm**（而非 Post-Norm）
- 合适的初始化（Xavier / He）
- 用 ReLU 及其变体（替代 sigmoid）

### 2) 梯度爆炸

**症状**：loss 突然爆炸成 NaN / inf。

**解决方案**：

- **梯度裁剪**（Gradient Clipping）：

  ```math
  g \leftarrow \min\left(1, \frac{c}{\lVert g\rVert_2}\right)g
  ```

  常见 $c = 1.0$
- 权重衰减（weight decay）
- 较小的学习率

### 3) Loss Spike（训练突然崩）

LLM 大规模训练常见，但不能只归因于“极少数坏样本”。还可能来自学习率/动量过激、数据分布突变、attention mask 错误、数值溢出、通信错误或恢复 checkpoint 后优化器状态不一致。

**解决**：

- BF16 替代 FP16（动态范围更大）
- 记录触发 spike 的数据 shard、各层 grad norm、激活最大值和 loss scale，先定位是数据、数值还是系统问题
- 只有确认单个 batch 损坏时才跳过；盲目跳过会掩盖可重复的实现错误
- 若要求精确续训，checkpoint 应同时恢复 optimizer、scheduler、RNG 和 dataloader 位置，否则训练轨迹会跳变

### 追问

- **Q：梯度裁剪应该用 norm 还是 value？**
  A：**Norm** 更主流（按梯度向量的 L2 norm 缩放），保持方向；value 容易扭曲。

---

## Q05 · 混合精度训练

### 核心做法

**FP16/BF16** 主要用于矩阵计算，归一化、归约、优化器状态等敏感环节常保留 FP32。经典 FP16 方案维护 FP32 master weights；BF16/FSDP/不同优化器是否保留完整主权重取决于实现，不能一概而论。

### 三种精度对比

| 精度 | 位数 | 范围 | 用途 |
|---|---|---|---|
| **FP32** | 32 | $\pm 3.4 \times 10^{38}$ | 主权重、优化器状态 |
| **FP16** | 16 | $\pm 6.5 \times 10^{4}$ | 前向 / 反向 |
| **BF16** | 16 | $\pm 3.4 \times 10^{38}$ | 前向 / 反向（更稳定） |

### BF16 vs FP16

|  | BF16 | FP16 |
|---|---|---|
| 指数位 | 8 | 5 |
| 尾数位 | 7 | 10 |
| 数值范围 | 大（接近 FP32） | 小 |
| 精度 | 低 | 高 |
| 训练稳定性 | **好**（不需要 loss scaling） | 差（需要 loss scaling） |
| 现代 LLM | **首选** | 旧选择 |

### Loss Scaling（FP16 必备）

FP16 动态范围小，梯度容易**下溢成 0**。
做法：把 loss 乘一个大数（比如 1024），反向传播时梯度也被放大；优化器更新前再除回来。

BF16 范围接近 FP32，**不需要这个技巧**。

### 追问

- **Q：训练用混合精度，参数量怎么算显存？**
  A：参数 (FP16) 2B + 梯度 (FP16) 2B + 主权重 (FP32) 4B + Adam m,v (FP32) 8B = **每参数 16B**。所以 7B 模型 ~112GB 起步。

---

## Q06 · 分布式训练

### 数据并行家族

| 方法 | 原理 | 显存优化 |
|---|---|---|
| **DP**（DataParallel） | 单进程多 GPU，主进程聚合 | 无优化，已淘汰 |
| **DDP**（DistributedDataParallel） | 每 GPU 一个进程，all-reduce 同步梯度 | 标准基线 |
| **FSDP**（Fully Sharded DP） | 参数 / 梯度 / 优化器都分片 | **现代主流** |
| **DeepSpeed ZeRO** | 类似 FSDP，分三级 | 工业落地 |

### ZeRO 三级

| 级别 | 切分内容 | 显存节省 |
|---|---|---|
| **ZeRO-1** | 优化器状态 | 优化器状态部分约缩小为 $1/N$ |
| **ZeRO-2** | + 梯度 | 再分片梯度 |
| **ZeRO-3** | + 参数 | 三类模型状态都约按 $1/N$ 分片 |

不能把总显存简单写成固定“4×/8×”：activation、临时 all-gather buffer 和碎片不随 ZeRO stage 等比例下降。ZeRO-3 与 FULL_SHARD FSDP 思路相近，前向/反向需要按层 all-gather 参数并 reduce-scatter 梯度，以通信换模型状态显存。

### ZeRO-Offload / ZeRO-Infinity

- **Offload**：把优化器状态、梯度卸载到 CPU 内存
- **Infinity**：连参数都可以放到 NVMe SSD
- 适合**单机训练超大模型**（牺牲速度换显存）

### 追问

- **Q：DDP 和 FSDP 怎么选？**
  A：模型能放进单卡用 DDP；放不下用 FSDP/ZeRO-3。

---

## Q07 · 并行策略

### 四大并行

| 类型 | 切分维度 | 通信 | 适用 |
|---|---|---|---|
| **DP**（数据并行） | batch 维 | gradient all-reduce | 模型能装单卡 |
| **TP**（Tensor Parallel） | 层内矩阵切分 | activation all-reduce | 层太大装不下 |
| **PP**（Pipeline Parallel） | 层间切分 | 流水线传 activation | 层数多 |
| **SP**（Sequence Parallel） | 序列长度 | 节省 activation | 长上下文 |
| **EP**（Expert Parallel） | MoE 专家 | all-to-all | MoE 模型 |

实际训练常组合：**DP + TP + PP**（"3D 并行"），DeepSeek 还加 EP。

### Megatron-LM 风格 TP

把 attention 的 Q/K/V 投影按头切分到不同 GPU，FFN 的两层矩阵按列/行切分。
**通信发生在 attention 内部和 FFN 内部**（all-reduce）。

### PP 的 bubble 问题

每个阶段 GPU 等待上游 → 利用率低。
解决：**1F1B / interleaved 1F1B**（micro-batch 流水线）减少 bubble。

### 追问

- **Q：3D 并行的具体维度怎么分？**
  A：DP 跨节点（带宽要求低），TP 节点内（高带宽 NVLink），PP 节点间（少 stage）。

- **Q：长上下文为什么要 SP？**
  A：activation 随序列长度增长，SP 可把 LayerNorm、dropout、残差等逐 token 运算沿序列维分片。但标准 attention 仍需要跨分片获取 K/V 或采用 ring/context parallel 通信；“每张卡只算一段”不等于没有跨卡依赖。

---

[⬅ 回到首页](../README.md)
