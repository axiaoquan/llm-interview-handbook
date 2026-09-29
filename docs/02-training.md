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
- [Q08 · 预训练数据与计算预算](#q08--预训练数据与计算预算)
- [Q09 · 有效 token、梯度累积与精确续训](#q09--有效-token梯度累积与精确续训)

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

### 为什么是负对数：从最大似然到交叉熵

给定上下文 $x$ 和观测回答 $y$，自回归模型用链式法则分解序列概率。这不是假设 token 彼此独立，而是每一步都条件于之前的 token：

```math
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{1:t-1})
```

训练希望提高观测数据的似然。log 严格单调，最大化 log 似然与最大化似然具有相同最优解；log 将连乘变成求和，负号再将最大化转为最小化：

```math
L=-\log p_\theta(y\mid x)
  =-\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{1:t-1})
```

对一个位置，one-hot 标签 $q$ 让 $-\sum_j q_j\log p_j$ 只剩 $-\log p_y$。对软标签，它是目标分布下的负 log 概率期望。由 $H(q,p)=H(q)+D_{KL}(q\Vert p)$，固定目标 $q$ 时，最小化 CE 等价于最小化该方向的 KL；不是说 CE 本身总等于 KL。有限样本的 one-hot 是一次观测，也不代表该上下文只有唯一合理续写。

### 为什么和 softmax 配合？梯度从哪里来？

softmax 将任意实数 logits 转为归一化类别概率。令 $p_j=e^{z_j}/\sum_k e^{z_k}$，且 $\sum_j q_j=1$：

```math
L=\log\sum_k e^{z_k}-\sum_j q_jz_j,
\qquad \frac{\partial L}{\partial z_j}=p_j-q_j
```

第一项导数是 softmax，第二项导数是目标概率，因此梯度恰好是“预测减目标”。提高所有 logits 同一个常数不会改变 softmax 或 loss；真正重要的是类别之间的相对分数。

例：真实类概率从 0.9 降至 0.1，loss 从约 0.105 增至 2.303。二分类真值为 1、$p=0.001$ 时，CE 对 logit 的梯度为 $p-1=-0.999$；若用半平方误差 $\frac12(p-1)^2$，梯度是 $(p-1)p(1-p)\approx-0.000998$。额外的 sigmoid 导数使“错得很自信”的平方损失更新较弱。

这不意味着 MSE 不能分类。**对 token ID 做回归**会引入编号之间没有语义的距离；**对 one-hot 概率做平方损失**则是有效的 Brier 类评分。CE 是类别似然的自然选择，且与 softmax 组合后没有额外的饱和导数因子。

数值实现使用 `log_softmax` 或稳定的 log-sum-exp：减去最大 logit 后再指数求和。`F.cross_entropy` 接收原始 logits，不要先 softmax 再传入，否则相当于把概率再次当 logits。完整推导背景见 [Deep Learning 第六章](https://www.deeplearningbook.org/contents/mlp.html)；实现见 [Loss 手撕 Q01](../llm-coding/07-loss-rl.md#q01--cross-entropy-lossnext-token-prediction)。

### 2) 回归问题

| 损失 | 公式 | 特点 |
|---|---|---|
| **MSE / L2** | $\frac{1}{n}\sum (y - \hat{y})^2$ | 对 outlier 敏感 |
| **MAE / L1** | $\frac{1}{n}\sum |y - \hat{y}|$ | 对 outlier 鲁棒 |
| **Huber** | 小误差用 L2，大误差用 L1 | 兼顾两者 |

### 3) Label Smoothing

将目标与均匀分布混合：$q_j=(1-\epsilon)\mathbb{1}[j=y]+\epsilon/K$，这是 [PyTorch `label_smoothing`](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) 的约定。另一种约定把正确类设为 $1-\epsilon$，其余各分 $\epsilon/(K-1)$；相同 ε 下两者不同。三分类 ε=0.1 时，分别为 `[0.9333,0.0333,0.0333]` 和 `[0.9,0.05,0.05]`。

软化标签可抑制过度置信，但普通 CE 已经会给非目标 logits 梯度，不能说只有 smoothing 才提供这种梯度。ε=0.1 只是分类任务常见起点，不是因果语言模型固定配置；应验证校准和生成质量。

### 追问

- **Q：为什么 LLM 用交叉熵不用 MSE？**
  A：CE 来自类别分布的最大似然；与 softmax 配合得到 $p-q$ 梯度。对概率使用 MSE 也可分类，但经过 softmax Jacobian 后可能在饱和区更新较弱；不能把任意 token 编号当连续回归目标。

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

### 为什么要做偏差修正？

从零初始化动量，在梯度均值固定为 $\mu$ 的简化假设下，展开指数滑动平均：

```math
m_t=(1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k}g_k,
\qquad
\mathbb{E}[m_t]=(1-\beta_1^t)\mu
```

前几步累计权重不足 1，所以除以 $1-\beta_1^t$；二阶矩对 $\mathbb{E}[g^2]$ 做同样修正。它修正的是零初始化造成的缩小，不保证非平稳训练中的统计量准确，也不使最终的比值更新成为“无偏梯度”。当 $\beta_1=0.9$、首步梯度为 2 时，$m_1=0.2$，修正后为 2。

分母中的 $\varepsilon$ 避免二阶矩接近零时除零，并限制极小梯度下的更新幅度；$\sqrt{\hat v}+\varepsilon$ 与 $\sqrt{\hat v+\varepsilon}$ 不是同一个优化器。bias、Norm 的 scale/bias 是否排除 weight decay 是参数分组策略，常见做法是排除，但不是 AdamW 定义所强制要求。

**一个区分 Adam+L2 与 AdamW 的反例**：令两个参数都为 1、当前数据梯度为 0，历史二阶矩为 $[1,100]$。暂忽略动量混合，L2 项经过预条件后，两个坐标的收缩贡献比例约为 $1:0.1$；AdamW 的独立收缩对两个坐标相同。这里是在隔离衰减机制，不是完整 Adam 一步更新的数值模拟。参考 [Adam](https://arxiv.org/abs/1412.6980) 与 [AdamW](https://arxiv.org/abs/1711.05101)。

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
  A：若状态都用 FP32，一份动量相对 Adam 的两份矩状态，可使优化器状态从每参数 8 bytes 降到 4 bytes，不是整个训练显存减半。在上面的全 FP32 口径下，模型状态总量是 16 → 12 bytes/param，激活与临时 buffer 还未计入。效果与调参需独立比较。

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

### AMP、累积与裁剪的顺序为什么不能交换？

一个优化器更新窗口内，应先完成所有 microbatch 的 backward；使用 loss scaling 时保持同一个 scale。然后依次做 **unscale → 全局梯度范数裁剪 → optimizer step → 更新 scale**，最后清理梯度。若先裁剪放大后的梯度，阈值实际也被错误缩放；若在累积中途 unscale 或更改 scale，后续梯度相加便不在同一尺度。

Inf/NaN 导致 optimizer step 被跳过时，按更新步数驱动的 scheduler 也应避免无条件前进。BF16 通常无需 scaler，但并不免疫溢出、错误 mask 或不稳定的归约。分片训练中的“全局范数”还需要正确聚合各分片，不能各卡独立裁剪后声称等价。库行为参见 [PyTorch 2.8 AMP 示例](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html)。

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

### 分片参数的一次生命周期

以计算后重新分片的 full-shard 策略为例，分片单位通常是一个参数组或模块，不必恰好一层：

| 时点 | 每张卡持有什么 | 通信与释放 |
|---|---|---|
| 空闲 / 更新后 | 自己的参数、梯度及优化器状态分片 | 常驻状态较小 |
| 某模块前向前 | 聚合该模块完整参数 | all-gather；前向后可释放完整参数 |
| 该模块反向前 | 再次聚合完整参数，读取保存的激活 | all-gather；计算输入梯度及参数梯度 |
| 该模块反向后 | 规约后的梯度分片 | reduce-scatter；释放完整参数及用完的激活 |
| optimizer step | 本卡参数分片及对应状态 | 每卡只更新自己负责的分片 |

prefetch 会提前持有下一组完整参数，通信与计算重叠能提速，却提高峰值显存。改变 reshard 策略也会改变反向是否再次 all-gather。不能用“常驻参数除以卡数”预测整轮峰值。参见 [PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

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
在平衡 stage、忽略通信、每个 microbatch 前反向代价固定的简化模型下，非交错流水线的 bubble 占比约为 $(p-1)/(m+p-1)$，其中 $p$ 是 stage 数，$m$ 是 microbatch 数。4 个 stage、8 个 microbatch 时约为 $3/11=27.3\%$；增加到 32 个时约为 $3/35=8.6\%$。

更多 microbatch 可以摊薄填充/排空开销，但太小的 microbatch 会损害算子效率。**1F1B** 相比先做完所有前向再反向，主要降低同时保存的激活数量，并不自动消除上述气泡；interleaved 调度可进一步改变气泡与通信开销。实际结果还取决于 stage 负载不均、通信和调度实现。参见 [Megatron-LM 大规模训练研究](https://arxiv.org/abs/2104.04473)。

### 追问

- **Q：3D 并行的具体维度怎么分？**
  A：DP 跨节点（带宽要求低），TP 节点内（高带宽 NVLink），PP 节点间（少 stage）。

- **Q：长上下文为什么要 SP？**
  A：activation 随序列长度增长，SP 可把 LayerNorm、dropout、残差等逐 token 运算沿序列维分片。但标准 attention 仍需要跨分片获取 K/V 或采用 ring/context parallel 通信；“每张卡只算一段”不等于没有跨卡依赖。

---

## Q08 · 预训练数据与计算预算

### 数据流水线为什么不只是“收集更多文本”？

原始语料需要经历来源与许可记录、解析、质量过滤、去重、评测去污染、数据配比和 tokenization。顺序会影响结果：例如把同一文档的不同版本先随机拆到 train/validation，再各自去重，仍会造成跨集合泄漏。许可与隐私要求应单独检查，不能把“网上可下载”当作可训练的充分条件。

| 环节 | 解决的问题 | 容易付出的代价 |
|---|---|---|
| 规则 / 分类器质量过滤 | 乱码、模板噪声、低信息密度 | 过强过滤会删除方言、小语种和非主流文体 |
| 精确去重 | 完全相同的文档反复出现 | 不能识别改写、局部复制或格式变化 |
| 近重复检测 | 用规范化、n-gram/MinHash 等找高重合文本 | 阈值过低会误删共享事实但有独立价值的内容 |
| 文档 / 来源分组切分 | 防止同源近重复跨训练和评测 | 分组不合理也会造成分布差异 |
| 评测去污染 | 检查题干、答案、解析和改写的泄漏 | 检测不完全，未命中不等于无污染 |

去重减少重复暴露与记忆风险，但不是无条件越强越好；应同时看有效 token 数、来源覆盖及分桶验证 loss。相关实证见 [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)。

### 数据配比为什么不能只按原始体量？

设领域 $k$ 有 $n_k$ 个 token，一种采样分布是 $p_k=n_k^\alpha/\sum_j n_j^\alpha$。$\alpha=1$ 按体量采样；$\alpha=0$ 在非空领域间均匀；介于两者之间会提高小领域占比。它是配比策略，不是普遍最优公式。

若训练总预算为 $D$ 个 token，领域 $k$ 的期望暴露次数约为 $Dp_k/n_k$。小领域被反复采样可能改善覆盖，也可能更早过拟合；要报告重复暴露率，不能把重复的 token 都算作新增知识。配比也可分阶段调整，但每次变化都应看分领域验证集及通用能力回归。

### Scaling law 如何帮助预算决策？

常见经验拟合用模型参数量 $N$、训练 token 数 $D$ 表示验证损失：

```math
L(N,D)\approx E+\frac{a}{N^\alpha}+\frac{b}{D^\beta},
\qquad C\approx 6ND
```

这里指数与系数需用实验拟合；$6ND$ 是稠密模型训练主干的粗略 FLOPs 口径，忽略长上下文 attention、重计算等额外项。在固定预算下，只增大模型会减少可训练 token，使模型可能训练不足；只增数据也会遇到容量限制。

[Chinchilla 研究](https://arxiv.org/abs/2203.15556) 给出了特定实验范围内的计算最优配比证据，不意味着“每参数固定配 20 个 token”是所有模型的定律。若部署期会生成海量 token，训练更小但更充分的模型可能降低总成本；训练计算最优与全生命周期成本最优不是同一个问题。

---

## Q09 · 有效 token、梯度累积与精确续训

### 为什么 microbatch 的平均 loss 不能直接再平均？

令 microbatch $j$ 的有效监督 token 数为 $n_j$，loss 总和为 $S_j$。若目标是每个有效 token 等权，正确目标为：

```math
L=\frac{\sum_j S_j}{\sum_j n_j}
=\sum_j\frac{n_j}{\sum_k n_k}\,\bar L_j
```

例：两个 microbatch 分别有 2 和 8 个有效 token，平均 loss 为 3 和 1。简单平均得到 2；按 token 加权为 $(2\times3+8\times1)/10=1.4$。梯度也有相同的权重差异。除以固定 accumulation steps 仅在各 microbatch 分母相同、或刻意追求 microbatch 等权时成立。

可以先获知整个累积窗口的有效 token 总数，再归一化每个 microbatch 的 loss sum；也可累积 sum 梯度后统一缩放。最后不足完整窗口的残余 batch 同样按真实分母处理。全 ignore 的窗口应跳过并记录，不能对零分母求平均。

### DDP 为什么还要考虑 world size？

默认 DDP 对 $W$ 张卡的梯度取平均。若全局有效 token 总数为 $N_{\mathrm{valid}}$，每卡累积自己的 loss sum 后，需令最终梯度等于 $\sum_r\nabla S_r/N_{\mathrm{valid}}$。因此可将每卡 loss sum 乘 $W/N_{\mathrm{valid}}$，补偿 DDP 的 $1/W$；不能再额外除一次 accumulation steps。

这是默认平均规约、无自定义通信 hook 的推导；如果框架已经处理 token 归一化，不能重复缩放。各 rank 应协调空 batch 与更新步数，否则可能通信挂起。中间 microbatch 可用 DDP 的 no_sync 减少通信，但上下文须覆盖前向和反向；FSDP 的不同累积策略可能额外保留完整梯度，需单独测显存。参见 [PyTorch DDP 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)。

### “恢复权重”与“精确续训”有什么区别？

要尽可能延续同一训练轨迹，除参数外，还需恢复 optimizer、scheduler、AMP scaler、各 rank 的随机数状态、sampler/dataloader 位置、累积步数和必要的未更新梯度。优先在完整 optimizer step 边界保存，可以简化恢复。

还应记录 tokenizer/chat template、数据版本与配比、并行配置、软件版本。即使这些都恢复，换硬件、world size 或非确定性 kernel 仍可能改变数值轨迹；应区分“可继续训练”“统计可复现”和“逐位一致”。

---

[⬅ 回到首页](../README.md)
