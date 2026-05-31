# 05 · Inference 推理优化

推理速度 / 显存 / 长上下文相关。

## 📑 本章目录

- [Q01 · KV Cache](#q01--kv-cache)
- [Q02 · Flash Attention](#q02--flash-attention)
- [Q03 · PagedAttention（vLLM）](#q03--pagedattentionvllm)
- [Q04 · Continuous Batching](#q04--continuous-batching)
- [Q05 · Speculative Decoding 投机解码](#q05--speculative-decoding-投机解码)
- [Q06 · 解码策略（Greedy / Beam / 采样）](#q06--解码策略)
- [Q07 · 重复惩罚](#q07--重复惩罚)
- [Q08 · 模型量化（PTQ / QAT / GPTQ / AWQ）](#q08--模型量化)
- [Q09 · 长上下文（YaRN / NTK-aware）](#q09--长上下文)

---

## Q01 · KV Cache

### 🎯 核心思想

每生成一个新 token，需要对所有之前的 token 计算 K 和 V。
KV Cache 把已计算的 K、V **缓存**起来，避免重复计算 → **空间换时间**。

### 📖 KV Cache 显存计算

$$
\text{KV Cache} = 2 \times \text{batch} \times \text{seq\_len} \times n_{heads} \times d_{dim} \times \text{layers} \times \text{sizeof(dtype)}
$$

- 系数 2：K 和 V 各一份
- 例：LLaMA-7B（32 层、32 头、d=128），batch=1，seq=2048，FP16
  $\approx 2 \times 1 \times 2048 \times 32 \times 128 \times 32 \times 2 = 1\text{ GB}$

→ batch / 上下文一拉长，KV Cache 直接就是几十 GB，**经常成为显存瓶颈**。

### 📖 如何减少 KV Cache？

| 方法 | 原理 | 减少比例 |
|---|---|---|
| **MQA** | 所有 Q 头共享 1 个 KV 头 | $1/h$ |
| **GQA** | 每组 Q 头共享 1 个 KV 头 | $g/h$ |
| **MLA** | 压缩到低维潜在空间 | ~6.7% |
| **量化** | KV Cache 用 INT8/INT4 | 50% / 75% |
| **窗口注意力** | 只保留最近的 KV | window/total |

### 🪤 追问

- **Q：Encoder 用 KV Cache 吗？**
  A：不用。Encoder 一次性看到所有 token，没有"逐 token 生成"的过程。KV Cache 只在 **Decoder 自回归生成**时有用。

- **Q：prefill 阶段需要 KV Cache 吗？**
  A：prefill 阶段一次算完整 prompt，**生成 KV Cache**；decode 阶段每步**读取 + 追加** KV Cache。

---

## Q02 · Flash Attention

### 🎯 核心问题

标准 Attention 需要算 $n \times n$ 的注意力矩阵 → 显存 $O(n^2)$。

### 📖 核心思想

**分块计算 + 软最大值合并**：

- 把 K 和 V **分块**，从 HBM（显存）加载到 SRAM（芯片）中计算
- 用 softmax 的**平移不变性**，算局部最大值再合并

### 📊 效果

- **显存**：$O(n^2) \to O(n)$
- **速度**：2-4× 加速（不是因为减少 FLOPs，而是减少 HBM IO）

### 📖 关键洞察

GPU 的瓶颈往往是**显存带宽（IO）**，不是计算。Flash Attention 的本质是
**IO-aware** 算法：减少 HBM 读写次数，把更多操作 fuse 到 SRAM 内。

### 🪤 追问

- **Q：Flash Attention 和稀疏 attention 有什么区别？**
  A：Flash Attention **不改变数学结果**，只是更高效计算；稀疏 attention（Longformer 等）会**改变 attention 模式**，是近似。

- **Q：Flash Attention v2 / v3 改了什么？**
  A：v2 进一步减少非矩阵乘法 FLOPs；v3 利用 H100 的异步特性。

---

## Q03 · PagedAttention（vLLM）

### 🎯 解决什么问题

传统 KV Cache 需要预分配连续显存（"我猜这个请求会生成 N 个 token"），
但**实际生成长度不确定** → 大量显存浪费。

### 📖 做法

不要把每个请求的 KV Cache 强行存成一大段连续内存，而是**像操作系统的分页**：

- 把 KV Cache 拆成**固定大小的 block（页）**
- 用一个**映射表**记录"逻辑上的第几段上下文"对应"物理上的哪一块显存"

### 📊 效果

- 显存利用率接近 **100%**
- 吞吐量提升 **2-4×**
- 不同请求之间还能**共享** prefix（system prompt 等）

### 🪤 追问

- **Q：PagedAttention 有性能损失吗？**
  A：访存有少量额外开销（查映射表），但显存利用率提升带来的更大 batch 远远抵消。

---

## Q04 · Continuous Batching

### 🎯 核心思想

**静态 batching**：整批请求一起开始，必须等这批最慢的也结束才能进下一批 → 拖累整体。

**Continuous batching**：**动态插入**，某请求一结束立刻移出，新请求马上加入。

### 📊 好处

- GPU 利用率高
- 总吞吐量高
- 延迟更好（短请求不被长请求拖累）

### 🪤 追问

- **Q：和 PagedAttention 什么关系？**
  A：互补。PagedAttention 解决"显存高效共享"，Continuous Batching 解决"调度高效"。vLLM 同时用两者。

---

## Q05 · Speculative Decoding 投机解码

### 🎯 核心思想

让一个**更小、更快的草稿模型**一次猜出后面几步 token，再让**大模型一次并行检查**这些猜测。

如果猜对了，等于**用一次大模型前向 commit 多个 token**，而不是普通自回归"一次出 1 个"。

### 📖 流程

1. **Draft model** 自回归生成 $\gamma$ 个 token
2. **Target model 一次前向**验证这 $\gamma$ 个 token
3. 从第一个不匹配位置开始 reject，保留匹配的
4. 从不匹配位置重新采样一个正确的 token

### 🎯 关键性质

生成结果的分布**与仅用大模型生成完全一致**（无损加速）。

### ⚠️ 局限

如果草稿模型太差，接受率低，大模型经常要回退修正，**额外的草稿开销可能抵消收益**。

### 🪤 追问

- **Q：草稿模型怎么选？**
  A：要求快 + 与目标模型行为相似。常见做法：
  1. 用同系列小模型（LLaMA-7B 配 1B 草稿）
  2. **EAGLE / Medusa** 等：在大模型基础上加几个轻量头自己当草稿

- **Q：和 self-speculation 有什么关系？**
  A：Medusa 等用"模型自己做草稿"，省了部署两个模型的麻烦。

---

## Q06 · 解码策略

语言模型每一步生成一个概率分布，解码策略要决定**怎么选 token**。

### 1) Greedy 贪心

每一步选概率最大的 token。

**优点**：快、稳定、实现简单
**缺点**：易陷入重复，缺乏多样性，局部最优 ≠ 全局最优

适合：**确定性任务、结构化输出、不强调多样性**

### 2) Beam Search

每一步保留 $b$ 个**最好的候选序列**，比较的是整条路径得分：

$$
\mathrm{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t \mid y_{<t})
$$

- 用 log 概率之和（连乘易下溢）
- 加**长度惩罚** $\alpha$ 防止偏向短句

### 3) Temperature Sampling 温度采样

$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

| T | 效果 |
|---|---|
| T < 1 | 分布尖锐，更确定 |
| T = 1 | 原始分布 |
| T > 1 | 分布平坦，更随机 |

**T = 0** 等价于贪心（理论上）。但实际中由于不同推理后端、数值精度差异，**最高 token 可能产生微小波动**，导致输出仍可能不一致。

### 4) Top-K Sampling

只从概率最大的 **K 个 token** 中采样 → 排除长尾垃圾 token。

### 5) Top-P (Nucleus) Sampling

从**累积概率达到 $p$** 的最小 token 集合中采样 → 自动调整候选集大小。

### 📊 Top-K vs Top-P

| 特性 | Top-K | Top-P |
|---|---|---|
| 候选集大小 | 固定 K | 动态 |
| 分布尖锐时 | 可能含低概率词 | 自动缩小 |
| 分布平坦时 | 可能排除合理词 | 自动扩大 |
| 推荐 | 简单场景 | 更灵活 |

实际部署常**同时用** Temperature + Top-P：先温度调形状，再 nucleus 截断。

---

## Q07 · 重复惩罚

防止生成重复内容。

$$
P'(x_i) =
\begin{cases}
P(x_i) / \alpha, & x_i \in \text{generated} \\
P(x_i), & \text{otherwise}
\end{cases}
$$

$\alpha > 1$ 时，出现过的 token 会被压低。

| 类型 | 公式 |
|---|---|
| **Repetition Penalty** | 已生成 token 概率除以 $\alpha$（典型 1.1） |
| **Frequency Penalty** | 按出现**次数线性**惩罚 |
| **Presence Penalty** | 出现过就**惩罚固定值** |

---

## Q08 · 模型量化

### 🎯 核心思想

把模型权重和/或激活值从高精度（FP32/FP16）转换为低精度（INT8/INT4）。
浮点数 → 整数网格 → 计算时反量化。

### 📊 量化的三条轴

- **按位宽**：FP16/BF16、INT8、INT4、INT2
- **按对象**：只量化权重 vs 连激活也量化（**weight-only** 最常见）
- **按时机**：训练后量化 PTQ vs 量化感知训练 QAT

### 📊 PTQ vs QAT

| 方法 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| **PTQ** | 训练完成后直接量化 | 简单快速 | 低比特精度损失大 |
| **QAT** | 训练时模拟量化 | 精度保持好 | 训练成本高 |

### 📖 主流量化方法

| 方法 | 位宽 | 类型 | 关键思想 |
|---|---|---|---|
| **INT8** | 8-bit | 基线 | 直接映射到 8-bit 整数 |
| **GPTQ** | 3/4-bit | PTQ | 用近似 Hessian 指导量化，考虑误差对输出的影响 |
| **AWQ** | 4-bit | PTQ | "**并非所有权重都同样重要**"——根据**激活分布**保护关键通道 |
| **GGUF** | 2-8 bit | PTQ | 不是量化算法，是**文件格式**（GGML 的存储） |
| **QLoRA** | 4-bit | QAT | **量化微调**：冻结 4-bit 模型，训 LoRA 适配器 |

### 📖 几个常见误区

- **GGUF**：本质是**文件格式**，不是量化算法
- **QLoRA**：是**微调方案**，不是推理量化

### 📊 选型

| 方法 | 位宽 | 类型 | 适用场景 |
|---|---|---|---|
| FP16 | 16 | 基线 | GPU 充足 |
| INT8 | 8 | PTQ | 轻度压缩 |
| GPTQ | 3/4 | PTQ | GPU 推理 |
| AWQ | 4 | PTQ | GPU 推理（中文常优于 GPTQ） |
| GGUF | 2-8 | PTQ | CPU 推理（llama.cpp） |
| QLoRA | 4 | QAT | 微调 |

### 🪤 追问

- **Q：AWQ 比 GPTQ 好在哪？**
  A：基于"激活离群值"的洞察，少量关键通道保高精度，整体精度更稳，对中文等长尾分布尤其友好。

---

## Q09 · 长上下文

### 📖 RoPE 外推方案

| 方法 | 思想 |
|---|---|
| **NTK-aware Scaling** | 调整 RoPE base，让低频维度变化更慢 |
| **YaRN** | NTK 进阶版，对不同频率分段处理 |
| **Position Interpolation (PI)** | 把测试位置缩放到训练范围内 |

### 📖 注意力优化

- **Sliding Window Attention**（Mistral）：只关注最近 window 内的 token
- **StreamingLLM**：保留 attention sink + 滑窗
- **Ring Attention**：把 KV 切到多卡，绕一圈算完

### 🪤 追问

- **Q：长上下文最大瓶颈是什么？**
  A：**KV Cache 显存** + Attention $O(n^2)$ 计算。前者用 GQA/MLA + 分页存储缓解，后者用 Flash Attention + 稀疏化缓解。

---

[⬅ 回到首页](../README.md)
