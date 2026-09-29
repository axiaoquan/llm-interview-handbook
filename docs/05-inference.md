# 05 · Inference 推理优化

推理速度 / 显存 / 长上下文相关。

## 本章目录

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

### 核心思想

每生成一个新 token，需要对所有之前的 token 计算 K 和 V。
KV Cache 把历史 token 已计算的 K、V **缓存**起来，避免每个 decode step 重算旧 token 的投影 → **空间换时间**。它不缓存 Q，因为历史 query 不会再次使用；当前 query 仍要与全部历史 key 做点积，所以单步 attention 仍随上下文长度线性增长。

### KV Cache 显存计算

$$
\text{KV Cache} = 2 \times B \times L \times H \times D \times N \times \text{sizeof(dtype)}
$$

其中：B = batch、L = seq length、H = **KV 头数**、D = d_head、N = n_layers。MHA 的 Q/KV 头数相同，GQA/MQA 则不能把 query 头数代入。

- 系数 2：K 和 V 各一份
- 例：LLaMA-7B（32 层、32 头、d=128），batch=1，seq=2048，FP16
  $2 \times 1 \times 2048 \times 32 \times 128 \times 32 \times 2 = 1\text{ GiB}$，约 1.074 GB；此处未计分页及元数据开销。

→ batch / 上下文一拉长，KV Cache 直接就是几十 GB，**经常成为显存瓶颈**。

### 如何减少 KV Cache？

| 方法 | 原理 | 减少比例 |
|---|---|---|
| **MQA** | 所有 Q 头共享 1 个 KV 头 | $1/h$ |
| **GQA** | 每组 Q 头共享 1 个 KV 头 | $g/h$ |
| **MLA** | 缓存潜在向量及必要的位置 key | 取决于潜在维度、位置维度和原 KV 形状，不是固定比例 |
| **量化** | KV Cache 用 INT8/INT4 | 50% / 75% |
| **窗口注意力** | 只保留最近的 KV | window/total |

### 追问

- **Q：Encoder 用 KV Cache 吗？**
  A：普通一次性 encoder self-attention 不需要跨步 cache；但 encoder-decoder 模型在生成时可以缓存 encoder 输出投影得到的 cross-attention K/V。准确说，cache 服务于“多次解码重复使用的 K/V”，并非只要叫 Encoder 就绝对不用。

- **Q：prefill 阶段需要 KV Cache 吗？**
  A：prefill 阶段一次算完整 prompt，**生成 KV Cache**；decode 阶段每步**读取 + 追加** KV Cache。

---

## Q02 · Flash Attention

### 核心问题

标准 Attention 需要算 $n \times n$ 的注意力矩阵 → 显存 $O(n^2)$。

### 核心思想

**分块计算 + 软最大值合并**：

- 把 K 和 V **分块**，从 HBM（显存）加载到 SRAM（芯片）中计算
- 用 softmax 的**平移不变性**，算局部最大值再合并

### 效果

- **中间显存**：不再物化完整 $n\times n$ score/probability 矩阵，额外存储从 $O(n^2)$ 降到近似 $O(n)$
- **速度**：通过减少 HBM IO 等开销提速；收益取决于形状、硬件及比较基线，不给通用固定倍数

### 关键洞察

GPU attention 在许多形状下受**显存带宽（IO）**限制。Flash Attention 的本质是 IO-aware：分块把 Q/K/V 搬进片上 SRAM，利用 online softmax 维护每行的运行最大值和归一化和，在不保存完整注意力矩阵的情况下得到与标准 attention 等价的结果。它没有把 dense attention 的理论 FLOPs 从 $O(n^2)$ 变成线性；速度收益也随序列长度、head dimension、mask 和硬件而变化。

### Online softmax 为什么能合并不同块？

固定一个 query，令已处理分数的最大值为 $m$，指数和为 $\ell$，未归一化的加权 value 和为向量 $u$。新块分数为 $s_j$、value 为 $v_j$，先更新最大值 $m'$，再统一指数的参考点：

```math
m'=\max\!\left(m,\max_j s_j\right)
```

```math
\ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'},
\qquad
u'=e^{m-m'}u+\sum_j e^{s_j-m'}v_j,
\qquad o'=\frac{u'}{\ell'}
```

旧指数原本以 $m$ 为基准，新块可能带来更大最大值；乘 $e^{m-m'}$ 后就与新块同尺度。分子、分母同时缩放，归一化结果不变。初始化可设 $m=-\infty,\ell=0,u=0$，但首块或整行全 masked 时须显式处理，避免 $-\infty-(-\infty)$ 与零分母。

手算两个分数 $[0,\log2]$，对应标量 value $[1,3]$。第一块后 $m=0,\ell=1,u=1$；第二块令 $m'=\log2$，旧项乘 $1/2$，得到 $\ell'=1.5,u'=3.5$，输出 $7/3$，与权重 $[1/3,2/3]$ 的直接计算一致。

反向可利用保存的归一化统计量和输出，分块重算概率而不保存整张矩阵，以额外计算换 IO 和存储。这里的“exact”指未改变 dense attention 的数学算子，不保证不同精度、求和顺序下逐位相同；训练 dropout 的反向还需复现相应随机掩码。来源：[FlashAttention](https://arxiv.org/abs/2205.14135)。

### 追问

- **Q：Flash Attention 和稀疏 attention 有什么区别？**
  A：Flash Attention **不改变数学结果**，只是更高效计算；稀疏 attention（Longformer 等）会**改变 attention 模式**，是近似。

- **Q：Flash Attention v2 / v3 改了什么？**
  A：v2 进一步减少非矩阵乘法 FLOPs；v3 利用 H100 的异步特性。

---

## Q03 · PagedAttention（vLLM）

### 解决什么问题

传统 KV Cache 需要预分配连续显存（"我猜这个请求会生成 N 个 token"），
但**实际生成长度不确定** → 大量显存浪费。

### 做法

不要把每个请求的 KV Cache 强行存成一大段连续内存，而是**像操作系统的分页**：

- 把 KV Cache 拆成**固定大小的 block（页）**
- 用一个**映射表**记录"逻辑上的第几段上下文"对应"物理上的哪一块显存"

### 效果

- 显著减少为最大生成长度预留造成的内部/外部碎片
- 允许在同样显存下容纳更多并发请求，吞吐通常因此提升
- 不同请求之间还能**共享** prefix（system prompt 等）

### 追问

- **Q：PagedAttention 有性能损失吗？**
  A：映射查找和非连续访存有开销。在受 KV 容量限制的并发场景中，减少浪费可能允许更大 batch 并提高吞吐；小 batch、短上下文或不同 kernel 下未必净收益。应同时比较显存占用、吞吐和延迟。

---

## Q04 · Continuous Batching

### 核心思想

**静态 batching**：整批请求一起开始，必须等这批最慢的也结束才能进下一批 → 拖累整体。

**Continuous batching**：**动态插入**，某请求一结束立刻移出，新请求马上加入。

### 好处

- GPU 利用率高
- 总吞吐量高
- 延迟更好（短请求不被长请求拖累）

### 追问

- **Q：和 PagedAttention 什么关系？**
  A：互补。PagedAttention 解决"显存高效共享"，Continuous Batching 解决"调度高效"。vLLM 同时用两者。

---

## Q05 · Speculative Decoding 投机解码

### 核心思想

让一个**更小、更快的草稿模型**一次猜出后面几步 token，再让**大模型一次并行检查**这些猜测。

如果猜对了，等于**用一次大模型前向 commit 多个 token**，而不是普通自回归"一次出 1 个"。

### 流程

1. **Draft model** 自回归生成 $\gamma$ 个 token
2. **Target model 一次前向**验证这 $\gamma$ 个 token
3. 对草稿 token $x$ 按 $\min(1, p(x)/q(x))$ 接受，其中 $p$ 是 target 分布、$q$ 是 draft 分布；在第一个拒绝处停止
4. 拒绝时从校正后的残差分布 $\mathrm{norm}(\max(0,p-q))$ 采样；若整段都接受，还可从 target 的下一位置再采一个 token

### 关键性质

上述接受—拒绝校正保证采样分布与只用 target model 相同，因此是分布意义上的无损加速。若实现只是比较 argmax 是否一致，或直接保留“看起来猜对”的 token，则不具备这个保证。

### 为什么拒绝后的修正分布恰好补齐概率？

固定相同前缀，记接受率为 $A=\sum_x\min(p(x),q(x))$。草稿采到 $x$ 并被接受的概率质量是 $q(x)\min(1,p(x)/q(x))=\min(p(x),q(x))$；$q(x)=0$ 时该项按零处理。

当 $A<1$ 时，拒绝概率为 $1-A=\sum_x[p(x)-q(x)]_+$，其中 $[z]_+=\max(z,0)$。因此从残差分布采样所增加的质量为：

```math
(1-A)\frac{[p(x)-q(x)]_+}{1-A}
=[p(x)-q(x)]_+
```

两部分之和 $\min(p(x),q(x))+[p(x)-q(x)]_+=p(x)$。若 $A=1$ 则不会进入拒绝分支，不能再对零残差归一化。逐前缀应用这个结论，才得到序列分布的一致性，而不只是单步巧合。

例：词表三个 token，$p=[0.5,0.3,0.2]$，$q=[0.2,0.5,0.3]$。接受部分质量为 $[0.2,0.3,0.2]$，总接受率 0.7；拒绝时残差只落到第一个 token，补上 0.3，最终正好恢复 $p$。

这里的 $p,q$ 必须是**实际使用的归一化分布**。若普通 target 解码包含温度、top-p 或重复惩罚，验证与残差也要使用同一套 target 变换；不能采样时截断、算接受率时却用原始概率。对有限精度误差、词表对齐和 EOS 也须定义一致语义。来源：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)。

### 拒绝后为什么要回滚 KV Cache？

Target 并行验证时可能已计算被拒绝 token 及其后缀的 KV；这些状态基于未被采用的前缀，不能继续复用。逻辑缓存只保留已接受前缀，再正确处理修正 token；draft 状态也要对齐。EOS 一旦被采纳，应停止该序列，不能为凑够草稿块继续输出。缓存回滚错了，即使接受公式正确，也不再是同一个条件分布。

### 局限

如果草稿模型太差，接受率低，大模型经常要回退修正，**额外的草稿开销可能抵消收益**。

### 追问

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
\mathrm{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t \mid y_{1:t-1})
$$

- 用 log 概率之和（连乘易下溢）
- 加**长度惩罚** $\alpha$ 防止偏向短句

每个新增 token 的 logprob 通常非正，原始 logprob 之和容易偏向短序列；除以长度幂会改变排序，并非单纯改善数值稳定性。例如长度 2、4 的候选，logprob 和为 -2、-3：原始分数选前者，$\alpha=1$ 时分数为 -1、-0.75，转而选后者。长度归一化是在改变搜索目标，也可能偏向冗长。

完成的 EOS 候选应与未完成 beam 分开管理，不再继续扩展。提前停止应比较完成候选与未完成候选在当前评分规则下的可达上界；有长度归一化时，不能简单认为“第一个 EOS 出现就找到最优答案”。beam 再大也只是更充分地搜索给定模型评分，不保证事实性、任务奖励或人类偏好更好。

### 3) Temperature Sampling 温度采样

$$
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

| T | 效果 |
|---|---|
| T < 1 | 分布尖锐，更确定 |
| T = 1 | 原始分布 |
| T > 1 | 分布平坦，更随机 |

$T=0$ 时除法没有定义；严格说是 $T\to0^+$ 时分布收敛到最大 logit 的 token（并列最大时还需 tie-breaking）。工程实现应显式走 greedy 分支。即使 greedy，硬件非确定性、量化和不同 kernel 仍可能在近似并列的 logits 上产生不同结果。

### 4) Top-K Sampling

只从概率最大的 **K 个 token** 中采样 → 排除长尾垃圾 token。

### 5) Top-P (Nucleus) Sampling

从**累积概率达到 $p$** 的最小 token 集合中采样 → 自动调整候选集大小。

### Top-K vs Top-P

| 特性 | Top-K | Top-P |
|---|---|---|
| 候选集大小 | 固定 K | 动态 |
| 分布尖锐时 | 可能含低概率词 | 自动缩小 |
| 分布平坦时 | 可能排除合理词 | 自动扩大 |
| 推荐 | 简单场景 | 更灵活 |

实际部署常**同时用** Temperature + Top-P：先温度调形状，再 nucleus 截断。

---

## Q07 · 重复惩罚

防止生成重复内容。常见 repetition penalty 作用在 **logit** 而非 softmax 后概率上，并对正负 logit 分段处理，避免负 logit 除以 $\alpha$ 后反而变大：

```math
z_i'=
\begin{cases}
z_i/\alpha, & z_i>0\ \text{且 token }i\text{ 已出现}\\
\alpha z_i, & z_i<0\ \text{且 token }i\text{ 已出现}\\
z_i, & \text{其他}
\end{cases}
```

$\alpha>1$ 时降低已出现 token 的相对 logit，之后再统一 softmax。直接修改概率还必须重新归一化，且与常见库实现口径不同。

| 类型 | 公式 |
|---|---|
| **Repetition Penalty** | 对已生成 token 的 logit 做乘/除惩罚 |
| **Frequency Penalty** | 按出现**次数线性**惩罚 |
| **Presence Penalty** | 出现过就**惩罚固定值** |

---

## Q08 · 模型量化

### 核心思想

把模型权重和/或激活值从高精度（FP32/FP16）转换为低精度（INT8/INT4）。
浮点数 → 整数网格 → 计算时反量化。

均匀仿射量化通常写成 $q=\mathrm{clip}(\mathrm{round}(x/s)+z)$，其中 scale $s$ 决定网格间距，zero-point $z$ 让实数 0 可被精确表示。粒度可按 tensor、channel 或 group：group 越小越能适应局部离群值，但需要存更多 scale/zero-point，也增加 kernel 复杂度。低比特的主要难点往往不是平均误差，而是少量离群通道决定了量化范围，挤压了大多数权重可用的码点。

### 量化的三条轴

- **按位宽**：FP16/BF16、INT8、INT4、INT2
- **按对象**：只量化权重 vs 连激活也量化（**weight-only** 最常见）
- **按时机**：训练后量化 PTQ vs 量化感知训练 QAT

### PTQ vs QAT

| 方法 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| **PTQ** | 训练完成后直接量化 | 简单快速 | 低比特精度损失大 |
| **QAT** | 训练时模拟量化 | 精度保持好 | 训练成本高 |

### 主流量化方法

| 方法 | 位宽 | 类型 | 关键思想 |
|---|---|---|---|
| **INT8** | 8-bit | 基线 | 直接映射到 8-bit 整数 |
| **GPTQ** | 3/4-bit | PTQ | 用近似 Hessian 指导量化，考虑误差对输出的影响 |
| **AWQ** | 4-bit | PTQ | "**并非所有权重都同样重要**"——根据**激活分布**保护关键通道 |
| **GGUF** | 2-8 bit | PTQ | 不是量化算法，是**文件格式**（GGML 的存储） |
| **QLoRA** | 基座 4-bit | 量化基座上的 PEFT | 冻结量化权重，仅训练 LoRA 适配器 |

### 几个常见误区

- **GGUF**：本质是**文件格式**，不是量化算法
- **QLoRA**：是冻结量化基座上的微调方案，不应直接等同于更新基座、模拟量化误差的常规 QAT；量化后训练了 adapter 不代表量化权重本身可学习。见 [QLoRA 原论文](https://arxiv.org/abs/2305.14314)
- **权重大小缩小 4× 不等于推理加速 4×**：还取决于是否有对应低比特 kernel、反量化开销、memory-bound 程度和 batch shape

### 选型

| 方法 | 位宽 | 类型 | 适用场景 |
|---|---|---|---|
| FP16 | 16 | 基线 | GPU 充足 |
| INT8 | 8 | PTQ | 轻度压缩 |
| GPTQ | 3/4 | PTQ | GPU 推理 |
| AWQ | 4 | PTQ | GPU 推理；效果需在相同模型、校准集、group size 与后端上比较 |
| GGUF | 2-8 | PTQ | CPU 推理（llama.cpp） |
| QLoRA | 基座 4 | PEFT | 显存受限的微调 |

### 追问

- **Q：AWQ 比 GPTQ 好在哪？**
  A：AWQ 利用校准激活识别敏感通道，通过等价缩放改变权重量化的相对误差；不是简单让关键通道保留 FP16。以行向量约定，$XW=(XD^{-1})(DW)$，量化 $DW$ 后，逆缩放可降低重要通道误差的影响，但也会改变其他通道的量化范围，因此要搜索缩放。GPTQ 则利用近似二阶信息补偿逐步量化误差。谁更好取决于校准数据、位宽、粒度和 kernel，不能无来源地声称 AWQ 在中文上总更好。见 [AWQ 原论文](https://arxiv.org/abs/2306.00978)。

---

## Q09 · 长上下文

### RoPE 外推方案

| 方法 | 思想 |
|---|---|
| **NTK-aware Scaling** | 调整 RoPE base，让低频维度变化更慢 |
| **YaRN** | NTK 进阶版，对不同频率分段处理 |
| **Position Interpolation (PI)** | 把测试位置缩放到训练范围内 |

这些方法本质上在重新安排不同频率维度的相位增长，避免测试长度下出现训练未见的高频旋转。只把 `max_position_embeddings` 改大不会改变 RoPE 相位，也不会自动获得长上下文能力；通常还需要长序列继续训练/微调，并检查短上下文退化、远距离检索和不同位置分桶表现。

### 注意力优化

- **Sliding Window Attention**（Mistral）：只关注最近 window 内的 token
- **StreamingLLM**：保留 attention sink + 滑窗
- **Ring Attention**：把 KV 切到多卡，绕一圈算完

### 追问

- **Q：长上下文最大瓶颈是什么？**
  A：**KV Cache 显存** + Attention $O(n^2)$ 计算。前者用 GQA/MLA + 分页存储缓解，后者用 Flash Attention + 稀疏化缓解。

---

[⬅ 回到首页](../README.md)
