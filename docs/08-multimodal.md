# 08 · Multimodal 多模态

## 本章目录

- [Q01 · CLIP 训练机制（对比学习）](#q01--clip-训练机制)
- [Q02 · BLIP / BLIP-2 / Q-Former](#q02--blip--blip-2--q-former)
- [Q03 · LLaVA / MiniGPT 架构](#q03--llava--minigpt-架构)
- [Q04 · Vision Encoder 选型（CLIP / SigLIP / DINOv2）](#q04--vision-encoder-选型)
- [Q05 · Qwen2.5-VL 设计要点](#q05--qwen25-vl-设计要点)
- [Q06 · 多模态 RAG](#q06--多模态-rag)
- [Q07 · Diffusion Model 基础](#q07--diffusion-model-基础)

---

## Q01 · CLIP 训练机制

### 一句话

CLIP = **图文对比学习**：让匹配的（图，文）对在向量空间靠近，不匹配的远离。

### 训练目标

InfoNCE 损失。对一个 batch 内 $N$ 对（图，文），构造 $N \times N$ 相似度矩阵，对角线为正样本：

$$
L = -\frac{1}{2N}\sum_i \left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]
$$

（图→文 + 文→图 双向对比）

### 关键点

- **大 batch 很重要**：batch 内其他样本充当负例，候选更多通常提升辨别难度，但也增加“语义相同却被当作负例”的 false negatives
- 训练数据是**图文配对的网页数据**（4 亿对）
- 推理时**零样本分类**：给定 N 个类别名 → 编码成文本向量 → 与图像向量做相似度

温度 $\tau$ 决定相似度 logit 的尺度：$\tau$ 小会让 softmax 更尖、强调 hardest negatives，同时也放大噪声配对的梯度；$\tau$ 大则信号过平。CLIP 学习的是跨模态全局对齐，不自动获得区域定位、计数或字符级 OCR，这些能力需要更细粒度视觉特征和训练目标。

### 追问

- **Q：CLIP 局限是什么？**
  A：1) 对**细粒度区分**弱（"金毛 vs 拉布拉多"）；2) 对**OCR/计数**等任务不行；3) **空间定位**能力差。

---

## Q02 · BLIP / BLIP-2 / Q-Former

### BLIP-2 关键创新：Q-Former

要把视觉特征接入 LLM，挑战是**视觉 token 太多**（一张图几百个）。

Q-Former 用一组**可学的 query token**（默认 32 个）通过 cross-attention 从视觉特征中"抽取"信息：

```
图像 → ViT → 视觉 tokens
                ↓
            cross-attention
                ↑
   Q-Former 的 N 个可学 query
                ↓
         32 个视觉摘要 token → 投影到 LLM 输入空间
```

### 优点

- 视觉 token 数量**可控**
- LLM 主体冻结，只训 Q-Former + 投影层
- 训练成本低

可学习 query 的作用类似固定容量的信息瓶颈：无论 ViT 输出多少 patch，都压成固定数量 token，控制 LLM 上下文成本；cross-attention 让每个 query 学会从视觉 token 中提取不同信息。代价是压缩可能丢失小目标和密集文字，所以 query 数不是越少越好，需要在信息保真与序列成本之间折中。

---

## Q03 · LLaVA / MiniGPT 架构

### LLaVA 极简架构

```
图像 → ViT (CLIP) → Projector → 语言模型
                                        ↑
                              文本 tokens
```

**核心思想**：直接用 projector 把视觉特征映射到 LLM embedding 空间，结构更简单。它不是证明 Q-Former “多余”，而是选择保留更多视觉 token、把跨模态融合压力交给 LLM；换来实现简单和信息保真，也增加上下文长度与 attention 成本。

### 训练两阶段

1. **Stage 1（Pretrain）**：只训投影层，用图文 pair 数据对齐
2. **Stage 2（Instruct Tuning）**：训投影层 + LLM。原始 LLaVA 使用 language-only GPT-4，以图像 captions、目标类别/边界框等文字化信息生成指令数据，并非让 GPT-4V 直接看图生成

不要把版本混成一个模型：原始 LLaVA 使用线性 projector；LLaVA-1.5 使用两层 MLP projector 和更高分辨率配置。视觉 token 数随分辨率、patch size 与裁剪策略变化，576 不是所有 LLaVA 的固定值。见[原始 LLaVA](https://arxiv.org/abs/2304.08485) 与 [LLaVA-1.5](https://arxiv.org/abs/2310.03744)。

### LLaVA vs BLIP-2

| | LLaVA | BLIP-2 |
|---|---|---|
| 视觉 token 数 | 保留 patch token；如 336/14 配置为 24×24=576 | 原论文典型配置为 32 个 query |
| 视觉 → LLM | 原始版线性；1.5 为 MLP | Q-Former 加投影 |
| 优点 | 简单、信息全 | token 少、推理快 |
| 缺点 | 输入 token 多 | 信息有损 |

---

## Q04 · Vision Encoder 选型

| 编码器 | 特点 | 训练目标 |
|---|---|---|
| **CLIP-ViT** | 文图对齐，语义强 | 对比学习 |
| **SigLIP** | CLIP 的改进，sigmoid 替代 softmax | 对比学习 |
| **DINOv2** | 自监督，纯视觉 | 自蒸馏 |
| **EVA-CLIP** | scaling up 的 CLIP | 对比 + MIM |

### 追问

- **Q：DINOv2 vs CLIP 怎么选？**
  A：DINOv2 视觉特征更细（适合 segmentation 等下游）；CLIP 语义对齐好（适合接 LLM）。

- **Q：SigLIP 比 CLIP 好在哪？**
  A：CLIP 对一整行/列样本做 softmax 竞争，loss 和全局候选集合耦合；SigLIP 把每个图文 pair 视为独立二分类，用 sigmoid loss 处理正负对，减少对全局 softmax 归一化和超大 batch all-gather 的依赖。它仍受负样本数量、采样和类别不平衡影响，不能简化成“小 batch 一定更好”。

---

## Q05 · Qwen2.5-VL 设计要点

### 模型能力

1. **万物识别**：扩展可识别图像类别
2. **精确视觉定位**：bbox 和 point，支持层级化定位 + 规范 JSON 输出
3. **文字识别和理解**：OCR 能力大幅提升 + 信息抽取
4. **特色文档解析**：精确识别文档文本 + 提取元素位置
5. **视频理解**：支持小时级超长视频
6. **视觉 Agent**：能操作电脑和手机

### 关键架构创新

#### 1) 时间和图像尺寸增强

- **空间维度**：动态把不同尺寸图像转化为不同长度 token，**直接用图像实际尺寸**表示坐标，不归一化 → 模型直接学图像尺度
- **时间维度**：动态 FPS 训练 + **绝对时间编码**，把 **mRoPE id 与时间流速对齐**

#### 2) 更简洁高效的视觉编码器

- 从头训原生**动态分辨率 ViT**
- **窗口注意力** 减少计算
- **GQA**（分组查询注意力）省 KV Cache
- **SwiGLU** 激活函数
- **RoPE** 编码位置
- **RMSNorm + 预归一化** 稳定训练

### 后训练

- SFT **100 万+ 指令数据**
- 多阶段 RL：**离线 DPO + 在线 GRPO**

### 追问

- **Q：Qwen2.5-VL 用绝对坐标而不归一化，好处是什么？**
  A：绝对像素坐标保留输入分辨率信息，便于把输出直接映射回原图；归一化坐标则跨分辨率更统一。两者各有代价，“绝对坐标一定更适合小物体”不是由坐标形式单独保证的，还取决于动态分辨率、视觉 token 密度、标注精度和训练分布。

---

## Q06 · 多模态 RAG

把图片、视频也作为知识库的一部分。

### 实现思路

1. **图像 → embedding**：用 CLIP / BGE-VL 等
2. **多模态向量库**：存图、文及其向量
3. **跨模态检索**：文 → 图 / 图 → 文 / 图+文 → 文

### 应用场景

- **医学影像 RAG**：结合医学知识库 + 多模态大模型生成带证据的影像报告
- **产品图搜**：用图找相似商品 + 描述
- **文档 + 图表混合检索**：PDF 内的文字和图表都能查

---

## Q07 · Diffusion Model 基础

### 核心思想

学习**逐步去噪**的过程：从纯噪声开始，逐步生成清晰图像。

### 前向过程（加噪）

定义 $T$ 步噪声调度，每步：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

最终 $x_T$ 接近纯高斯噪声。

前向链可以合并成一步采样：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。因此训练时不必真的从 1 走到 $t$，可以随机采一个时间步直接构造 $x_t$；多步只发生在生成阶段。这也是扩散训练能够并行的关键。

### 反向过程（去噪）

训练神经网络 $\epsilon_\theta(x_t, t)$ 预测每步加的噪声：

$$
L = \mathbb{E}_{x_0, t, \epsilon}\left[\Vert\epsilon - \epsilon_\theta(x_t, t)\Vert^2\right]
$$

推理时从纯噪声出发，反向走 $T$ 步生成图像。

### 主要变体

| 模型 | 关键创新 |
|---|---|
| **DDPM** | Diffusion 开山之作 |
| **DDIM** | 确定性采样，加速生成 |
| **Stable Diffusion** | Latent Diffusion，在 VAE 隐空间做扩散 |
| **Flow Matching** | 用 ODE 替代 SDE，连续生成 |

### Diffusion vs LLM

| | Diffusion | LLM |
|---|---|---|
| 范式 | 并行去噪 | 串行下一 token |
| 推理速度 | 慢（多步） | 自回归慢 |
| 控制 | classifier-free guidance | prompting |
| 用途 | 图、视频生成 | 文本生成 |

> **DLM（Diffusion LLM）** = 把扩散思想用到文本生成，详见 [99 · Frontier](99-frontier.md)。

---

[⬅ 回到首页](../README.md)
