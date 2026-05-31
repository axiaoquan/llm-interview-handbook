# 99 · Frontier 前沿

## 📑 本章目录

- [Q01 · Diffusion LLM (DLM)](#q01--diffusion-llm-dlm)
- [Q02 · State Space Models / Mamba](#q02--state-space-models--mamba)
- [Q03 · Inference-time Scaling（o1 / DeepSeek-R1）](#q03--inference-time-scaling)
- [Q04 · MoE 最新进展（DeepSeek V2/V3 / Mixtral / Qwen3.5）](#q04--moe-最新进展)

---

## Q01 · Diffusion LLM (DLM)

### 🎯 核心思想

把扩散模型用于**文本生成**：不再自回归，而是**并行去噪**。

### 📖 工作流程

1. 输入是被 **mask 的 token 序列**（类似填空）
2. 模型一次预测**所有 mask 位置的 token**
3. 多步迭代去噪，逐步细化

### 📖 优势

- **并行生成**（不像自回归一次只出 1 个）
- 可双向利用上下文
- 推理速度可控（步数 ↔ 质量）

### ⚠️ 挑战

- **长度偏置**：朴素贪心倾向短输出
- **全局搜索**：传统 beam search 在去掩码过程中**缺乏全局上下文感知**

### 📖 GOS（两阶段全局最优搜索）

DLM 推理优化方向（用户研究方向相关）：

- **阶段一**：贪心确定回答长度（消除长度偏置）
- **阶段二**：在固定长度上做全局 beam search
  - 邻域加权机制：距离越近分数越高
  - 基于 HSIC 的 Beam 选择：防止退化

### 📖 代表工作

- **Diffusion-LM**（Stanford 2022）
- **SEDD**（Score-based Discrete Diffusion）
- **LLaDA**（2025）

---

## Q02 · State Space Models / Mamba

### 🎯 核心思想

用**状态空间模型**替代 attention，实现**线性复杂度**且不丢失长程建模能力。

### 📖 数学形式

$$
h_t = A h_{t-1} + B x_t, \quad y_t = C h_t
$$

类似 RNN，但**矩阵 A、B、C 与输入相关**（input-dependent），可学习"什么该记什么该忘"。

### 📖 与 Transformer 对比

| | Transformer | Mamba |
|---|---|---|
| 复杂度 | $O(n^2)$ | $O(n)$ |
| 并行训练 | ✅ | ✅（用 parallel scan） |
| 推理 | KV Cache 大 | 固定 hidden state |
| 长上下文 | 难 | 天然适合 |
| 召回能力 | 强 | 弱（精确召回不如 attention） |

### 📖 实际趋势

- 纯 Mamba **召回弱**于 Transformer
- 主流走**混合架构**：少量 attention 层 + 大量 SSM 层（如 Jamba、Qwen3.5）

---

## Q03 · Inference-time Scaling

### 🎯 范式转移

从**训练时投入更多算力** → **推理时投入更多算力**：

让模型"想得更久"，质量更好（OpenAI o1、DeepSeek-R1）。

### 📖 实现方式

- **Chain-of-Thought 长推理链**：模型先输出大段思考过程
- **Best-of-N 采样**：采 N 个答案选最好
- **Tree of Thoughts**：探索多条推理路径
- **Process Reward Model**：对推理过程的每一步打分

### 📖 DeepSeek-R1 关键贡献

- 用 **GRPO + 规则奖励**（数学题答案对错）做 RL
- **R1-Zero**：完全没有 SFT，纯 RL 也能涌现长推理
- 蒸馏到小模型：小模型也能有不错的推理能力

### 🪤 追问

- **Q：为什么 GRPO 适合 reasoning？**
  A：规则可验证（答案对错） + 多采样有意义 + 不要 critic 省资源。

---

## Q04 · MoE 最新进展

### 📖 DeepSeek-V2/V3

- **细粒度专家**：把每个专家拆得更小，激活更多个（默认 6 个 routed + 2 个 shared）
- **MLA**：极致压缩 KV Cache
- **辅助损失更轻**（auxiliary-loss-free）
- V3 是 671B 总参 / 37B 激活的开源最强 MoE

### 📖 Mixtral 8x7B

- 8 个专家，每 token 激活 2 个
- 总参 47B，激活 13B
- 早期开源 MoE 标杆

### 📖 Qwen3.5

- **混合架构**：线性注意力（Gated Delta Networks）+ 稀疏 MoE
- Qwen3.5-Plus / Flash 支持 **1M 上下文**
- 支持**混合思考模式**（thinking / non-thinking）

### 📖 趋势

1. **稀疏度越来越高**（激活比 < 5%）
2. **细粒度专家**（数量增多、单个变小）
3. **共享专家**（捕捉通用知识）
4. **路由器更稳**（noisy top-k、辅助 loss 调整）

---

[⬅ 回到首页](../README.md)
