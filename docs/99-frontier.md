# 99 · Frontier 前沿

## 本章目录

- [Q01 · Diffusion LLM (DLM)](#q01--diffusion-llm-dlm)
- [Q02 · State Space Models / Mamba](#q02--state-space-models--mamba)
- [Q03 · Inference-time Scaling（o1 / DeepSeek-R1）](#q03--inference-time-scaling)
- [Q04 · MoE 最新进展（细粒度专家 / 混合稀疏架构）](#q04--moe-最新进展)

---

## Q01 · Diffusion LLM (DLM)

### 核心思想

把扩散模型用于**文本生成**：不再自回归，而是**并行去噪**。

### 工作流程

1. 输入是被 **mask 的 token 序列**（类似填空）
2. 模型一次预测**所有 mask 位置的 token**
3. 多步迭代去噪，逐步细化

### 优势

- 单个去噪 step 可并行更新多个位置（不像自回归严格按 token 顺序）
- 可双向利用上下文
- 推理速度可控（步数 ↔ 质量）

### 挑战

- **长度偏置**：朴素贪心倾向短输出
- **全局搜索**：传统 beam search 在去掩码过程中**缺乏全局上下文感知**

“位置并行”不等于端到端一定更快：DLM 需要多个去噪 step，每步可能仍处理整段序列；若反复修改大量低置信位置，总计算可能超过自回归 decode。比较时应在相同质量下看去噪步数、每步序列长度、状态复用能力和 wall-clock，而不是只看是否并行。

### 代表工作

- **Diffusion-LM**（Stanford 2022）
- **SEDD**（Score-based Discrete Diffusion）
- **LLaDA**（2025）

---

## Q02 · State Space Models / Mamba

### 核心思想

用**状态空间模型**替代 attention，使序列计算线性增长、decode state 保持固定大小，并尝试保留长程建模能力。

### 数学形式

$$
h_t = A h_{t-1} + B x_t, \quad y_t = C h_t
$$

基础线性 SSM 的 $A,B,C$ 固定；Mamba 的 selective 机制让步长 $\Delta$ 以及 $B,C$ 由当前输入生成，而 $A$ 通常仍是学习到但不随 token 变化的结构化参数。输入依赖的离散化控制哪些信息写入、读出，selective scan 则把递推高效实现为并行训练。

### 与 Transformer 对比

| | Transformer | Mamba |
|---|---|---|
| 复杂度 | $O(n^2)$ | $O(n)$ |
| 并行训练 | 支持 | 支持（用 parallel scan） |
| 推理 | KV Cache 大 | 固定 hidden state |
| 长上下文 | 难 | 天然适合 |
| 召回能力 | 强 | 弱（精确召回不如 attention） |

### 实际趋势

- 纯 Mamba **召回弱**于 Transformer
- 主流走**混合架构**：少量 attention 层 + 大量 SSM 层（如 Jamba、Qwen3.5）

---

## Q03 · Inference-time Scaling

### 范式转移

从**训练时投入更多算力** → **推理时投入更多算力**：

让模型"想得更久"，质量更好（OpenAI o1、DeepSeek-R1）。

### 实现方式

- **Chain-of-Thought 长推理链**：模型先输出大段思考过程
- **Best-of-N 采样**：采 N 个答案选最好
- **Tree of Thoughts**：探索多条推理路径
- **Process Reward Model**：对推理过程的每一步打分

推理时扩展只有在“额外计算能产生有差异的候选”且“选择器能识别更好候选”时才有效。Best-of-N 的上限常受 verifier/reward model 限制：选择器分不清时，更多采样只产生不可用多样性；选择器有偏时还会放大 reward hacking。应同时报告候选 oracle 上限、选择后准确率和每题计算预算。

### DeepSeek-R1 关键贡献

- 用 **GRPO + 规则奖励**（数学题答案对错）做 RL
- **R1-Zero**：完全没有 SFT，纯 RL 也能涌现长推理
- 蒸馏到小模型：小模型也能有不错的推理能力

### 追问

- **Q：为什么 GRPO 适合 reasoning？**
  A：规则可验证（答案对错） + 多采样有意义 + 不要 critic 省资源。

---

## Q04 · MoE 最新进展

### DeepSeek-V2/V3

- **细粒度专家**：把专家拆得更小，每个 token 激活少量 routed experts，并额外设置 shared experts；具体激活数因版本而异
- **MLA**：极致压缩 KV Cache
- **辅助损失更轻**（auxiliary-loss-free）
- V3 展示了“总参数远大于每 token 激活参数”的大规模稀疏设计

### Mixtral 8x7B

- 8 个专家，每 token 激活 2 个
- 总参 47B，激活 13B
- 早期开源 MoE 标杆

### 混合稀疏架构

前沿模型常把稀疏 MoE 与 attention、线性 attention 或 SSM 层组合：attention 提供精确内容寻址，线性/状态空间层降低长序列成本，MoE 扩大参数容量。判断设计时应拆开总参数、每 token 激活参数、KV/状态大小、路由通信和训练稳定性，不能只用“总参数很大、激活很小”概括。

### 趋势

1. **稀疏度越来越高**（激活比 < 5%）
2. **细粒度专家**（数量增多、单个变小）
3. **共享专家**（捕捉通用知识）
4. **路由器更稳**（noisy top-k、辅助 loss 调整）

---

[⬅ 回到首页](../README.md)
