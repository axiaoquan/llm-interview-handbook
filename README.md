# LLM Interview Handbook · 大模型 + 算法面试手册

> 一个仓库覆盖 **AI 算法工程师面试** 两条核心战线：
> **🤖 大模型知识** + **💻 代码手撕**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
![Made for Interviews](https://img.shields.io/badge/Made%20for-Interviews-blue)

---

## 📚 这是什么？

为 **AI / 算法工程师** 准备的结构化面试手册。三个独立的知识体系：

- 🤖 **[大模型知识 (`docs/`)](docs/)** — Transformer / 训练 / 微调 / RAG / 推理优化 / 多模态 / 前沿
- 💻 **[算法手撕 (`algorithms/`)](algorithms/)** — 数组 / 链表 / 树 / DP / Hot100
- 🧠 **[LLM 手撕 (`llm-coding/`)](llm-coding/)** — Attention / RoPE / LoRA / Beam Search

**每章一个文件**，顶部带本章目录，顺着读完即可，不用频繁跳转。

---

## 🤖 大模型知识

| 章节 | 内容 |
|---|---|
| [00 · Foundations](docs/00-foundations.md) | 概率统计 / 机器学习基础 |
| [01 · Architecture](docs/01-architecture.md) | Transformer / Attention / Position / Norm |
| [02 · Training](docs/02-training.md) | 损失 / 优化器 / 分布式 / 混合精度 |
| [03 · Fine-tuning](docs/03-fine-tuning.md) | LoRA / QLoRA / PEFT / 指令微调 |
| [04 · Alignment](docs/04-alignment.md) | RLHF / PPO / DPO / GRPO |
| [05 · Inference](docs/05-inference.md) | KV Cache / Flash Attention / 量化 / 解码 |
| [06 · RAG](docs/06-rag.md) | 索引 / 检索 / 重排 / 高级 RAG |
| [07 · Agent](docs/07-agent.md) | ReAct / Tool Use / Multi-Agent |
| [08 · Multimodal](docs/08-multimodal.md) | CLIP / BLIP / LLaVA / Diffusion |
| [09 · Evaluation](docs/09-evaluation.md) | MMLU / Hallucination / LLM-as-Judge |
| [10 · System](docs/10-system.md) | 显存计算 / 服务化 / 部署 |
| [99 · Frontier](docs/99-frontier.md) | DLM / Mamba / o1 推理时扩展 |

## 💻 算法手撕

| 章节 | 内容 |
|---|---|
| [01 · Array](algorithms/01-array.md) | 二分 / 双指针 / 滑动窗口 / 前缀和 |
| [02 · LinkedList](algorithms/02-linked-list.md) | 反转 / 双指针 / 环形 / 相交 |
| [03 · HashTable](algorithms/03-hash-table.md) | 数组哈希 / set / map |
| [04 · String](algorithms/04-string.md) | 反转 / KMP / 重复子串 |
| [05 · StackQueue](algorithms/05-stack-queue.md) | 互相实现 / 单调队列 / 优先队列 |
| [06 · BinaryTree](algorithms/06-binary-tree.md) | 三种遍历 / 层序 / 路径 |
| [07 · Backtrack](algorithms/07-backtrack.md) | 组合 / 切割 / 子集 / 排列 |
| [08 · MonotonicStack](algorithms/08-monotonic-stack.md) | 每日温度 / 接雨水 / 柱状图 |
| [09 · DP](algorithms/09-dp.md) | 基础 / 背包 / 子序列 |
| [10 · Hot100](algorithms/10-hot100.md) | LeetCode Hot100 重点题 |
| [99 · Tips](algorithms/99-tips.md) | Python 易错点 / IO 模板 |

## 🧠 LLM 手撕

| 章节 | 内容 |
|---|---|
| [01 · Attention](llm-coding/01-attention.md) | Scaled Dot-Product · MHA · Causal · GQA · KV Cache |
| [02 · Position Encoding](llm-coding/02-position-encoding.md) | Sinusoidal · RoPE · ALiBi |
| [03 · Normalization](llm-coding/03-normalization.md) | LayerNorm · RMSNorm · Pre/Post-Norm |
| [04 · Tokenizer](llm-coding/04-tokenizer.md) | BPE 训练 / 编码 / 词频 |
| [05 · Decoding](llm-coding/05-decoding.md) | Greedy · Top-k · Top-p · Beam Search · 重复惩罚 |
| [06 · PEFT](llm-coding/06-peft.md) | LoRA · QLoRA · Adapter |
| [07 · Loss & RL](llm-coding/07-loss-rl.md) | CE · Label Smooth · DPO · PPO · GRPO |
| [08 · Misc](llm-coding/08-misc.md) | MoE · Flash Attention · Tied Embedding · 梯度技巧 |

---

## 📝 单题模板

每题按统一结构组织：

```markdown
## QXX · 题目
> 难度 / 公司 / 标签

### 🎯 一句话答案 / 思路
### 📖 详细展开 / 代码
### 🪤 追问 / 易错点
### 📚 参考
```

模板：[`docs/_TEMPLATE.md`](docs/_TEMPLATE.md) · [`algorithms/_TEMPLATE.md`](algorithms/_TEMPLATE.md)

---

## 🤝 贡献

欢迎 Issue / PR，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 License

[MIT](LICENSE)

---

> 维护者：[@axiaoquan](https://github.com/axiaoquan) · 个人主页：<https://axiaoquan.github.io/>
