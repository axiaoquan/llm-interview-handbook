# LLM Interview Handbook · 大模型 + 算法面试手册

> 一个仓库覆盖 **AI 算法工程师面试** 两条核心战线：
> **🤖 大模型知识** + **💻 代码手撕**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
![Made for Interviews](https://img.shields.io/badge/Made%20for-Interviews-blue)

---

## 📚 这是什么？

为 **AI / 算法工程师** 准备的结构化面试手册。两个独立的知识体系：

### 🤖 [大模型知识 (`docs/`)](docs/)
LLM 八股 —— Transformer / 训练 / 微调 / RAG / 推理优化 / 多模态 / 前沿方向。
每题按统一模板：**一句话答案 / 详细展开 / 追问 / 参考 / 相关问题**。

### 💻 [代码手撕 (`algorithms/`)](algorithms/)
通用算法 / 数据结构刷题笔记 —— 数组 / 链表 / 哈希 / 字符串 / 栈队列 / 树 / 回溯 / 动态规划 / Hot100。
每题：**题目链接 + 核心思路 + 代码 + 时空复杂度 + 易错点**。

### 🧠 [LLM 手撕题 (`llm-coding/`)](llm-coding/)
跟 LLM 强绑定的代码题 —— 手写 Self-Attention / RoPE / LoRA / Beam Search / KV Cache。
跟 `algorithms/` 区分：这里写的是**模型组件**，那里写的是**通用算法**。

---

## 🗂 大模型知识 · 全局目录

| 章节 | 内容 | 进度 |
|---|---|---|
| [00 · Foundations](docs/00-foundations/) | 概率统计 / 机器学习基础 | 🚧 |
| [01 · Architecture](docs/01-architecture/) | Transformer / Attention / Position / Norm | 🚧 |
| [02 · Training](docs/02-training/) | 损失 / 优化器 / 分布式 / 混合精度 | 🚧 |
| [03 · Fine-tuning](docs/03-fine-tuning/) | LoRA / QLoRA / PEFT / 指令微调 | 🚧 |
| [04 · Alignment](docs/04-alignment/) | RLHF / PPO / DPO / GRPO | 🚧 |
| [05 · Inference](docs/05-inference/) | KV Cache / Flash Attention / 量化 / 解码 | 🚧 |
| [06 · RAG](docs/06-rag/) | 索引 / 检索 / 重排 / 高级 RAG | 🚧 |
| [07 · Agent](docs/07-agent/) | ReAct / Tool Use / Multi-Agent | ⏳ |
| [08 · Multimodal](docs/08-multimodal/) | CLIP / BLIP / LLaVA / Diffusion | ⏳ |
| [09 · Evaluation](docs/09-evaluation/) | MMLU / Hallucination / LLM-as-Judge | ⏳ |
| [10 · System](docs/10-system/) | 显存计算 / 服务化 / 部署 | ⏳ |
| [99 · Frontier](docs/99-frontier/) | DLM / Mamba / o1 推理时扩展 | ⏳ |

## 🗂 代码手撕 · 全局目录

| 章节 | 内容 | 进度 |
|---|---|---|
| [01 · Array 数组](algorithms/01-array/) | 二分 / 双指针 / 滑动窗口 / 前缀和 | 🚧 |
| [02 · LinkedList 链表](algorithms/02-linked-list/) | 反转 / 双指针 / 环形 / 相交 | 🚧 |
| [03 · HashTable 哈希表](algorithms/03-hash-table/) | 数组哈希 / set / map | 🚧 |
| [04 · String 字符串](algorithms/04-string/) | 反转 / KMP / 重复子串 | 🚧 |
| [05 · StackQueue 栈与队列](algorithms/05-stack-queue/) | 互相实现 / 单调队列 / 优先队列 | 🚧 |
| [06 · BinaryTree 二叉树](algorithms/06-binary-tree/) | 三种遍历 / 层序 / 路径 | 🚧 |
| [07 · Backtrack 回溯](algorithms/07-backtrack/) | 组合 / 切割 / 子集 / 排列 / 棋盘 | ⏳ |
| [08 · MonotonicStack 单调栈](algorithms/08-monotonic-stack/) | 每日温度 / 接雨水 / 柱状图 | 🚧 |
| [09 · DP 动态规划](algorithms/09-dp/) | 基础 / 背包 / 子序列 | 🚧 |
| [10 · Hot100](algorithms/10-hot100/) | LeetCode Hot100 重点题 | 🚧 |
| [99 · Tips](algorithms/99-tips/) | Python 易错点 / 输入输出技巧 | 🚧 |

> 🚧 = 正在补充内容　⏳ = 暂未开工

---

## 🚀 怎么用

### 准备 LLM 面试
按 `docs/` 章节顺序读，重点看**🪤 追问**部分。

### 刷算法 / 系统手撕
按 `algorithms/` 章节顺序读，每章会标记**重点题**。

### 工作里查东西
直接 `Ctrl+F` / GitHub 搜索关键词。

---

## 📝 单题模板

每篇文档都按这个结构组织（[`docs/_TEMPLATE.md`](docs/_TEMPLATE.md)）：

```markdown
# QXX · 标题
> 难度：⭐⭐⭐ · 分类：xxx · 常见公司：xxx

## 🎯 一句话答案
（30 秒）

## 📖 详细展开
（5 分钟）

## 🪤 面试常见追问
- Q：...
- A：...

## 📚 参考

## 🔗 相关问题
```

算法题略有不同（[`algorithms/_TEMPLATE.md`](algorithms/_TEMPLATE.md)）：

```markdown
# QXX · 题目名

> [LeetCode 链接](https://leetcode.cn/problems/...) · 难度：⭐⭐⭐

## 🎯 思路

## 🛠 代码

## 📊 复杂度

## 🪤 易错点

## 🔗 同类题
```

---

## 🤝 贡献

欢迎 Issue / PR，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 License

[MIT](LICENSE)

---

> 维护者：[@axiaoquan](https://github.com/axiaoquan) · 个人主页：<https://axiaoquan.github.io/>
