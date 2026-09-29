# LLM 手撕题集 · LLM Coding Drills

LLM 模型组件的**纯 PyTorch 手撕代码**，跟 [`docs/`](../docs/) 原理章节互为补充。

> **跟 [`algorithms/`](../algorithms/) 的区别**：
> - 这里 = **大模型相关**（Attention / RoPE / LoRA / Beam Search...）
> - `algorithms/` = **通用算法 / 数据结构**

---

## 章节目录

| 章节 | 内容 |
|---|---|
| [01 · Attention](01-attention.md) | Scaled Dot-Product · MHA · Causal Mask · GQA · KV Cache |
| [02 · Position Encoding](02-position-encoding.md) | Sinusoidal · RoPE · ALiBi |
| [03 · Normalization](03-normalization.md) | LayerNorm · RMSNorm · BatchNorm 对比 |
| [04 · Tokenizer](04-tokenizer.md) | 字符级 BPE 训练 · 编码 · 词表构建 |
| [05 · Decoding](05-decoding.md) | Greedy · Top-k · Top-p · Beam Search · 重复惩罚 |
| [06 · PEFT](06-peft.md) | LoRA · QLoRA Linear · Adapter |
| [07 · Loss & RL](07-loss-rl.md) | CE · DPO · PPO · GRPO · Reward Model · 多奖励归一化 |
| [08 · Misc](08-misc.md) | MoE 路由 · 分块 Attention · Tied Embedding · 梯度技巧 |

---

## 使用建议

每题围绕以下 5 部分组织：

1. **一句话目标**：要实现什么、关键点在哪
2. **实现代码**：区分可调用定义、依赖同章组件的代码、需要外部模型的接口示例和局部片段
3. **设计动机**：关键缩放、参数、维度为什么这样选
4. **易错点**：面试官常追问的细节与适用边界
5. **形状变化**：每行后注释 tensor shape

先执行同章的 import 与依赖定义，再运行对应组件。示例中的 `model`、输入张量和优化器通常由调用者提供，不能将任意单个代码块当完整程序。生产 kernel、量化存储和调度系统的简化实现均应按正文边界理解。

## 本地回归验证

在安装 PyTorch 的 Python 环境，从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

测试直接从 Markdown 提取函数/类定义，避免另外维护一套“能通过测试”的复制代码。当前覆盖 LoRA、Norm、Attention/cache、CE/smoothing、GRPO 聚合、BPE、EOS 和 MoE 等重点回归，不代表全仓每段代码都已运行，也不替代 GPU 性能或 GitHub 公式渲染检查。详情见 [tests/README.md](../tests/README.md)。

---

[⬅ 回到首页](../README.md)
