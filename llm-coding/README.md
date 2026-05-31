# 🧠 LLM 手撕题集 · LLM Coding Drills

LLM 模型组件的**纯 PyTorch 手撕代码**，跟 [`docs/`](../docs/) 原理章节互为补充。

> 💡 **跟 [`algorithms/`](../algorithms/) 的区别**：
> - 这里 = **大模型相关**（Attention / RoPE / LoRA / Beam Search...）
> - `algorithms/` = **通用算法 / 数据结构**

---

## 📚 章节目录

| 章节 | 内容 |
|---|---|
| [01 · Attention](01-attention.md) | Scaled Dot-Product · MHA · Causal Mask · GQA · KV Cache |
| [02 · Position Encoding](02-position-encoding.md) | Sinusoidal · RoPE · ALiBi |
| [03 · Normalization](03-normalization.md) | LayerNorm · RMSNorm · BatchNorm 对比 |
| [04 · Tokenizer](04-tokenizer.md) | BPE 训练 · 编码 · 解码 |
| [05 · Decoding](05-decoding.md) | Greedy · Top-k · Top-p · Beam Search · 重复惩罚 |
| [06 · PEFT](06-peft.md) | LoRA · QLoRA Linear · Adapter |
| [07 · Loss & RL](07-loss-rl.md) | Cross-Entropy · DPO Loss · PPO Loss · GRPO |
| [08 · Misc](08-misc.md) | MoE 路由 · Flash Attention 简化版 · Mixture of Depths |

---

## 🎯 使用建议

每题都有 4 部分：

1. **🎯 一句话目标**：要实现什么、关键点在哪
2. **🛠 完整可运行代码**：粘到 jupyter 就能跑
3. **🪤 易错点**：面试官常追问的细节
4. **📊 形状变化**：每行后注释 tensor shape（这是 LLM 手撕的灵魂）

---

[⬅ 回到首页](../README.md)
