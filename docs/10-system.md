# 10 · System 工程系统

## 📑 本章目录

- [Q01 · GPU 显存计算（参数+梯度+优化器+激活）](#q01--gpu-显存计算)
- [Q02 · 服务化（Triton / vLLM / TGI / SGLang）](#q02--服务化)
- [Q03 · P/D 分离（Prefill / Decode）](#q03--pd-分离)
- [Q04 · 推理成本估算](#q04--推理成本估算)
- [Q05 · KV Cache 共享与 Prefix Cache](#q05--kv-cache-共享与-prefix-cache)

---

## Q01 · GPU 显存计算

### 🎯 训练显存估算

每参数（FP16 + FP32 主权重 + AdamW）需要：

| 项 | bytes/param |
|---|---|
| 参数（FP16） | 2 |
| 梯度（FP16） | 2 |
| 主权重（FP32） | 4 |
| Adam 一阶矩（FP32） | 4 |
| Adam 二阶矩（FP32） | 4 |
| **合计** | **16** |

7B 模型 → ~112 GB（**还不算 activation**）。

### 📖 Activation 显存

约 $O(B \times L \times d \times N)$（B=batch, L=seq len, N=layers）。
长序列下经常超过参数显存 → 用 **Activation Checkpointing**（重新计算换显存）。

### 📖 推理显存

| 项 | bytes/param |
|---|---|
| 参数（FP16） | 2 |
| KV Cache（动态） | $2 \times \text{batch} \times \text{seq} \times n_h \times d_h \times L \times 2$ |

7B 模型推理：~14 GB 参数 + 几 GB KV Cache。

### 🪤 追问

- **Q：怎么把 70B 模型塞进 24G 卡推理？**
  A：4-bit 量化（70B × 0.5B = 35GB → 还放不下单卡）→ 需要多卡 TP，或卸载到 CPU。

---

## Q02 · 服务化

| 框架 | 特点 |
|---|---|
| **vLLM** | PagedAttention + Continuous Batching，**吞吐量之王** |
| **TGI**（HuggingFace） | 工业级，K8s 友好 |
| **SGLang** | 结构化输出 + RadixAttention，复杂 prompt 高效 |
| **Triton Inference Server** | NVIDIA 官方，多模型多框架 |
| **llama.cpp** | CPU/Apple Silicon 推理 |
| **MLC-LLM** | 跨平台（含手机） |

### 🪤 追问

- **Q：为什么 vLLM 吞吐高？**
  A：**PagedAttention**（显存利用率近 100%）+ **Continuous Batching**（请求即来即走）+ 高效 CUDA Kernel。

---

## Q03 · P/D 分离（Prefill / Decode）

### 🎯 核心观察

LLM 推理两个阶段特性完全不同：

| 阶段 | 计算特性 | 瓶颈 |
|---|---|---|
| **Prefill** | 一次处理整个 prompt（并行） | **算力**（compute-bound） |
| **Decode** | 一次出一个 token | **显存带宽**（memory-bound） |

### 📖 P/D 分离架构

把 prefill 和 decode 部署在**不同的 GPU 上**：

- Prefill 用算力强的 GPU（H100）
- Decode 用显存带宽强的 GPU
- 中间通过 **KV Cache 迁移**衔接

代表系统：DistServe、Mooncake、SGLang。

### 🪤 追问

- **Q：P/D 分离的代价？**
  A：KV Cache 迁移有网络开销 → 需要 NVLink / RDMA 等高速互联。

---

## Q04 · 推理成本估算

### 📖 经验公式（FLOPs）

每生成一个 token 的 FLOPs ≈ **2 × 参数量**（不算 attention 部分）

例：7B 模型生成 1 个 token ≈ 14 GFLOPs。

### 📖 经济成本

按 cloud GPU 大致价格估算（2026 行情，仅作量级参考）：

- A100 80G：~$2/小时
- H100：~$4/小时
- 一张 H100 BF16 跑 LLaMA-7B 大约 ~3000 tokens/s 单流

→ 每 1M output token 成本 ~$0.3-1（与 batch 大小、序列长度、是否量化关系大）。

---

## Q05 · KV Cache 共享与 Prefix Cache

### 🎯 场景

很多应用有**共同的 system prompt** 或**重复的 prefix**（多用户问同一个 long context）。

### 📖 优化

- **Prefix Cache**：把常见 prefix 的 KV Cache 持久化，新请求直接复用
- **RadixAttention**（SGLang）：用基数树管理 prefix，自动复用

### 🪤 追问

- **Q：能省多少？**
  A：在 system prompt 较长的场景（如 RAG），prefill 显著加速；同 system prompt 多用户场景吞吐 2-5×。

---

[⬅ 回到首页](../README.md)
