# 10 · System 工程系统

## 本章目录

- [Q01 · GPU 显存计算（参数+梯度+优化器+激活）](#q01--gpu-显存计算)
- [Q02 · 服务化（Triton / vLLM / TGI / SGLang）](#q02--服务化)
- [Q03 · P/D 分离（Prefill / Decode）](#q03--pd-分离)
- [Q04 · 推理成本估算](#q04--推理成本估算)
- [Q05 · KV Cache 共享与 Prefix Cache](#q05--kv-cache-共享与-prefix-cache)
- [Q06 · 从算术强度到延迟与吞吐](#q06--从算术强度到延迟与吞吐)

---

## Q01 · GPU 显存计算

### 训练显存估算

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

16 bytes/param 是一种常见混合精度 AdamW 口径，不是常数：有的实现梯度为 FP32、有的 BF16 不保留独立 master copy，8-bit optimizer 又会压缩状态。面试估算应先声明 dtype 和是否分片，再分别计算 persistent states、activation、临时 buffer 与碎片。

### Activation 显存

记 $B$ 为 batch、$S$ 为序列长度、$d$ 为 hidden size、$N$ 为层数。逐 token 激活通常包含 $O(BSdN)$ 项，其常数还取决于 FFN 宽度、激活函数、保存策略和 dtype；若物化并保存多头注意力矩阵，还会出现 $O(BH_qS^2N)$ 项。不能在普通 attention 与 FlashAttention 下使用同一个无条件估算。

Activation Checkpointing 用反向重算减少保存的中间量，但 checkpoint 边界输入、当前重算工作集和通信 buffer 仍要占空间。FlashAttention 避免完整注意力矩阵存储，不代表所有训练激活都消失。

### 推理显存

| 项 | 总字节数口径 |
|---|---|
| 参数（FP16/BF16） | $2P$，$P$ 为参数量 |
| KV Cache（各请求长度相同） | $2BSH_{kv}d_hNb_{kv}$，$b_{kv}$ 为每个缓存元素字节数 |
| 其他 | 量化 scale/zero-point、分页元数据、临时 workspace、采样和运行时开销 |

KV 的第一个系数 2 表示 K/V 两份；$H_{kv}$ 是 KV 头数，不是 query 头数。变长请求将 $BS$ 换成各请求缓存长度之和；分页还应按已分配块数计费。该式用于普通 MHA/GQA/MQA，MLA 要按实际潜在向量及位置 key 的缓存形状重算。

### 一组统一单位的手算

设参数量 7B，BF16 权重；32 层、32 个 query 头、8 个 KV 头、head dim=128；4 个请求各缓存 8192 个 token，KV 为 BF16：

```math
M_{\mathrm{weight}}=7\times10^9\times2
=14\ \mathrm{GB}\approx13.04\ \mathrm{GiB}
```

```math
M_{\mathrm{KV}}
=2\times4\times8192\times8\times128\times32\times2
=4\ \mathrm{GiB}
```

权重加 KV 约为 17.04 GiB，仍不是峰值显存。若 KV 头数改为 32（MHA），其余不变，KV 变为 16 GiB；不能误用 query 头数把 GQA 缓存算大四倍。这里 $1\ \mathrm{GB}=10^9$ bytes，$1\ \mathrm{GiB}=2^{30}$ bytes。

以本章 16 bytes/param 的训练状态口径为例，7B 的状态合计 112 GB；理想 8 路完整分片约为每卡 14 GB **常驻模型状态**。当前模块 all-gather、prefetch、激活和碎片会叠加形成更高峰值，不能据此保证某容量 GPU 一定能训练。分片生命周期见 [Training Q06](02-training.md#q06--分布式训练)。

### 追问

- **Q：怎么把 70B 模型塞进 24G 卡推理？**
  A：理想 4-bit 权重下限约为 $70\text{B}\times0.5$ byte $\approx35$ GB，还没算 scale/zero-point、KV Cache、workspace 和运行时开销，因此单张 24 GB 卡仍放不下完整模型。需要更激进量化、多卡切分或 CPU/offload，并接受相应速度和精度代价。

---

## Q02 · 服务化

| 框架 | 特点 |
|---|---|
| **vLLM** | PagedAttention + Continuous Batching，通用高吞吐方案 |
| **TGI**（HuggingFace） | 工业级，K8s 友好 |
| **SGLang** | 结构化输出 + RadixAttention，复杂 prompt 高效 |
| **Triton Inference Server** | NVIDIA 官方，多模型多框架 |
| **llama.cpp** | CPU/Apple Silicon 推理 |
| **MLC-LLM** | 跨平台（含手机） |

### 追问

- **Q：为什么 vLLM 吞吐高？**
  A：Paged KV 管理减少预留和碎片，Continuous Batching 提高调度利用率，再配合高效 kernel。实际吞吐取决于 workload、模型和硬件，不保证接近某个固定显存利用率。

---

## Q03 · P/D 分离（Prefill / Decode）

### 核心观察

LLM 推理两个阶段特性完全不同：

| 阶段 | 计算特性 | 瓶颈 |
|---|---|---|
| **Prefill** | 一次处理多个 prompt token，矩阵乘复用权重 | 较容易 compute-bound；短输入、小 batch 也可能受其他开销限制 |
| **Decode** | 每条序列通常一次出一个 token | 小 batch 常受权重/KV 带宽限制；长上下文或大 batch 也可能转向计算瓶颈 |

### P/D 分离架构

把 prefill 和 decode 部署在**不同的 GPU 池**：

- Prefill 池按矩阵计算吞吐和 prompt 长度配置较大 batch
- Decode 池按显存容量/带宽、并发数和 token latency 配置
- 中间通过 **KV Cache 迁移**衔接

代表系统：DistServe、Mooncake、SGLang。

### 追问

- **Q：P/D 分离的代价？**
  A：KV Cache 迁移有网络开销 → 需要 NVLink / RDMA 等高速互联。

---

## Q04 · 推理成本估算

### 经验公式（FLOPs）

对稠密 Transformer，在 batch 很小且忽略 attention、embedding、采样等项时，每个 decode token 的主干矩阵乘 FLOPs 常粗估为 **2 × 激活参数量**；MoE 应使用每 token 激活参数而非总参数。

例：7B 模型生成 1 个 token ≈ 14 GFLOPs。

### 为什么 FLOPs 不能直接换算成 tokens/s？

Decode 常受权重和 KV Cache 的显存带宽限制，单流很难达到 GPU 峰值 FLOPs；增加 batch 可复用权重读取、提高算术强度，但会增加每个请求延迟和 KV 显存。成本估算应以目标 workload 实测：

```text
每百万 token 成本
= GPU 每小时成本 × GPU 数 × 运行小时 / 有效生成 token × 1,000,000
```

必须分别报告 input/output token、TTFT、TPOT、并发、上下文长度分布、成功率和利用率。背某张卡固定 tokens/s 或固定美元数没有可迁移性。

---

## Q05 · KV Cache 共享与 Prefix Cache

### 场景

很多应用有**共同的 system prompt** 或**重复的 prefix**（多用户问同一个 long context）。

### 优化

- **Prefix Cache**：把常见 prefix 的 KV Cache 持久化，新请求直接复用
- **RadixAttention**（SGLang）：用基数树管理 prefix，自动复用

Prefix cache 通常要求 token 级前缀完全一致；哪怕空格、模板版本或 special token 不同都会 miss。复用的是模型某一版本、某组位置编码和推理配置下的 KV，换权重或影响 K/V 的 adapter 后通常必须失效。多租户系统还要把 cache key 加入权限域，避免通过命中时延或错误复用泄露其他用户的私有前缀。

### 追问

- **Q：能省多少？**
  A：上限取决于可复用 prefix 占输入的比例和命中率。它主要省 prefill 计算，不能减少后续 decode token 的模型计算；应报告 cache hit rate、复用 token 数、TTFT 变化和 cache 占用，而不是给固定倍数。

---

## Q06 · 从算术强度到延迟与吞吐

### 为什么同样的 FLOPs，会有不同的速度？

令计算量为 $F$ FLOPs、实际搬运数据量为 $M$ bytes、有效算力为 $C$ FLOPs/s、有效带宽为 $BW$ bytes/s。忽略其他开销，执行时间受两者较慢的一项限制：

```math
t\gtrsim\max\!\left(\frac{F}{C},\frac{M}{BW}\right),
\qquad I=\frac{F}{M}
```

$I$ 是算术强度。它低于设备的算力/带宽比值时，更可能受访存限制；高于该值也不代表必然满算力，因为还存在 kernel launch、通信、布局和并行度等限制。

只看稠密权重主导的 decode 矩阵乘，忽略 KV 和其他项：$P$ 个参数为 $B$ 个 token 各计算约 $2P$ FLOPs，权重每步读一次、每参数占 $b$ bytes，则 $I\approx2B/b$。BF16 的 $b=2$，batch=1 时约 1 FLOP/byte，batch=16 时约 16。这解释了 batch 如何摊薄权重读取，但实际 KV 读取随上下文和并发增加，收益不会无限线性增长。

### 为什么吞吐提高不代表用户更快拿到答案？

- **TTFT**：从请求到首 token，通常包含排队、prefill 与调度等待；测量端点必须说明。
- **TPOT / ITL**：后续 token 的平均时间或逐 token 间隔分布；平均值可能掩盖卡顿。
- **端到端延迟**：近似 TTFT 加后续 token 时间，但实际间隔可能不均匀。
- **吞吐**：单位时间处理的有效 token 或成功请求；需说明输入/输出、并发与长度分布。

增加 batch 可提高总 tokens/s，却可能让单请求等待更久。Chunked prefill 把长 prompt 分段，与 decode 交错，能减少长 prefill 对已有请求的阻塞，但会改变新请求 TTFT、调度和 kernel 效率。P/D 分离还应把 KV 传输、两端排队与资源失衡算入收益，不能只比较两个独立 kernel 的速度。

### 一个有用的基准测试应报告什么？

固定模型/量化、硬件拓扑、软件版本与采样设置；给出输入/输出长度分布、并发或到达率、prefix 命中条件、预热方式。联合报告成功率、有效 output tokens/s、TTFT 和 ITL 的 p50/p95/p99。

闭环测试中客户端等完成后才发下一次请求，会在系统变慢时自动降低到达率；固定到达率的开环测试更容易暴露排队失控。两者回答不同问题。讨论容量时应报告“满足延迟约束的成功吞吐”，不能把超时或丢弃请求从统计中静默删除。算术强度背景见 [Berkeley Lab Roofline 资料](https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/ppan-roofline-publications/)。

---

[⬅ 回到首页](../README.md)
