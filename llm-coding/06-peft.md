# 06 · PEFT 手撕

参数高效微调：LoRA / QLoRA / Adapter。

## 本章目录

- [Q01 · LoRA Linear](#q01--lora-linear)
- [Q02 · 把 LoRA 注入已有模型](#q02--把-lora-注入已有模型)
- [Q03 · Merge LoRA 权重回原模型](#q03--merge-lora-权重回原模型)
- [Q04 · QLoRA 的 4-bit 量化思路](#q04--qlora-的-4-bit-量化思路)
- [Q05 · Adapter Layer](#q05--adapter-layer)

---

## Q01 · LoRA Linear

### 目标

把原 Linear $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 改成：

$$
y = Wx + \frac{\alpha}{r} BA x
$$

其中 $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$，$r$ 远小于 $d$。

### 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16, dropout: float = 0.0,
                 base_linear: nn.Linear = None):
        super().__init__()
        # 原 Linear（冻结）
        if base_linear is not None:
            self.base = base_linear
        else:
            self.base = nn.Linear(in_features, out_features, bias=False)
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA 旁路
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 常见初始化：A 用 Kaiming uniform，B 用 0
        # 这样初始时 BA = 0，不影响原模型输出
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """x: [..., in_features]"""
        base_out = self.base(x)                                 # 原模型主路
        lora_out = self.lora_dropout(x) @ self.lora_A.T         # [..., r]
        lora_out = lora_out @ self.lora_B.T                     # [..., out_features]
        return base_out + lora_out * self.scaling
```

### 易错点（必考！）

1. **为什么一个因子置零、另一个随机**：两者都为零时，$\nabla_A L\propto B^T$、$\nabla_B L\propto A^T$ 都为零，训练无法启动。A 随机、B 为零可保证初始 $BA=0$，第一步先更新 B；反过来在数学上也能启动（第一步先更新 A），并非“绝对不能反”，只是常见实现采用前者
2. **`scaling = α / r`**：把 rank 与分支强度分开调节，但不保证任意初始化下更新范数完全与 r 无关；较大 rank 时也有 $\alpha/\sqrt r$ 的稳定缩放变体
3. **冻结原 Linear**：`requires_grad = False` 必须做，否则没省到参数
4. **`@ self.lora_A.T` 不是 `@ self.lora_A`**：注意 weight 的 shape 约定（out, in），跟 nn.Linear 一致

LoRA 假设低秩的是任务更新 $\Delta W$，不是基座 $W$。$r$ 控制更新子空间容量：太小欠拟合，太大增加显存与过拟合风险；应结合覆盖层数、任务跨度和验证集选择，而不是只背 8/16/64。

---

## Q02 · 把 LoRA 注入已有模型

### 目标

给一个已经定义好的模型（比如 GPT），把所有 `nn.Linear` 替换成 LoRALinear，但保留原权重。

### 代码

```python
def inject_lora(model: nn.Module, target_modules=('q_proj', 'v_proj'),
                r=8, alpha=16):
    """
    把 model 中名字匹配 target_modules 的 Linear 替换成 LoRALinear。
    """
    for name, module in model.named_modules():
        # 只处理叶子模块的 Linear
        if not isinstance(module, nn.Linear):
            continue
        # 名字最后一段在 target_modules 里
        last_name = name.split('.')[-1]
        if last_name not in target_modules:
            continue

        # 构造 LoRALinear，复用原 Linear 的权重
        new_module = LoRALinear(
            module.in_features,
            module.out_features,
            r=r, alpha=alpha,
            base_linear=module    # 直接传入原 Linear，权重不变
        )

        # 找到父模块并替换
        parent_name = '.'.join(name.split('.')[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, last_name, new_module)

    return model


# === 用法 ===
# model = AutoModelForCausalLM.from_pretrained("...")
# model = inject_lora(model, target_modules=('q_proj', 'v_proj'), r=16, alpha=32)
# 现在只有 lora_A / lora_B 是 trainable
```

### 易错点

- **`get_submodule`**：处理嵌套路径（如 `transformer.h.0.attn.q_proj`）的安全访问
- **target_modules 选哪些**：`q_proj, v_proj` 是常见轻量起点；覆盖 `k_proj, o_proj` 和 FFN 会提高容量，也增加可训练参数。没有跨任务固定最优集合
- **norm / embedding 要单独决策**：LoRA 主要包装矩阵权重；norm 或 embedding 可冻结，也可通过 `modules_to_save` 等方式直接训练，取决于词表变化和任务跨度

---

## Q03 · Merge LoRA 权重回原模型

### 目标

部署时不想多两个矩阵乘法，把 $W' = W + \frac{\alpha}{r} BA$ 合并回去。

### 代码

```python
def merge_lora(model: nn.Module):
    """把 LoRALinear 合并回 nn.Linear"""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        # 计算合并后的权重：W + (α/r) * B @ A
        merged_weight = module.base.weight.data + module.scaling * (module.lora_B @ module.lora_A)

        # 创建新 Linear 替换
        new_linear = nn.Linear(
            module.base.in_features,
            module.base.out_features,
            bias=(module.base.bias is not None)
        )
        new_linear.weight.data = merged_weight
        if module.base.bias is not None:
            new_linear.bias.data = module.base.bias.data.clone()

        # 替换
        parent_name = '.'.join(name.split('.')[:-1])
        last_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, last_name, new_linear)

    return model
```

### 易错点

- **合并后结构零额外矩阵乘**：数学上把两支合成一个权重；有限精度下若 merge dtype 不同可能有微小数值差异
- **只能 merge 一次**：merge 之后想再换一份 LoRA，要么从头加载原模型，要么记录 merge 前的 base weight
- **训练完保存**：通常只保存 `lora_A` / `lora_B`（几 MB），加载时再 merge 或注入

---

## Q04 · QLoRA 的 4-bit 量化思路

### 完整实现太复杂（依赖 bitsandbytes 的 cuda kernel），但核心思想可以手撕

```python
# 4-bit 量化的关键三步：
#   1) NF4 量化原 weight 到 4-bit（节省显存）
#   2) 反量化时 dequant，但只在前向用
#   3) LoRA 旁路依然用 fp16/bf16 训练

class QLoRALinear(nn.Module):
    """简化教学版（不用 bitsandbytes，仅展示思想）"""
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        # 量化后的权重 + 量化参数（实际生产用 NF4 + double quant）
        self.weight_int4 = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.int8),
            requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.ones(out_features, 1),  # 简化：每行一个 scale
            requires_grad=False
        )
        # LoRA 旁路（保持 fp16/bf16）
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Dequantize（只在前向，反向不算梯度）
        weight = self.weight_int4.float() * self.scale
        base_out = F.linear(x, weight)
        # LoRA 旁路
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


# 量化函数（教学版的对称量化）
def quantize_int4(weight: torch.Tensor):
    """对每行做对称量化到 [-8, 7]"""
    abs_max = weight.abs().max(dim=-1, keepdim=True).values
    scale = (abs_max / 7.0).clamp_min(torch.finfo(weight.dtype).eps)
    quantized = (weight / scale).round().clamp(-8, 7).to(torch.int8)
    return quantized, scale
```

### QLoRA 真实工程要点

- **NF4（NormalFloat 4-bit）**：基于权重正态分布的非均匀量化，比 INT4 损失更小
- **Double Quantization**：把 scale 也量化（节省 0.4 bits/param）
- **Paged Optimizer**：CPU 内存里放 optimizer states，避免 OOM
- **完整实现看 bitsandbytes 库**

教学代码用 `int8` 张量保存数值范围为 INT4 的整数，**没有真正把两个 4-bit 值打包进一个 byte**，因此不能展示真实显存节省；生产 kernel 还要处理 block/group scale、packing 和 fused dequant-GEMM。NF4 的码点按近似正态权重的分位数非均匀分布，目标是把有限码点更多放在高概率区域，而不是简单换一个 dtype 名称。

---

## Q05 · Adapter Layer

### 目标

在每个 Transformer block 后插入小的 bottleneck：

```
输入 → 降维 (d → r) → ReLU → 升维 (r → d) → 残差连接
```

### 代码

```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.act = nn.ReLU()

        # 关键：up projection 初始化为 0，让初始 adapter 等于恒等
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))    # 残差


# 用法：在 Transformer block 输出后加
class BlockWithAdapter(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter
        # 冻结原 block
        for p in self.block.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.adapter(self.block(x))
```

### LoRA vs Adapter

| 维度 | LoRA | Adapter |
|---|---|---|
| 插入位置 | 旁路（并联） | 串行（在 block 后） |
| 推理开销 | merge 后无额外矩阵乘 | 多两层 Linear（不可消） |
| 参数量 | 极少（rank） | 较少（bottleneck） |
| 现代使用 | 广泛 | 仍用于需要模块化串行适配等场景 |

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
