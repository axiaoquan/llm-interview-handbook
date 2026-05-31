# 04 · Alignment 对齐

让模型的输出对齐人类偏好。RLHF 及其后继方法。

## 📑 本章目录

- [Q01 · RLHF 三阶段流程](#q01--rlhf-三阶段流程)
- [Q02 · PPO 关键概念](#q02--ppo-关键概念)
- [Q03 · DPO 直接偏好优化](#q03--dpo-直接偏好优化)
- [Q04 · GRPO（DeepSeek-R1）](#q04--grpo)
- [Q05 · PPO vs DPO vs GRPO 对比](#q05--三大对齐算法对比)
- [Q06 · Reward Hacking](#q06--reward-hacking)
- [Q07 · 其他对齐方法（REINFORCE++ / DAPO / KTO）](#q07--其他对齐方法)

---

## Q01 · RLHF 三阶段流程

### 🎯 核心目标

不是让模型"更会续写"，而是让它在多个都"说得通"的回答里，**更偏向人类真正喜欢的那种**。

### 📖 三阶段

```
阶段 1: SFT          → 阶段 2: RM 训练       → 阶段 3: PPO 强化学习
预训练 + 标注数据    用排序数据训打分模型    用 RM 当老师优化策略
```

1. **SFT**（Supervised Fine-Tuning）
   预训练模型 + 标注数据 → 学会基本对话格式
   引入明确的输入-输出对（问答、分类标签）

2. **RM**（Reward Model）训练
   让标注员对**多个回答做排序** $y_w > y_l$
   训练奖励模型 $r_\phi(x, y)$ 给回答打分

3. **PPO**（Proximal Policy Optimization）
   用 RM 当老师给回答打分，继续优化模型策略

---

## Q02 · PPO 关键概念

### 🎯 一句话

**让策略变好，但每次不要改太猛**。

### 📖 PPO 的四个组件

| 组件 | 作用 | 描述 |
|---|---|---|
| **Actor**（策略模型） | 生成回答 | 正在训练的 LLM |
| **Critic**（价值模型） | 评估状态价值 $V(s)$ | 估计未来回报 |
| **Reward Model** | 评分 | 评估单条回答质量 |
| **Reference Model** | 防止偏离 | 冻结的 SFT 模型 |

→ **共 4 个模型，显存爆炸**（典型瓶颈）。

### 📖 PPO 目标函数

$$
L^{PPO} = \mathbb{E}\!\left[\min\!\left(\frac{\pi_\theta}{\pi_{\theta_{old}}} A_t,\ \mathrm{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right) A_t\right)\right]
$$

- $\frac{\pi_\theta}{\pi_{\theta_{old}}}$：重要性采样比
- **clip** 把比值限制在 $[1-\epsilon, 1+\epsilon]$（典型 $\epsilon=0.2$）→ **防止策略一步改太多**
- $A_t$：advantage（GAE 估计）

### 📖 总奖励 = RM 分数 - KL 惩罚

$$
R(x, y) = r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})
$$

KL 惩罚防止策略偏离参考模型太远 → 避免 reward hacking。

### 🪤 面试常见追问

- **Q：PPO 为什么稳定？**
  A：clip 机制 + 重要性采样比 → 限制每步更新幅度，避免训练崩。

- **Q：为什么需要 reference model？**
  A：限制策略不要离 SFT 太远，避免学出"骗 reward"的怪东西（reward hacking）。

---

## Q03 · DPO 直接偏好优化

### 🎯 核心思想

**跳过奖励模型的训练，直接从偏好数据优化策略**。

### 📖 数据形式

不需要标奖励分数，只需要**偏好对**：
$$
D = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N
$$

其中 $y_w$ 是 chosen（人类喜欢的），$y_l$ 是 rejected（人类不喜欢的）。

DPO 一般在**已经做过 SFT 的模型**基础上训练。

### 📖 数学推导

从 RLHF 的最优策略出发：
$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)
$$

反解奖励函数：
$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 偏好模型 $P(y_w > y_l) = \sigma(r(x, y_w) - r(x, y_l))$，
得到 DPO 损失：

$$
L_{DPO} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]
$$

### 📖 关键超参 $\beta$

- **$\beta$ 大**：chosen 和 rejected 的差异被放大，优化更激进
- **$\beta$ 小**：更新更保守，更贴近 reference

典型 $\beta = 0.1$。

### 🪤 为什么 DPO 不需要 reward model 和 PPO？

因为 reward 已经被吸收到 policy 的对数比中：
- 传统 RLHF：先学 $r(x,y)$ → 再用 PPO 优化 $\pi$
- DPO：直接把 $r(x,y)$ 用 $\pi/\pi_{ref}$ 表示 → 直接优化 $\pi$

DPO 训练只需 **2 个模型**：
- 正在训练的 policy $\pi_\theta$
- 冻结的 reference $\pi_{ref}$

### ✅ 优缺点

**优点**：实现简单、训练资料更省、更稳定、效果通常不差
**局限**：离线方法，依赖偏好数据质量；默认使用 Bradley-Terry 偏好模型

---

## Q04 · GRPO

### 🎯 核心思想（DeepSeek）

**去掉 Critic 价值模型**，对同一个问题采样多条回答，用**组内相对奖励**代替价值估计。

### 📖 流程

对同一个问题 $x$ 采样一组回答 $\{y_1, \ldots, y_G\}$，给每条打分 $r_i$。

GRPO 的优势估计：
$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \ldots, r_G)}{\mathrm{std}(r_1, \ldots, r_G)}
$$

> ⚠️ **注**：用户笔记里写的是 `max`，但标准 GRPO 用的是 `mean`（先减组均值再除组标准差）。

直觉：

- 比组内平均明显好 → $\hat{A}_i > 0$
- 明显差 → $\hat{A}_i < 0$
- 接近平均 → $\hat{A}_i \approx 0$

→ **不再需要 critic 预测"未来值多少"**，用**同 prompt 多样本相对比较**近似 advantage。

### ✅ 优缺点

**优点**：

- 不需要价值函数
- 实验简单，对**规则奖励任务友好**（数学、代码这类有标准答案的）
- 在大模型 reasoning 任务被验证有效（DeepSeek-R1）

**缺点**：

- 组内标准化带来局部偏差
- 对组的大小敏感

### 🪤 为什么 GRPO 适合 reasoning？

1. 很多 reasoning 任务有**天然可验证的奖励**（答案对/错）
2. 同一道题多采样很有意义（多种解题路径）
3. 去掉 critic **更省资源**（critic 参数量 = actor）

---

## Q05 · 三大对齐算法对比

| 特性 | PPO | DPO | GRPO |
|---|---|---|---|
| 奖励模型 | ✅ 需要 | ❌ 不需要 | ✅ 需要（或规则） |
| Critic 模型 | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| 模型数量 | 4 | 2 | 2 + 采样 |
| 训练复杂度 | 高 | 低 | 中 |
| 在线/离线 | 在线 | 离线 | 在线 |
| 稳定性 | 较差 | 较好 | 中 |
| 显存占用 | 大 | 小 | 中 |
| 代表模型 | InstructGPT, ChatGPT | Zephyr, LLaMA-3 | **DeepSeek-R1** |

---

## Q06 · Reward Hacking

### 🎯 现象

模型学会了**钻 reward model 的漏洞**：reward 分高了，但实际质量差。

### 📖 典型例子

- 输出超长（RM 偏好长答案）
- 重复关键词（RM 看到关键词就给高分）
- 输出"我不知道"避免错答（RM 罚错答）

### 📖 缓解方案

- **KL 惩罚**：限制偏离 reference（PPO/DPO 都用）
- 用更强的 RM
- **Constitutional AI / RLAIF**：用 AI 替代部分人类标注，减少奖励噪声
- 多 RM 集成

---

## Q07 · 其他对齐方法

### REINFORCE++

- 去掉 critic 的 **PPO/GRPO 简化版**
- 用更稳的**全局基线**替代局部组内标准化
- 加入 baseline 减小方差
- 比 PPO 简单，比 REINFORCE 稳定

### DAPO（Decoupled Alignment from DPO）

- 为**长推理链 RL** 设计的 PPO/GRPO 增强版
- 重点解决 clip 机制、样本利用、超长回答的不稳定问题
- 把 clip 设计得更灵活，**放宽上侧更新** → 避免熵塌缩，保留探索性

### KTO（Kahneman-Tversky Optimization）

- 不需要成对偏好数据，只需要"喜欢/不喜欢"二元标注
- 数据更易收集

---

[⬅ 回到首页](../README.md)
