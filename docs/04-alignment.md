# 04 · Post-Training 与 Alignment

从 SFT、偏好学习到在线强化学习：不仅要会背 PPO / DPO / GRPO，还要能解释奖励信号如何设计、为什么训练会失效，以及怎样定位问题。

## 本章目录

- [Q01 · 后训练全景：SFT、偏好优化与在线 RL](#q01--后训练全景)
- [Q02 · 奖励模型如何训练](#q02--奖励模型如何训练)
- [Q03 · 奖励函数如何设计](#q03--奖励函数如何设计)
- [Q04 · PPO：带 Critic 的在线 RL](#q04--ppo)
- [Q05 · DPO：直接偏好优化](#q05--dpo)
- [Q06 · GRPO：组相对策略优化](#q06--grpo)
- [Q07 · DAPO：让长推理 RL 更稳定](#q07--dapo)
- [Q08 · GSPO 与 GDPO](#q08--gspo-与-gdpo)
- [Q09 · 其他后训练算法](#q09--其他后训练算法)
- [Q10 · Reward Hacking 与 Reward Overoptimization](#q10--reward-hacking-与-reward-overoptimization)
- [Q11 · 训练监控与故障定位](#q11--训练监控与故障定位)
- [Q12 · 算法选型](#q12--算法选型)

---

## Q01 · 后训练全景

### 一句话

**预训练学习“语言和知识”，SFT 学习“如何按要求回答”，偏好优化与 RL 学习“多个可行回答中应该偏向哪一个”。**

### 典型链路

```text
Pre-training
  → SFT / CoT 冷启动
  → 偏好数据或可验证任务
  ├─ Reward Model → PPO / GRPO 等在线 RL
  ├─ Rule-based Reward → RLVR / GRPO / DAPO
  └─ chosen-rejected 偏好对 → DPO / IPO / ORPO 等离线偏好优化
  → 离线评测、红队与数据回流
```

| 阶段 | 数据 | 训练信号 | 主要作用 |
|---|---|---|---|
| SFT | `(prompt, response)` | token-level CE | 教格式、知识、基本行为和初始推理轨迹 |
| Reward Modeling | `(prompt, chosen, rejected)` 或排序 | 偏好比较损失 | 学习一个可微或可调用的评分器 |
| 离线偏好优化 | chosen-rejected / 点式反馈 | DPO、KTO 等损失 | 不在线采样，直接提高偏好回答的相对概率 |
| 在线 RL | prompt + policy rollout | RM / 规则 / 验证器奖励 | 在当前策略分布上探索并优化高奖励轨迹 |

### RLHF、RLAIF 与 RLVR

- **RLHF**：反馈主要来自人类偏好标注。
- **RLAIF**：使用强模型依据准则生成偏好或评分，成本低但会继承 Judge 偏差。
- **RLVR**：Reinforcement Learning with Verifiable Rewards，使用答案验证器、单元测试、编译器或形式系统等可验证信号。奖励更客观，但只适用于可验证任务。

### LLM 中的 RL 要素

| RL 概念 | LLM 中的对应 |
|---|---|
| Policy / Actor | 待训练模型，通常从 SFT checkpoint 初始化 |
| Reference Policy | 冻结模型，用于约束策略不要漂移太远 |
| State | prompt 与已经生成的 token 前缀 |
| Action | 下一个 token |
| Trajectory | 一条完整 response |
| Reward | 对整条 response 或中间步骤的评分 |
| Critic / Value Model | 估计状态价值，为 policy gradient 提供低方差 baseline |

### 面试常见追问

- **Q：RL 能不能凭空教会模型新知识？** A：通常不擅长。RL 更像从模型已有分布中放大高奖励轨迹；若基础能力和探索概率几乎为零，应先补预训练、SFT、蒸馏或更好的数据。

- **Q：DPO 算不算强化学习？** A：它来自 KL 正则化 RLHF 的闭式最优策略推导，但工程上是使用固定偏好数据的监督式离线优化，不需要在线 rollout、环境交互或显式 Reward Model。

---

## Q02 · 奖励模型如何训练

### 核心目标

给定 prompt $x$ 和回答 $y$，学习标量奖励 $r_\phi(x,y)$，使人类更偏好的回答得分更高。

### Bradley-Terry 成对偏好损失

对 chosen 回答 $y_w$ 和 rejected 回答 $y_l$：

```math
P(y_w \succ y_l \mid x)
= \sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
```

```math
\mathcal{L}_{RM}
=-\mathbb{E}\left[
\log \sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\right]
```

模型通常在最后一个有效 token 的 hidden state 上接一个 scalar head。该损失只约束**分差**，奖励整体加同一个常数不影响偏好概率。

### 常见反馈形式

| 形式 | 示例 | 优点 | 局限 |
|---|---|---|---|
| Pointwise | 单条回答 1~5 分 | 数据直观 | 标注者尺度不一致，校准困难 |
| Pairwise | A 优于 B | 判断更稳定，最常用 | 不能直接表达差距大小和平局 |
| Listwise | 多条回答整体排序 | 单个 prompt 信息密度高 | 标注成本高，排序一致性难保证 |
| Binary feedback | 喜欢 / 不喜欢 | 收集便宜，可用于 KTO | 信号粗，类别比例可能失衡 |
| Process label | 中间步骤对 / 错 | credit assignment 更细 | 标注或自动验证成本高 |

### ORM 与 PRM

- **ORM（Outcome Reward Model）**：只评最终结果。便宜，但正确答案可能来自错误推理，且信号稀疏。
- **PRM（Process Reward Model）**：逐步评价中间推理。credit assignment 更细，但步骤切分、标注一致性和错误传播更复杂。
- 实践中可组合：结果正确性作为主目标，过程信号用于塑形和诊断，但要防止模型为了迎合过程模板而牺牲真正正确性。

### 数据质量要点

1. **同 prompt 比较**：避免 prompt 难度成为奖励分差的混杂变量。
2. **控制位置偏差**：交换候选顺序，过滤不一致标注。
3. **允许 tie / 无法判断**：强迫细微差异形成胜负会注入噪声。
4. **保留难负例**：语法流畅但事实错误的回答比明显乱码更能训练出有效 RM。
5. **按来源和难度切分**：防止相似回答泄漏到 train / test。
6. **监控标注一致性**：看标注者间一致率、重标一致率和各维偏好分布。

### RM 训练准确率很高，为什么 RL 仍可能失败？

- RM 只在标注数据分布上准确，policy 更新后会生成 OOD 回答；
- policy 会主动搜索 RM 的漏洞，普通测试集未覆盖这种对抗分布；
- pairwise accuracy 只看排序对错，不代表奖励尺度已校准；
- RM 可能学到长度、格式、措辞等伪相关特征。

---

## Q03 · 奖励函数如何设计

### 设计原则

奖励函数不是“把所有好指标加起来”，而是把目标转成**稳定、可区分、难投机且与优化粒度一致**的训练信号。

一个抽象的多目标奖励可以写成：

```math
R(y)=
w_cR_{correct}
+w_pR_{process}
+w_fR_{format}
+w_sR_{safety}
-w_lP_{length}
-w_hP_{hack}
```

这只是结构示意；直接加权前必须处理每个分量的尺度、方差、稀疏度和优先级。

### 奖励来源

| 来源 | 适合评价 | 优点 | 主要风险 |
|---|---|---|---|
| 规则 / 验证器 | exact match、编译、单测、格式 | 快、稳定、可解释 | 覆盖窄，规则边界可被利用 |
| Reward Model | 偏好、风格、安全、综合质量 | 可泛化到开放输出 | 分布外失真、reward hacking |
| LLM-as-Judge | 语义质量、事实一致性、成对比较 | 无需单独训练 RM | 位置、长度、自我偏好；成本与延迟高 |
| 外部工具 | 检索核验、执行结果、形式证明 | 客观、可追溯 | 工具错误、沙箱和超时会污染奖励 |

### 硬约束、软奖励与门控

- **硬约束**适合“违反即不可用”的条件，例如无法解析、危险输出。可以直接判无效，但过多硬门控会造成奖励稀疏。
- **软奖励**适合连续质量，例如完整性、简洁性、过程质量。
- **条件化奖励**可表达优先级：只有主任务正确时才给格式或简洁奖励，防止模型只刷容易指标。

```text
不推荐：总分 = 0.9 × 格式 + 0.1 × 正确性
推荐：  正确性不通过 → 不发放风格奖励
        正确性通过   → 再比较格式、过程与效率
```

权重很大不等于优先级一定可靠：如果一个奖励容易达到、方差又大，它仍可能主导梯度。

### 多奖励归一化

设第 $k$ 个奖励分量为 $r_{i,k}$。常见方案包括：

1. **先加权求和，再统一归一化**：简单，但不同奖励组合可能映射到相同总分。
2. **各维先标准化，再加权求和**：保留不同维度的相对差异，GDPO 属于这一思路。
3. **约束 / 词典序优化**：先满足核心目标，再优化次级目标。

对各维奖励至少监控：均值、标准差、通过率、饱和率、与总奖励的相关性，以及它对梯度方向的实际贡献。

### 稀疏奖励与 Reward Shaping

只有最终 0/1 奖励时，训练容易出现方差大、探索难和大量零优势组。可以加入过程奖励或潜势型 shaping，但要满足两点：

- shaping 信号和最终目标一致，不能鼓励“看起来像正确”的套路；
- 分开记录原始 outcome reward 与 shaping reward，避免总分上涨掩盖任务成功率下降。

### 长度奖励

长度既不是天然正奖励，也不是天然负奖励。

- 奖励长度容易诱导冗长和重复；
- 统一惩罚长答案会伤害确实需要长推理的样本；
- 超过最大长度就突然给负分，会把“答案错”和“被截断”混成一个信号。

更稳妥的做法是：在保证正确性的条件下鼓励效率，并在接近长度上限时使用平滑惩罚。DAPO 的 overlong reward shaping 就用于减小硬截断噪声。

### 奖励函数上线前检查

1. 用人工构造的好 / 坏 / 对抗样例做单元测试。
2. 检查改变无关特征（措辞、位置、长度）是否意外改变奖励。
3. 看每个 prompt 的 rollout 是否有足够分差，而不只看全局方差。
4. 对 RM / Judge 做 held-out 与对抗集评测。
5. 同时报告任务指标和代理奖励，禁止只看 reward 曲线。
6. 小规模训练后人工检查“高奖励但低质量”样本，再扩大训练。

### 面试回答模板

> 我会先把目标拆成结果正确性、过程质量、格式 / 安全约束等分量，区分硬门控与软奖励；再校准各维尺度和稀疏度，检查组内区分度及权重是否真的影响梯度；最后用对抗样例、held-out Judge 和真实任务指标监控 reward hacking，不能只看总 reward 上升。

---

## Q04 · PPO

### 一句话

PPO 用 Critic 估计 advantage，并通过 clipped surrogate objective 限制每次策略更新幅度。

### 参与组件

| 组件 | 作用 | 是否训练 |
|---|---|---|
| Actor / Policy | 生成回答并接受策略梯度 | 是 |
| Critic / Value Model | 估计每个 token 状态的价值 | 是 |
| Reward Model / Function | 对 rollout 评分 | 通常冻结或外部调用 |
| Reference Model | 计算 KL 约束 | 冻结 |

工程上不一定真有四份完整模型：Actor 与 Critic 可共享骨干，奖励也可能来自规则或远程服务。但 PPO 的状态、显存和训练链路通常仍最复杂。

### Clipped objective

```math
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}
```

```math
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
```

clip 不是把梯度永远限制在固定范围，而是在样本会推动策略越过可信区间时截断其收益，形成保守更新。

### GAE

```math
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
```

```math
\hat A_t^{GAE}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
```

- $\lambda$ 小：偏差更大、方差更小；
- $\lambda$ 大：偏差更小、方差更大；
- LLM 常只有序列末端 outcome reward，需要把回报分配到生成 token。

### KL 约束

常见目标：

```math
\max_\pi\;
\mathbb{E}[r(x,y)]
-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})
```

$\beta$ 可固定，也可根据 target KL 自适应调整。KL 太小容易过度优化奖励，太大则几乎学不到新偏好。

### 优缺点

- **优点**：在线探索当前策略分布，能利用标量奖励，适合复杂非静态目标。
- **缺点**：Actor / Critic 协同训练难，rollout 成本高，对 reward scale、KL、GAE 和 batch 配置敏感。

---

## Q05 · DPO

### 核心思想

将 KL 正则化 RLHF 的隐式奖励改写为 policy 与 reference 的对数概率比，直接用偏好对优化策略。

### 从最优策略到 DPO

KL 正则化奖励最大化的最优策略满足：

```math
\pi^*(y\mid x)=
\frac{1}{Z(x)}\pi_{ref}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
```

因此：

```math
r(x,y)=
\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}
+\beta\log Z(x)
```

将其代入 Bradley-Terry 偏好模型，$Z(x)$ 在同一个 prompt 的奖励差中抵消：

```math
\mathcal{L}_{DPO}
=-\mathbb{E}\left[
\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right)
\right]
```

### $\beta$ 到底控制什么？

在上述 RLHF 推导中，$\beta$ 是 KL 正则强度：

- **$\beta$ 大**：更强地约束策略接近 reference，最优策略更保守；
- **$\beta$ 小**：允许更大策略偏移，但更容易过拟合偏好或训练不稳。

不要只从损失里“$\beta$ 乘 logits”就反向解释。超参含义要结合它来自的 KL 正则化目标理解。

### 实现与数据易错点

1. 只对 response token 求 log-prob，prompt 和 padding 必须 mask。
2. chosen / rejected 必须对应同一个 prompt 和模板。
3. 同时监控 reward margin、chosen / rejected accuracy 和 KL，不只看 loss。
4. 序列 log-prob 求和会带来长度效应；是否做长度归一化是算法变体和任务选择，不能默默改变定义。
5. 偏好数据来自旧策略时会有 off-policy / coverage 问题；DPO 不会主动探索数据外的更好回答。

### 优缺点

- **优点**：无需在线 rollout、显式 RM 和 Critic，训练简单稳定。
- **局限**：依赖固定偏好数据的覆盖与质量；不能直接使用任意标量奖励；对噪声、长度偏差和 reference 选择敏感。

---

## Q06 · GRPO

### 核心思想

对同一个 prompt 采样一组回答，用组内相对奖励构造 advantage，从而去掉 Critic / Value Model。

### 组相对 advantage

对 prompt $x$ 采样 $G$ 条回答 $\{y_1,\ldots,y_G\}$，奖励为 $\{r_1,\ldots,r_G\}$：

```math
\hat A_i=
\frac{r_i-\mathrm{mean}(r_1,\ldots,r_G)}
{\mathrm{std}(r_1,\ldots,r_G)+\varepsilon}
```

| 情况 | Advantage | 训练方向 |
|---|---:|---|
| 高于组均值 | 正 | 提高该回答 token 的概率 |
| 低于组均值 | 负 | 降低该回答 token 的概率 |
| 接近组均值 | 约为 0 | 更新很弱 |

随后可使用 PPO 风格的重要性比率、clipping 和对 reference 的 KL 惩罚更新 policy。

### 为什么不用 Critic？

PPO 用 $V(s_t)$ 作为 baseline；GRPO 用同 prompt 多次 rollout 的经验均值作为 baseline。这样省掉一个与 policy 规模接近的价值模型，但代价是每个 prompt 需要多次采样。

### 什么任务适合 GRPO？

- 有可靠的规则奖励或验证器；
- 同一 prompt 能采样出有明显质量差异的多条轨迹；
- 需要在线探索，而固定偏好数据覆盖不足；
- 基座模型已经有一定成功概率。

GRPO 不要求一定训练 Reward Model；数学答案、代码单测等场景常直接使用规则奖励。

### 主要失效模式

1. **全对 / 全错组**：奖励方差为 0，组内标准化后 advantage 接近 0。
2. **组太小**：均值和方差估计噪声大。
3. **奖励太粗**：大量 rollout 同分，分不出优劣。
4. **难度偏置**：组内标准化会改变不同 prompt 对梯度的相对贡献。
5. **长序列噪声**：sequence-level reward 广播给所有 token，credit assignment 粗。
6. **熵塌缩**：策略越来越确定，探索路径消失。

### 高频追问

- **Q：一组奖励为 `[1, 1, 1, 1]` 会怎样？** A：去均值后全为 0，整组几乎没有策略梯度。工程上要记录 zero-variance group rate，并考虑动态采样、更细奖励或补充数据。

- **Q：GRPO 一定比 PPO 稳定吗？** A：不一定。它省掉了 Critic 的误差与训练成本，但引入组内统计噪声、额外 rollout 成本和零优势组问题。

---

## Q07 · DAPO

### 一句话

DAPO（**Decoupled Clip and Dynamic sAmpling Policy Optimization**）不是“DPO 的增强版”，而是一套针对长推理在线 RL 的 GRPO 风格训练改进。

### 四个关键技术

#### 1. Clip-Higher：缓解探索不足

将对称 clipping 拆成不同上下界：

```math
\mathrm{clip}
\left(r_{i,t}(\theta),
1-\varepsilon_{low},
1+\varepsilon_{high}\right),
\qquad
\varepsilon_{high}>\varepsilon_{low}
```

更高的上界给低概率 token 增加概率的空间，缓解策略过早确定化。它不是取消约束，而是对“下降”和“探索性上升”使用不对称信任区间。

#### 2. Dynamic Sampling：过滤零优势组

过采样并只保留奖励不全相同的 prompt 组，使 batch 中更多样本有有效梯度。代价是额外 rollout、困难度分布改变和潜在选择偏差，因此要同时监控保留率与采样成本。

#### 3. Token-Level Policy Gradient Loss：改变聚合权重

样本级聚合：

```math
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}g_{i,t}
```

token 级聚合：

```math
\frac{1}{\sum_i|y_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|y_i|}g_{i,t}
```

前者每条回答等权，后者 batch 内每个 token 等权。它改变了长短回答对梯度的相对贡献，不能简单理解成更细粒度的 token reward。

#### 4. Overlong Reward Shaping：降低截断噪声

在接近最大长度时平滑增加惩罚，而不是“只要截断就统一负分”。这样能区分任务失败和长度超限，减少奖励突变。

### DAPO 解决了什么，没解决什么？

- 解决：探索空间、有效样本比例、loss 聚合和超长截断噪声。
- 没有自动解决：奖励正确性、验证器漏洞、基础模型完全不会做题、多奖励互相冲突。

---

## Q08 · GSPO 与 GDPO

这两类较新的方法分别处理**策略更新粒度**和**多奖励归一化**，与 DAPO 的关注点不同。面试时应把它们表述为对 GRPO 训练问题的后续改进，而不是 GRPO 原始定义的一部分。

### GSPO：序列级重要性比率

GRPO / PPO 风格目标通常计算 token-level ratio：

```math
w_{i,t}(\theta)=
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,1:t-1})}
{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,1:t-1})}
```

GSPO（Group Sequence Policy Optimization）改为一条回答共享 sequence ratio：

```math
s_i(\theta)=
\left(
\frac{\pi_\theta(y_i\mid x)}
{\pi_{\mathrm{old}}(y_i\mid x)}
\right)^{1/|y_i|}
=
\exp\left(
\frac{1}{|y_i|}\sum_t
\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,1:t-1})}
{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,1:t-1})}
\right)
```

直觉：Reward 若是 sequence-level，importance ratio 和 clipping 也在 sequence-level 对齐；长度归一化的几何均值避免序列概率随长度指数缩小。论文还强调它能减轻长序列和 MoE 路由变化带来的训练不稳定。

### GDPO：多奖励解耦归一化

普通做法先合并多维奖励再组内归一化，不同奖励组合可能坍缩成相同 advantage。GDPO（Group reward-Decoupled Normalization Policy Optimization）对每个奖励维度先独立标准化：

```math
\hat A_{i,k}
=\frac{r_{i,k}-\mu_k}{\sigma_k+\varepsilon},
\qquad
A_i=\sum_{k=1}^{K}w_k\hat A_{i,k}
```

再做批次级稳定化，从而更完整地保留各维奖励差异。

### 三类改进放在一起看

| 环节 | 典型问题 | 方法 |
|---|---|---|
| 多奖励构造 | 不同奖励组合合并后坍缩 | GDPO：各维先归一化再合并 |
| policy ratio | token ratio 与序列奖励粒度错配 | GSPO：sequence ratio 与 sequence clipping |
| rollout | 全对 / 全错组无梯度 | DAPO：Dynamic Sampling |
| exploration | 低概率路径难以提升 | DAPO：Clip-Higher |
| loss 聚合 | 长短回答权重不合理 | DAPO：Token-Level PG Loss |
| 长度控制 | 硬截断制造奖励噪声 | DAPO：Overlong Reward Shaping |

---

## Q09 · 其他后训练算法

### REINFORCE / RLOO / ReMax

- **REINFORCE**：最基本的 Monte Carlo policy gradient，直接用完整回报更新，简单但方差大。
- **RLOO**（REINFORCE Leave-One-Out）：同 prompt 多次采样，第 $i$ 条回答用其他回答的平均奖励作 baseline，避免把自身奖励混入 baseline。
- **ReMax**：用贪心解码回答的 reward 作为 baseline，不训练 Critic；实现轻，但 baseline 质量依赖贪心输出。

它们和 GRPO 的共同目标是：不用 Value Model，通过合适的 baseline 降低 policy gradient 方差。

### KTO

KTO（Kahneman-Tversky Optimization）只需要 desirable / undesirable 单条二元反馈，不要求同 prompt 的成对偏好。适合历史点赞、踩等天然不成对的数据，但要注意正负样本不平衡与反馈噪声。

### IPO

IPO（Identity Preference Optimization）从偏好优化的正则化目标出发，使用平方损失约束偏好 margin，可减轻 DPO 在确定性偏好和可分数据上不断增大 margin 的倾向。

### ORPO

ORPO 将 SFT 的负对数似然与 chosen / rejected 的 odds-ratio 偏好项合并，在一个阶段内兼顾生成建模和偏好拉开，不需要单独 reference model。代价是 SFT 与偏好项的平衡需要调节。

### Constitutional AI

先定义一组原则，让模型进行批评、修订并生成 AI 偏好，再做 SFT / RLAIF。它是**数据与监督来源的方法论**，不是与 PPO / DPO 同层的单一优化器。

---

## Q10 · Reward Hacking 与 Reward Overoptimization

### 区别

- **Reward Hacking**：模型找到代理奖励的漏洞，例如堆关键词、重复模板、伪造格式或拖长回答。
- **Reward Overoptimization**：训练 reward 继续上升，但真实质量或 held-out Judge 开始下降；即使没有直观“作弊动作”，也属于 Goodhart's Law 下的代理目标失真。

### 为什么会发生？

1. 奖励只是人类目标的代理，不可能覆盖所有质量维度；
2. policy 会主动搜索 RM 训练分布之外的高分区域；
3. RM 容量有限、标注有噪声，还可能偏好长度和格式；
4. 固定 RM 在 RL 过程中不更新，误差会被策略反复放大；
5. 多奖励权重或门控设计不当，容易指标压过核心目标。

### 缓解方法

- KL 约束、early stopping、限制单轮 policy 更新幅度；
- 使用独立 held-out RM / Judge 检查泛化，不用训练奖励自证；
- RM ensemble 与不确定性惩罚，降低 OOD 高分样本权重；
- 人工抽检 top-reward 与 reward disagreement 样本；
- 对抗测试、反事实测试和规则单元测试；
- 将可验证 correctness 作为锚点，开放式质量作为辅助信号；
- 迭代更新 RM 数据，加入 policy 新产生的 hard negatives；
- 同时监控真实任务指标、奖励各分量、KL、长度和多样性。

### 为什么“加一个更强 RM”还不够？

更强 RM 仍是代理模型，也会有盲区。关键是形成闭环：**训练奖励、独立评估、对抗样本、人工审查与数据回流**相互制衡。

---

## Q11 · 训练监控与故障定位

### 必看指标

| 指标 | 观察什么 | 异常含义 |
|---|---|---|
| raw reward / 各分量 | 均值、方差、分位数、通过率 | 饱和、尺度漂移或单一奖励主导 |
| task metric | accuracy、pass rate、人工胜率 | 代理 reward 是否真的有效 |
| group reward std | 每个 prompt 的组内区分度 | 过低表示 GRPO 学习信号弱 |
| zero-advantage group rate | 全同奖励组占比 | 采样预算被无梯度样本浪费 |
| KL to reference | 策略漂移 | 过高可能失控，过低可能没学到 |
| entropy | 探索和多样性 | 快速下降可能熵塌缩 |
| clip fraction | 被截断更新比例 | 过高表示步子过大；过低也可能更新太弱 |
| response length / truncation | 长度分布和截断率 | 长度投机或 max length 噪声 |
| importance ratio | 新旧策略分布差 | 长尾异常可能引发不稳定 |
| gradient norm | 优化稳定性 | 爆炸、消失或异常尖峰 |
| rollout throughput | 采样速度和有效样本率 | RL 常受生成而非反向传播限制 |

### 症状到原因

| 症状 | 优先检查 | 常见处理 |
|---|---|---|
| reward 涨、任务指标跌 | reward hacking、长度与格式相关性 | 独立 Judge、重做奖励、早停、补 hard negatives |
| KL 突然上升 | 学习率、reward scale、clip、旧 log-prob | 降步长、调 KL、核对 rollout / train 一致性 |
| entropy 快速下降 | clip 上界、采样温度、奖励过于确定 | 增加探索、Clip-Higher、提高样本多样性 |
| GRPO loss 几乎为 0 | 全对 / 全错组、奖励量化过粗 | 动态采样、细化奖励、调整题目难度 |
| 回答越来越长 | 长度偏好、按样本归一化、截断奖励 | 条件化效率奖励、平滑长度惩罚 |
| 某个奖励很快满分 | 奖励过易或权重主导 | 分维监控、条件化奖励、GDPO 式归一化 |
| loss / ratio 出现尖峰 | 数值精度、新旧策略错配、MoE 路由 | log-space 计算、核对 mask、考虑序列级 ratio |

### 训练前先做什么小实验？

拿少量 prompt，每个采样多条回答，打印原始奖励分量、总分、组内 advantage、长度和解析结果。人工确认“更好的回答是否真的得到更高且可区分的 advantage”，再启动大规模训练。

---

## Q12 · 算法选型

| 方法 | 数据 / 奖励 | 在线采样 | Critic | 主要优势 | 主要局限 |
|---|---|---:|---:|---|---|
| PPO | 标量 RM / 规则奖励 | 是 | 是 | 通用、能在线探索 | 资源和调参成本最高 |
| DPO | chosen-rejected 偏好对 | 否 | 否 | 简单、稳定、成本低 | 受固定数据覆盖限制 |
| KTO | 单条喜欢 / 不喜欢 | 否 | 否 | 不需要配对数据 | 反馈粗、类别不平衡 |
| GRPO | 同 prompt 多 rollout + 标量奖励 | 是 | 否 | 省 Critic，适合可验证推理 | 依赖组内区分度，采样成本高 |
| DAPO | 可验证奖励 + 长推理 rollout | 是 | 否 | 改善探索、有效样本与长度训练 | 系统更复杂，仍依赖正确奖励 |
| GSPO | 序列级 reward + 在线 rollout | 是 | 否 | ratio 与序列奖励对齐，长序列 / MoE 更稳 | 序列级更新粒度未必适合所有任务 |
| GDPO | 多维奖励 + 组采样 | 是 | 否 | 保留多奖励差异 | 各维奖励本身仍需可靠设计 |

### 选型顺序

1. 只有高质量示范数据：先 SFT。
2. 有固定偏好对、算力有限：优先 DPO 类方法。
3. 有可靠验证器且需要探索新轨迹：GRPO / DAPO 类在线 RL。
4. 奖励开放且不可验证：先做好 RM 与独立评估，再考虑 PPO / GRPO。
5. 组内奖励经常全同：先修数据、采样和奖励，不要靠换优化器掩盖问题。
6. 多奖励组合丢失区分度：考虑分维归一化或约束式目标。
7. 长序列 / MoE 的 token ratio 不稳：考虑 sequence-level ratio 思路。

---

## 参考论文

- [Training language models to follow instructions with human feedback（InstructGPT）](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath：提出 GRPO](https://arxiv.org/abs/2402.03300)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Group Sequence Policy Optimization（GSPO）](https://arxiv.org/abs/2507.18071)
- [GDPO: Group reward-Decoupled Normalization Policy Optimization](https://arxiv.org/abs/2601.05242)

---

[⬅ 回到首页](../README.md)
