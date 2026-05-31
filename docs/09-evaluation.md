# 09 · Evaluation 评测

## 📑 本章目录

- [Q01 · Perplexity 与局限](#q01--perplexity-与局限)
- [Q02 · 主流基准（MMLU / GSM8K / HumanEval / HELM）](#q02--主流基准)
- [Q03 · LLM-as-Judge 偏差](#q03--llm-as-judge-偏差)
- [Q04 · Hallucination 检测](#q04--hallucination-检测)
- [Q05 · Lost in the Middle](#q05--lost-in-the-middle)

---

## Q01 · Perplexity 与局限

### 🎯 定义

Perplexity = 模型对测试集的不确定性，等于交叉熵的指数：

$$
\mathrm{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{1:i-1})\right)
$$

PPL 越低，模型越确定。直观理解：模型每一步在多少个候选中"犹豫"。

### ⚠️ 局限

- **不能跨 tokenizer 比较**（不同分词导致 N 不同）
- **不反映指令遵循能力**（PPL 低不等于回答好）
- 对**校准、风格、安全**全无感知

→ 现代 LLM 评测**不再以 PPL 为主**，更多看下游任务表现。

---

## Q02 · 主流基准

| 基准 | 测什么 | 形式 |
|---|---|---|
| **MMLU** | 通用知识（57 个领域） | 多选 |
| **GSM8K** | 小学数学推理 | 自由生成 + 答案 |
| **MATH** | 高中竞赛数学 | 自由生成 + 答案 |
| **HumanEval** | 代码生成 | pass@k |
| **MBPP** | Python 编程 | pass@k |
| **HellaSwag** | 常识推理 | 选择 |
| **TruthfulQA** | 真实性 | 选择 |
| **BBH** | Big Bench Hard 推理 | 多种 |
| **HELM** | 综合（公平、效率、健壮性） | 综合 |
| **C-Eval / CMMLU** | 中文通用知识 | 多选 |
| **AGIEval** | 高考 / 公务员等中文考试 | 选择 |

### 🪤 追问

- **Q：pass@k 怎么计算？**
  A：让模型生成 k 个解，**至少有一个通过单测就算 pass**。无偏估计公式：

  $$
  \text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
  $$

  其中 n 是采样总数，c 是通过数。

- **Q：基准容易刷分怎么办？**
  A：1) 防数据污染（hash 检测训练集泄漏）；2) 用 **LiveBench / Arena** 类不断刷新题库的评测；3) 看人类对战胜率（Chatbot Arena）。

---

## Q03 · LLM-as-Judge 偏差

用 LLM（通常 GPT-4）当评委给两个回答打分。

### ⚠️ 已知偏差

| 偏差 | 现象 | 缓解 |
|---|---|---|
| **位置偏好** | 偏向第一个回答 | 双向交换打两次 |
| **冗长偏好** | 偏向长答案 | 加长度惩罚 / 提示评委忽略长度 |
| **自我偏好** | 偏向同模型生成的答案 | 用更强的第三方模型当评委 |
| **风格偏好** | 偏向格式漂亮的 | 在 prompt 里强调内容优先 |

### 📖 替代方案

- **Chatbot Arena**：人类盲评 + ELO 评分
- **MT-Bench**：80 道精选题 + GPT-4 评委（两两对决）

---

## Q04 · Hallucination 检测

### 📖 类型

| 类型 | 例子 |
|---|---|
| **事实性幻觉** | 编造不存在的论文 / 人物 |
| **逻辑性幻觉** | 算错数 / 推理跳步 |
| **指令幻觉** | 明明要求中文却答英文 |

### 📖 检测方法

- **自洽性**：多次采样看是否一致（SelfCheckGPT）
- **外部检索**：检索证据验证（RAG 框架）
- **NLI 蕴含**：用 NLI 模型判断答案是否被 context 蕴含
- **Token 概率**：低概率 token 处是幻觉高发区

---

## Q05 · Lost in the Middle

### 🎯 现象

LLM 在长上下文中，**对开头和结尾的信息利用率高，对中间的低**——典型 U 型曲线。

### 📖 缓解

- 把**最重要的 context 放开头或结尾**
- 用 RAG **筛选少量相关 chunk**（少而精）
- 用支持长上下文的模型（Gemini-1.5、Claude-3 等显著缓解）
- 在评测中显式测试中间位置（Needle-in-a-Haystack）

---

[⬅ 回到首页](../README.md)
