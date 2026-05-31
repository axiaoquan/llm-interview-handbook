# 07 · Agent

## 📑 本章目录

- [Q01 · Agent 是什么 / 与 LLM 的区别](#q01--agent-是什么)
- [Q02 · ReAct / Plan-and-Solve / Reflexion](#q02--react--plan-and-solve--reflexion)
- [Q03 · Function Calling](#q03--function-calling)
- [Q04 · Tool Use 训练](#q04--tool-use-训练)
- [Q05 · Memory（短期 / 长期）](#q05--memory)
- [Q06 · Multi-Agent 协作](#q06--multi-agent-协作)

---

## Q01 · Agent 是什么

### 🎯 核心定义

Agent = **LLM + 工具 + 记忆 + 规划循环**。

简单 LLM 是"输入 → 输出"；Agent 能**循环决策**：

```
观察(O) → 思考(T) → 行动(A) → 观察(O') → ...
```

### 📖 与单纯 LLM 的差异

| | LLM | Agent |
|---|---|---|
| 输入 | prompt | prompt + 工具结果 |
| 输出 | 文本 | 文本 + 工具调用 |
| 状态 | 无 | 有（记忆） |
| 步骤 | 单步 | 多步循环 |

---

## Q02 · ReAct / Plan-and-Solve / Reflexion

### 📖 ReAct（Reasoning + Acting）

让模型**交替输出思考和行动**：
```
Thought: 我需要先搜一下当前股价
Action: search("AAPL stock price")
Observation: $185.4
Thought: 现在我可以回答用户了
Final Answer: ...
```

### 📖 Plan-and-Solve

**先规划，再执行**：
1. Plan：先把任务拆成子步骤
2. Solve：按计划逐步执行

适合复杂多步任务，比 ReAct 更结构化。

### 📖 Reflexion

引入**反思机制**：每次行动失败后，模型反思为什么失败，作为下次输入。
适合需要"试错"的场景。

---

## Q03 · Function Calling

### 🎯 实现机制

让模型输出**结构化的函数调用 JSON**，由外部代码执行。

```json
{
  "name": "get_weather",
  "arguments": {"city": "Shenzhen", "unit": "celsius"}
}
```

### 📖 工作流

```
1. 系统提示注册可用工具(name, description, params)
2. 模型决定要不要调用 + 调哪个 + 参数
3. 外部代码执行函数
4. 结果作为新 message 送回模型
5. 模型综合结果给最终回答
```

### 🪤 追问

- **Q：Function Calling 怎么训练？**
  A：构造（prompt + 工具描述 + 应该调的工具调用 JSON）数据集做 SFT，或用 RL 让模型学会"啥时候该调啥"。

---

## Q04 · Tool Use 训练

### 📖 数据构造

- **种子工具**：定义一组有用的工具（搜索、计算器、代码执行等）
- **生成数据**：让 GPT-4 生成"用户问题 + 多步工具调用 + 答案"
- **筛选**：保留**实际能跑通**的轨迹

### 📖 训练范式

1. **SFT**：直接学工具调用轨迹
2. **DPO**：偏好"成功完成 vs 失败的轨迹"
3. **RL**：以任务完成度为奖励（如 ToolBench）

---

## Q05 · Memory

### 📖 短期记忆

当前对话的上下文。
**问题**：上下文窗口有限，长对话会丢早期信息。
**方案**：滑窗 + 自动总结。

### 📖 长期记忆

跨会话保留信息。
常见做法：

- **向量数据库**：把对话历史向量化存储，需要时检索（RAG 思路）
- **结构化记忆**：用模型抽取关键事实存为 JSON
- **混合**：MemGPT 等

---

## Q06 · Multi-Agent 协作

### 📖 经典架构

| 架构 | 思想 |
|---|---|
| **辩论（Debate）** | 多 Agent 互相挑战，提升鲁棒性 |
| **分工（Role Play）** | 不同 Agent 扮演不同角色（CEO/工程师/QA） |
| **共识** | 多 Agent 投票决定 |

### 📖 代表框架

- **AutoGen**（微软）
- **MetaGPT**：模拟软件公司的多 Agent 协作
- **CrewAI**：业务流程类多 Agent

### 🪤 追问

- **Q：Multi-Agent 一定比 Single Agent 好吗？**
  A：不一定。**调度成本高**，简单任务反而 single 更好。复杂任务（写复杂代码、研究类任务）才有明显增益。

---

[⬅ 回到首页](../README.md)
