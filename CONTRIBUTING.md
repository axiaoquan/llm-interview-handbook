# Contributing

欢迎贡献新题目、修正错误、补充追问、推荐资料。

## 添加新题目

1. **选择章节**：在 `docs/` 下挑一个最贴合的文件（如 `01-architecture.md`）。
2. **复用模板结构**：参考 [`docs/_TEMPLATE.md`](docs/_TEMPLATE.md)，在现有章节中增加 `## QXX · 题目`，不是每题新建文件。
3. **必要时新建章节**：只有现有结构容纳不下时才创建 `XX-name.md`，避免重复解释同一知识点。
4. **填写内容**：一句话答案 / 目标与假设 / 推导 / 替代设计 / 手算或反例 / 实现边界 / 来源。
5. **更新索引**：维护文件顶部目录、根 README；手撕章还需更新 `llm-coding/README.md`，并检查实际标题锚点。

## 添加手撕题

放在 `llm-coding/` 目录下，命名 `XX-name.md`：

- 顶部一句话说明实现什么
- 明确运行级别：完整程序、可调用组件、依赖外部模型的示例或局部片段；局部控制流用 `# 片段：` 标注
- 关键行加注释
- 末尾列**易错点 / 面试常见追问**
- 解释关键常数、缩放、维度和参数为什么这样设计，而不只给代码结论
- 修改可运行组件时，在 `tests/` 增加针对正文代码的输出、梯度或边界回归测试

## 验证要求

- 先执行 `python -m pip install -r tests/requirements-docs.txt` 安装固定版本文档解析器。
- `python scripts/check_docs.py`：CommonMark 围栏、Python 编译、数学定界符/括号/已知危险写法、本地文件链接的静态检查；不能替代 LaTeX 引擎或 GitHub 渲染。
- `python -m unittest discover -s tests -v`：需要 PyTorch，当前测试范围见 [tests/README.md](tests/README.md)。
- 公式改动发布前须在目标 GitHub 页面复核，尤其是分式、下标、绝对值和多行式。
- 数学推导写清假设；库行为引用固定版本官方文档，模型结论对应具体论文版本。不能把数值无偏、梯度无偏、数学非负和浮点稳定混为一谈。
- main push 和 PR 自动运行文档、KaTeX 与 CPU 回归检查。工作流通过不等于全部示例已获得运行认证，仍需阅读测试覆盖边界。

## 风格

- 中英文混排时，中英文之间加空格（如 `RoPE 位置编码`）。
- 行内数学用 `$...$`；复杂展示公式优先使用 ` ```math ` 围栏，避免 Markdown 先解析 `<...>` 等内容。
- 代码块标语言（` ```python `）。
- 图片放 `assets/images/<chapter>/<filename>`。
- 标题不使用装饰性 emoji。
- 避免“永远、一定、最强、固定提速 X 倍”等无条件表述；注明假设、指标和适用边界。

## PR 流程

1. Fork → 新建分支 `feat/qxx-rope`
2. Commit message：`feat(arch): add Q06 RoPE`
3. 提 PR，描述：新增 / 修改了什么、参考来源
4. 等 Review

## 不接受的内容

- 大段抄袭且未注明来源
- 与技术面试无关的敏感内容
