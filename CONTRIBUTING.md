# Contributing

欢迎贡献新题目、修正错误、补充追问、推荐资料。

## 添加新题目

1. **选择章节**：在 `docs/` 下挑一个最贴合的目录（如 `01-architecture/`）。
2. **复制模板**：从 [`docs/_TEMPLATE.md`](docs/_TEMPLATE.md) 复制一份。
3. **命名规范**：`QXX-题目英文短名.md`，如 `Q06-rope.md`。
4. **填写内容**：保持模板结构（一句话答案 / 详细展开 / 追问 / 参考 / 相关）。
5. **更新章节 README**：在对应章节的 `README.md` 题目列表里加一行链接。

## 添加手撕题

放在 `coding/` 目录下，命名 `XX-name.md`：

- 顶部一句话说明实现什么
- 一份**能直接跑**的 PyTorch / Python 代码
- 关键行加注释
- 末尾列**易错点 / 面试常见追问**

## 风格

- 中英文混排时，中英文之间加空格（如 `RoPE 位置编码`）。
- 数学公式用 `$...$` / `$$...$$`。
- 代码块标语言（` ```python `）。
- 图片放 `assets/images/<chapter>/<filename>`。

## PR 流程

1. Fork → 新建分支 `feat/qxx-rope`
2. Commit message：`feat(arch): add Q06 RoPE`
3. 提 PR，描述：新增 / 修改了什么、参考来源
4. 等 Review

## 不接受的内容

- ❌ 大段抄袭未注明来源
- ❌ 政治 / 敏感话题
