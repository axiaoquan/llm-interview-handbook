# 本地验证

回归测试直接提取 `llm-coding/*.md` 中的 import、函数和类定义，不执行示例调用处的占位变量；不维护另一套实现副本。

## 运行

需要 Python 3.9+ 和 PyTorch（本轮验证版本为 2.8.0，CPU）。建议使用隔离虚拟环境：

```bash
python -m pip install -r tests/requirements-docs.txt
python -m pip install torch==2.8.0
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

静态检查使用固定版本的 markdown-it-py 解析 CommonMark 容器，无需 PyTorch。单独验证文档检查器可运行 `python -m unittest discover -s tests -p test_doc_checks.py -v`。测试失败时检查正文中的实现，不能只修改测试来迎合结果。

可选：在仓库外的临时 Node 环境安装 `katex@0.16.22`，将 `NODE_PATH` 指向其 `node_modules`，运行 `node scripts/check_math.cjs`。它通过 KaTeX 解析提取到的公式，能发现仅靠括号检查漏掉的语法错误；这仍不等于 GitHub 的 Markdown/MathJax 页面已视觉验证。

## 覆盖范围

- LoRA：四种初始化、首步梯度/有效更新、整个基座冻结、低精度参数、eval merge 等价与训练模式拒绝。
- Norm：LN/RMSNorm/二维 BN 的输出及输入/仿射参数梯度对齐；BN running statistics；常量、单特征、BF16 归约。
- Attention：布尔与加性 mask、全遮挡行、eval dropout、官方输出/梯度对齐、full/single/chunked cache 等价、未来泄漏与 padding。
- Loss：稳定 CE、LM shift、全 ignore、label smoothing 与官方定义对齐、GRPO 聚合口径/固定 old 与 ref/同奖励组。
- BPE：完整 token 边界、重叠 pair、频次累加、基础与中间 token 覆盖、两种编码流程。
- 生成与其他：逐样本 EOS、零生成长度、MoE 权重/梯度等价、dispatch 参考值、top-1 router 梯度、共享 embedding。

默认比较使用 `torch.testing.assert_close`，FP64 原理测试与 BF16 近似测试分开；测试使用固定随机种子和小张量，无需下载模型。

## 文档检查器与知识例子

- CommonMark 围栏解析覆盖缩进、列表、引用、反引号/波浪线和长围栏；要求显式闭合，不再仅匹配顶格三个反引号。
- 检查行内/展示数学的未闭合或混用定界符、数学括号、受限宏；代码块/行内代码不作为数学正文。
- 字面美元符号须写成 `\$`，跨行公式使用 `$$...$$` 或 math 围栏；这是仓库写作约定，不是声称完整模拟 GitHub 数学解析器。
- `test_doc_checks.py` 包含此前漏检的梯度裁剪公式回归，并覆盖引用式文件链接。
- `test_knowledge_examples.py` 复核知识章节的手算数值和数学恒等式，不属于新增手撕题，也不证明整个训练系统正确。

GitHub Actions 在 main push、pull request 或手动触发时执行文档检查、KaTeX 解析和 CPU 回归测试；工作流只有只读仓库权限，不部署、不修改文件。依赖下载、远端运行结果应与本地测试结果分别确认。

## 验证边界

这不是全仓所有示例的运行认证：测试不覆盖全部通用算法、完整 SFT/RL 训练、真实模型生成质量、CUDA 性能或量化 kernel。现有生成测试覆盖 greedy/cache，不代表组合 sample 路径已覆盖；量化教学组件和 RoPE 的混合精度接口也未获得完整运行认证。

Markdown 静态检查覆盖围栏、未标注片段的 Python 编译、数学定界符/括号及已知危险写法、本地文件链接，不解析全部 TeX 语法，也不验证标题锚点或 GitHub 在线渲染。KaTeX 检查覆盖提取到的数学片段；新增/修改公式仍需在发布目标页面检查。
