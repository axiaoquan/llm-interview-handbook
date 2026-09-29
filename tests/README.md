# 本地验证

回归测试直接提取 `llm-coding/*.md` 中的 import、函数和类定义，不执行示例调用处的占位变量；不维护另一套实现副本。

## 运行

需要 Python 3.9+ 和 PyTorch（本轮验证版本为 2.8.0，CPU）。建议使用隔离虚拟环境：

```bash
python -m pip install torch==2.8.0
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

静态检查仅使用标准库，无需 PyTorch。测试失败时检查正文中的实现，不能只修改测试来迎合结果。

可选：在仓库外的临时 Node 环境安装 `katex@0.16.22`，将 `NODE_PATH` 指向其 `node_modules`，运行 `node scripts/check_math.cjs`。它通过 KaTeX 解析提取到的公式，能发现仅靠括号检查漏掉的语法错误；这仍不等于 GitHub 的 Markdown/MathJax 页面已视觉验证。

## 覆盖范围

- LoRA：四种初始化、首步梯度/有效更新、整个基座冻结、低精度参数、eval merge 等价与训练模式拒绝。
- Norm：LN/RMSNorm/二维 BN 的输出及输入/仿射参数梯度对齐；BN running statistics；常量、单特征、BF16 归约。
- Attention：布尔与加性 mask、全遮挡行、eval dropout、官方输出/梯度对齐、full/single/chunked cache 等价、未来泄漏与 padding。
- Loss：稳定 CE、LM shift、全 ignore、label smoothing 与官方定义对齐、GRPO 聚合口径/固定 old 与 ref/同奖励组。
- BPE：完整 token 边界、重叠 pair、频次累加、基础与中间 token 覆盖、两种编码流程。
- 生成与其他：逐样本 EOS、零生成长度、MoE 权重/梯度等价、dispatch 参考值、top-1 router 梯度、共享 embedding。

默认比较使用 `torch.testing.assert_close`，FP64 原理测试与 BF16 近似测试分开；测试使用固定随机种子和小张量，无需下载模型。

## 验证边界

这不是全仓所有示例的运行认证：测试不覆盖全部通用算法、完整 SFT/RL 训练、真实模型生成质量、CUDA 性能或量化 kernel。Markdown 静态检查覆盖围栏、未标注片段的 Python 编译、数学括号及已知危险写法、本地文件链接，不解析全部 TeX 语法，也不验证标题锚点或 GitHub 在线渲染。新增/修改公式仍需在发布目标页面检查。
