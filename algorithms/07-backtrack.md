# 07 · Backtrack 回溯

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 组合](#q01--组合)
- [Q02 · 组合总和系列](#q02--组合总和系列)
- [Q03 · 分割回文串](#q03--分割回文串)
- [Q04 · 子集](#q04--子集)
- [Q05 · 排列](#q05--排列)
- [Q06 · N 皇后](#q06--n-皇后)

---

## 基础知识

**回溯 = 穷举**。把搜索空间画成树，DFS 遍历，过程中可剪枝。

### 适用场景

| 类型 | 例子 |
|---|---|
| **组合** | N 个数选 k 个 |
| **切割** | 字符串切分 |
| **子集** | 所有子集 |
| **排列** | 全排列 |
| **棋盘** | N 皇后 / 数独 |

### 模板

```python
def backtrack(参数):
    if 终止条件:
        ans.append(path[:])     # 必须深拷贝！
        return
    for 选择 in 当前层选择列表:
        path.append(选择)        # 处理
        backtrack(...)           # 递归
        path.pop()               # 回溯
```

### 三个关键

1. **path[:]** 深拷贝，否则后续修改会污染答案
2. **for 起点 startIndex** 控制不重复
3. **used[]** 数组在排列中用来去重

---

## Q01 · 组合

> [LeetCode 77](https://leetcode.cn/problems/combinations/) · 难度：⭐⭐

### 🎯 思路

从 [1, n] 选 k 个组合。用 `start` 控制不重复。

```python
class Solution:
    def combine(self, n, k):
        ans, path = [], []
        def backtrack(start):
            if len(path) == k:
                ans.append(path[:])
                return
            # 剪枝：剩余元素不够时提前结束
            for i in range(start, n - (k - len(path)) + 2):
                path.append(i)
                backtrack(i + 1)
                path.pop()
        backtrack(1)
        return ans
```

### 🪤 剪枝点

- 剩余元素 < 还需要的个数 → 跳过
- `n - (k - len(path)) + 1` 是最大可行起点

---

## Q02 · 组合总和系列

### 📖 组合总和（可重复用）

> [LeetCode 39](https://leetcode.cn/problems/combination-sum/)

每个元素**可重复使用** → 递归时下一层从 `i` 开始（不是 `i+1`）。

```python
def combinationSum(candidates, target):
    candidates.sort()
    ans, path = [], []
    def backtrack(start, remain):
        if remain == 0:
            ans.append(path[:]); return
        for i in range(start, len(candidates)):
            if candidates[i] > remain: break       # 剪枝
            path.append(candidates[i])
            backtrack(i, remain - candidates[i])   # i 不是 i+1
            path.pop()
    backtrack(0, target)
    return ans
```

### 📖 组合总和 II（每个用一次 + 去重）

> [LeetCode 40](https://leetcode.cn/problems/combination-sum-ii/)

**树层去重**：同一层相邻相同元素只能用第一个。

```python
if i > start and candidates[i] == candidates[i-1]:
    continue
```

---

## Q03 · 分割回文串

> [LeetCode 131](https://leetcode.cn/problems/palindrome-partitioning/) · 难度：⭐⭐⭐

### 🎯 思路

切割问题 ≈ 组合问题。`start` 表示下一段的起点。

```python
class Solution:
    def partition(self, s):
        ans, path = [], []
        def is_palin(t): return t == t[::-1]
        def backtrack(start):
            if start == len(s):
                ans.append(path[:]); return
            for i in range(start, len(s)):
                seg = s[start:i+1]
                if is_palin(seg):
                    path.append(seg)
                    backtrack(i + 1)
                    path.pop()
        backtrack(0)
        return ans
```

---

## Q04 · 子集

> [LeetCode 78](https://leetcode.cn/problems/subsets/) · 难度：⭐⭐

### 🎯 思路

子集问题 = **每个节点都收集**（不只是叶子）。

```python
class Solution:
    def subsets(self, nums):
        ans, path = [], []
        def backtrack(start):
            ans.append(path[:])         # 每个节点都收集
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return ans
```

---

## Q05 · 排列

> [LeetCode 46](https://leetcode.cn/problems/permutations/) · 难度：⭐⭐

### 🎯 思路

排列**有顺序** → 不能用 startIndex，每次都从头遍历，用 `used[]` 标记已用。

```python
class Solution:
    def permute(self, nums):
        ans, path = [], []
        used = [False] * len(nums)
        def backtrack():
            if len(path) == len(nums):
                ans.append(path[:]); return
            for i in range(len(nums)):
                if used[i]: continue
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False
        backtrack()
        return ans
```

### 📖 排列 II（含重复）

`used[i-1] == False` 表示同一层先前选过的相同元素已回溯 → **跳过**。

```python
if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
    continue
```

---

## Q06 · N 皇后

> [LeetCode 51](https://leetcode.cn/problems/n-queens/) · 难度：⭐⭐⭐⭐

### 🎯 思路

按行回溯，对每行尝试每一列，检查不同行/列/斜线冲突。

```python
class Solution:
    def solveNQueens(self, n):
        board = [['.'] * n for _ in range(n)]
        ans = []

        def valid(row, col):
            # 同列
            for i in range(row):
                if board[i][col] == 'Q': return False
            # 左上对角
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q': return False
                i -= 1; j -= 1
            # 右上对角
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == 'Q': return False
                i -= 1; j += 1
            return True

        def backtrack(row):
            if row == n:
                ans.append([''.join(r) for r in board]); return
            for col in range(n):
                if valid(row, col):
                    board[row][col] = 'Q'
                    backtrack(row + 1)
                    board[row][col] = '.'
        backtrack(0)
        return ans
```

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
