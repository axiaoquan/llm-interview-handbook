# 06 · BinaryTree 二叉树

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 二叉树的递归遍历](#q01--二叉树的递归遍历)
- [Q02 · 二叉树的迭代遍历](#q02--二叉树的迭代遍历)
- [Q03 · 二叉树的统一迭代法](#q03--二叉树的统一迭代法)
- [Q04 · 二叉树的层序遍历](#q04--二叉树的层序遍历)
- [Q05 · 翻转二叉树](#q05--翻转二叉树)
- [Q06 · 对称二叉树](#q06--对称二叉树)
- [Q07 · 二叉树的最大深度](#q07--二叉树的最大深度)
- [Q08 · 二叉树的最小深度](#q08--二叉树的最小深度)
- [Q09 · 完全二叉树的节点个数](#q09--完全二叉树的节点个数)
- [Q10 · 平衡二叉树](#q10--平衡二叉树)
- [Q11 · 二叉树的所有路径](#q11--二叉树的所有路径)
- [Q12 · 二叉树的左叶子之和](#q12--二叉树的左叶子之和)
- [Q13 · 找树左下角的值](#q13--找树左下角的值)
- [Q14 · 路径总和](#q14--路径总和)

---

## 基础知识

### 二叉树类型

- **满二叉树**：除叶子外，每个结点都有两个孩子，叶子在同一层
- **完全二叉树**：除最底层外，其他层填满；底层结点集中在左侧
- **二叉搜索树（BST）**：左 < 根 < 右
- **平衡二叉搜索树（AVL）**：左右子树高度差 ≤ 1

### 储存方式

- **链式储存**（指针）
- **顺序储存**（数组）：父结点 i → 左孩子 2i+1，右孩子 2i+2

### 遍历方式

| 类型 | 遍历 | 中节点位置 |
|---|---|---|
| **DFS** | 前序 | 中左右 |
| | 中序 | 左中右 |
| | 后序 | 左右中 |
| **BFS** | 层序 | 一层一层 |

### 深度 vs 高度

- **深度**：根到该节点最长路径上的节点数
- **高度**：该节点到叶子最长路径上的节点数

---

## Q01 · 二叉树的递归遍历

> 递归三要素：参数 + 终止条件 + 单层逻辑

```python
# 前序：中左右
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# 中序：左中右
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# 后序：左右中
def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

---

## Q02 · 二叉树的迭代遍历

### 📖 前序（中左右）

栈中**先压右后压左**（栈是 LIFO）：

```python
def preorder(root):
    if not root: return []
    stack, ans = [root], []
    while stack:
        node = stack.pop()
        ans.append(node.val)
        if node.right: stack.append(node.right)
        if node.left:  stack.append(node.left)
    return ans
```

### 📖 后序（左右中）

前序代码的左右结点压入顺序互换 → 得到「中右左」，最后**反转**结果即为后序。

```python
def postorder(root):
    if not root: return []
    stack, ans = [root], []
    while stack:
        node = stack.pop()
        ans.append(node.val)
        if node.left: stack.append(node.left)
        if node.right: stack.append(node.right)
    return ans[::-1]
```

### 📖 中序（左中右）

中序**访问和处理不同步**，需要辅助指针 `cur`：

```python
def inorder(root):
    stack, ans = [], []
    cur = root
    while cur or stack:
        if cur:                  # 一直往左
            stack.append(cur)
            cur = cur.left
        else:                    # 到叶子，回到栈顶处理
            cur = stack.pop()
            ans.append(cur.val)
            cur = cur.right
    return ans
```

---

## Q03 · 二叉树的统一迭代法

### 🎯 核心思想

**用 `None` 标记**待处理结点：

- 访问结点时按"右-中(标None)-左"压栈（前序为例）
- 弹出 None 标记时，处理下一个栈顶结点

```python
# 中序为例（其他遍历改顺序即可）
def inorder(root):
    if not root: return []
    stack, ans = [root], []
    while stack:
        node = stack.pop()
        if node is not None:
            if node.right: stack.append(node.right)
            stack.append(node)
            stack.append(None)            # 标记
            if node.left: stack.append(node.left)
        else:
            ans.append(stack.pop().val)
    return ans
```

---

## Q04 · 二叉树的层序遍历

> [LeetCode 102](https://leetcode.cn/problems/binary-tree-level-order-traversal/) · 难度：⭐⭐

### 🛠 代码

```python
from collections import deque

class Solution:
    def levelOrder(self, root):
        if not root: return []
        q = deque([root])
        ans = []
        while q:
            level = []
            for _ in range(len(q)):       # 关键：先记录当前层大小
                node = q.popleft()
                level.append(node.val)
                if node.left:  q.append(node.left)
                if node.right: q.append(node.right)
            ans.append(level)
        return ans
```

### 🔗 同类题

- 二叉树的层序遍历 II（自底向上）
- 二叉树的右视图（每层最后一个）
- 二叉树的层平均值
- N 叉树的层序遍历
- 在每个树行中找最大值
- 填充每个节点的下一个右侧节点指针

---

## Q05 · 翻转二叉树

> [LeetCode 226](https://leetcode.cn/problems/invert-binary-tree/) · 难度：⭐

### 🛠 代码

```python
class Solution:
    def invertTree(self, root):
        if not root: return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

### 🪤 易错点

- ⚠️ **不要用中序遍历**！中序会让有的子树被翻转两次。
- 前序、后序、层序都可以。

---

## Q06 · 对称二叉树

> [LeetCode 101](https://leetcode.cn/problems/symmetric-tree/) · 难度：⭐⭐

### 🎯 思路

**后序遍历**——必须先知道左右子树是否对称，才能判断父节点。需要**同步比较两棵子树**。

### 🛠 代码

```python
class Solution:
    def isSymmetric(self, root):
        def compare(L, R):
            if L is None and R is None: return True
            if L is None or R is None:  return False
            if L.val != R.val: return False
            return compare(L.left, R.right) and compare(L.right, R.left)

        return root is None or compare(root.left, root.right)
```

---

## Q07 · 二叉树的最大深度

> [LeetCode 104](https://leetcode.cn/problems/maximum-depth-of-binary-tree/) · 难度：⭐

```python
class Solution:
    def maxDepth(self, root):
        if not root: return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

后序遍历：先知道左右深度，父节点深度 = max + 1。

---

## Q08 · 二叉树的最小深度

> [LeetCode 111](https://leetcode.cn/problems/minimum-depth-of-binary-tree/) · 难度：⭐⭐

### 🪤 易错点

**叶子结点定义**：左右孩子**都为空**才算叶子。

如果左孩子为空但右孩子不空，**不能取 min(0, right)**，只能取右子树深度。

```python
class Solution:
    def minDepth(self, root):
        if not root: return 0
        if not root.left and not root.right: return 1
        if not root.left:  return 1 + self.minDepth(root.right)
        if not root.right: return 1 + self.minDepth(root.left)
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
```

---

## Q09 · 完全二叉树的节点个数

> [LeetCode 222](https://leetcode.cn/problems/count-complete-tree-nodes/) · 难度：⭐⭐⭐

### 🎯 思路（剪枝）

普通方法 O(n)。利用完全二叉树特性可做到 **O(log² n)**。

判断子树是否是**满二叉树**（左深 == 右深），是 → 直接 `2^h - 1`；否则递归。

```python
class Solution:
    def countNodes(self, root):
        if not root: return 0
        ld, rd = 0, 0
        l, r = root.left, root.right
        while l: l = l.left; ld += 1
        while r: r = r.right; rd += 1
        if ld == rd:
            return (1 << (ld + 1)) - 1     # 满二叉树
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

---

## Q10 · 平衡二叉树

> [LeetCode 110](https://leetcode.cn/problems/balanced-binary-tree/) · 难度：⭐⭐

### 🎯 思路

后序求高度，**遇到不平衡就提前返回 -1**（剪枝）。

```python
class Solution:
    def isBalanced(self, root):
        def height(node):
            if not node: return 0
            lh = height(node.left)
            if lh == -1: return -1
            rh = height(node.right)
            if rh == -1: return -1
            if abs(lh - rh) > 1: return -1
            return 1 + max(lh, rh)
        return height(root) != -1
```

---

## Q11 · 二叉树的所有路径

> [LeetCode 257](https://leetcode.cn/problems/binary-tree-paths/) · 难度：⭐⭐ · 标签：回溯

### 🎯 思路

前序遍历 + 回溯。把"当前路径"作为参数往下传，叶子节点保存结果。

```python
class Solution:
    def binaryTreePaths(self, root):
        def dfs(node, path):
            path.append(str(node.val))
            if not node.left and not node.right:
                ans.append('->'.join(path))
            else:
                if node.left:  dfs(node.left, path)
                if node.right: dfs(node.right, path)
            path.pop()                # 回溯

        ans = []
        if root: dfs(root, [])
        return ans
```

---

## Q12 · 二叉树的左叶子之和

> [LeetCode 404](https://leetcode.cn/problems/sum-of-left-leaves/) · 难度：⭐⭐

### 🎯 思路

后序遍历。注意**左叶子的判断必须在父节点处判断**：

```python
class Solution:
    def sumOfLeftLeaves(self, root):
        if not root: return 0
        if not root.left and not root.right: return 0
        left = 0
        if root.left and not root.left.left and not root.left.right:
            left = root.left.val
        else:
            left = self.sumOfLeftLeaves(root.left)
        right = self.sumOfLeftLeaves(root.right)
        return left + right
```

---

## Q13 · 找树左下角的值

> [LeetCode 513](https://leetcode.cn/problems/find-bottom-left-tree-value/) · 难度：⭐⭐

### 🎯 思路

**层序遍历**，最后一层第一个元素即答案。

```python
from collections import deque

class Solution:
    def findBottomLeftValue(self, root):
        q = deque([root])
        while q:
            ans = q[0].val
            for _ in range(len(q)):
                node = q.popleft()
                if node.left:  q.append(node.left)
                if node.right: q.append(node.right)
        return ans
```

---

## Q14 · 路径总和

> [LeetCode 112](https://leetcode.cn/problems/path-sum/) · 难度：⭐⭐

```python
class Solution:
    def hasPathSum(self, root, target):
        if not root: return False
        if not root.left and not root.right:
            return root.val == target
        return self.hasPathSum(root.left, target - root.val) or \
               self.hasPathSum(root.right, target - root.val)
```

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
