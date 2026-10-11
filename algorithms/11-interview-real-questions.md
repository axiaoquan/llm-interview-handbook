# 11 · 面试真题

整理实际面试中遇到的手撕题，不记录面试公司信息。

## 本章目录

- [一、LeetCode 原题](#一leetcode-原题)
- [二、LeetCode 变体](#二leetcode-变体)
- [三、非 LeetCode 算法题](#三非-leetcode-算法题)
- [四、机器学习 / 大模型手撕](#四机器学习--大模型手撕)

---

## 一、LeetCode 原题

## Q01 · 零钱兑换

> [LeetCode 322](https://leetcode.cn/problems/coin-change/) · 标签：完全背包 / 动态规划

给定硬币面额和目标金额，求凑出目标金额所需的最少硬币数，每种硬币可重复使用；无法凑出时返回 -1。

定义 dp[j] 为凑出金额 j 所需的最少硬币数：

$$
dp[j]=\min(dp[j],dp[j-c]+1)
$$

金额维度正序遍历。时间复杂度为 $O(|coins|\times amount)$，空间复杂度为 $O(amount)$。

## Q02 · 搜索二维矩阵

> [LeetCode 74](https://leetcode.cn/problems/search-a-2d-matrix/) · 标签：二分查找

每行升序，且下一行第一个元素大于上一行最后一个元素。把矩阵看成长度为 m*n 的一维有序数组，使用

$$
row=\lfloor mid/n\rfloor,\qquad col=mid\bmod n
$$

映射回二维坐标后进行二分。时间复杂度为 $O(\log(mn))$，空间复杂度为 $O(1)$。

## Q03 · 最大子数组和

> [LeetCode 53](https://leetcode.cn/problems/maximum-subarray/) · 标签：Kadane / 动态规划

维护以当前位置结尾的最大子数组和：

$$
current=\max(x,current+x)
$$

同时维护全局最大值。时间复杂度为 $O(n)$，空间复杂度为 $O(1)$。全是负数时不能把答案初始化为 0，子数组必须连续。

## Q04 · 最大为 N 的数字组合

> [LeetCode 902](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/) · 标签：数位 DP

给定数字集合 digits，每个数字可以重复使用，计算由这些数字构成且不超过 N 的正整数数量。

先统计长度小于 N 的数字，再从最高位比较长度相同的数字；选择小于当前位的数字后，后缀可以自由填充；一路相等时再判断 N 本身是否可构成。注意首位不能为零。

---

## 二、LeetCode 变体

## Q05 · 有效括号变体

> 对应 [LeetCode 20](https://leetcode.cn/problems/valid-parentheses/) · 标签：栈 / 字符串

给定括号字符串，判断括号是否匹配且嵌套顺序正确。基础版本包含圆括号、方括号和花括号；面试记录中说明是变体，若加入通配符、错误位置返回等要求，需要按现场版本修改。

~~~python
def is_valid_parentheses(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in s:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
        else:
            return False
    return not stack
~~~

时间复杂度为 $O(n)$，空间复杂度为 $O(n)$。

## Q06 · 中心向外的螺旋矩阵

给定一个 m 行 n 列矩阵，m 和 n 都是奇数。从矩阵中心开始，第一步向右，随后按“右、下、左、上”的顺时针方向向外扩展，返回访问到的所有元素序列。步长依次为右 1、下 1、左 2、上 2、右 3、下 3……矩阵可以是非正方形，越界位置跳过但方向和步长继续推进。

~~~python
def spiral_from_center(matrix):
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("rows and cols must be odd")
    row, col = rows // 2, cols // 2
    result = [matrix[row][col]]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction, step = 0, 1
    while len(result) < rows * cols:
        for _ in range(2):
            dr, dc = directions[direction % 4]
            for _ in range(step):
                row, col = row + dr, col + dc
                if 0 <= row < rows and 0 <= col < cols:
                    result.append(matrix[row][col])
                    if len(result) == rows * cols:
                        return result
            direction += 1
        step += 1
    return result
~~~

时间复杂度为 $O(mn)$，额外空间复杂度为 $O(mn)$。这不是 LeetCode 54 的原题。

## Q07 · 奇偶位置有序链表

给定单链表，奇数位置节点按升序排列，偶数位置节点按降序排列，将其重新组织为整体升序链表。思路是拆分、反转偶数链、合并两条升序链。这不是 LeetCode 328 的原题。

~~~python
class ListNode:
    def __init__(self, val=0, next_node=None):
        self.val, self.next = val, next_node


def reverse_list(head):
    previous, current = None, head
    while current:
        next_node = current.next
        current.next = previous
        previous, current = current, next_node
    return previous


def merge_sorted(first, second):
    dummy, tail = ListNode(), None
    tail = dummy
    while first and second:
        if first.val <= second.val:
            tail.next, first = first, first.next
        else:
            tail.next, second = second, second.next
        tail = tail.next
    tail.next = first if first else second
    return dummy.next


def sort_odd_even_positions(head):
    odd_dummy, even_dummy = ListNode(), ListNode()
    odd_tail, even_tail = odd_dummy, even_dummy
    node, position = head, 1
    while node:
        next_node = node.next
        node.next = None
        if position % 2:
            odd_tail.next, odd_tail = node, node
        else:
            even_tail.next, even_tail = node, node
        node, position = next_node, position + 1
    return merge_sorted(odd_dummy.next, reverse_list(even_dummy.next))
~~~

时间复杂度为 $O(n)$，额外空间复杂度为 $O(1)$。位置奇偶和节点值奇偶不是一回事。

## Q08 · 二叉树中和为目标值的所有路径

> 与 [LeetCode 437](https://leetcode.cn/problems/path-sum-iii/) 接近，但要求返回具体路径

给定二叉树和 target_sum，找出所有沿父节点到子节点方向连续向下的路径，使路径节点值之和等于 target_sum。路径可以从任意节点开始，也可以在任意节点结束，终点不要求是叶子节点；返回所有路径以及路径数量。

~~~python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def paths_with_sum(root, target_sum):
    result, current = [], []

    def dfs(node):
        if node is None:
            return
        current.append(node.val)
        total = 0
        for start in range(len(current) - 1, -1, -1):
            total += current[start]
            if total == target_sum:
                result.append(current[start:].copy())
        dfs(node.left)
        dfs(node.right)
        current.pop()

    dfs(root)
    return result
~~~

最坏时间复杂度为 $O(n^2)$，额外空间为 $O(h)$，不计结果数组。节点值可以为负数，不能用当前和超过目标值进行剪枝。

---

## 三、非 LeetCode 算法题

## Q09 · 使用随机梯度下降求解方程

使用随机梯度下降法求解 $x^2=5$。可将其转为最小化：

$$
f(x)=\frac12(x^2-5)^2,\qquad
\frac{\partial f}{\partial x}=2x(x^2-5)
$$

~~~python
def solve_x_squared_equals_five(
    initial_x=1.0, learning_rate=0.01,
    max_steps=10000, tolerance=1e-8
):
    x = float(initial_x)
    for _ in range(max_steps):
        error = x * x - 5.0
        if abs(error) <= tolerance:
            return x
        x -= learning_rate * 2.0 * x * error
    return x
~~~

时间复杂度为 $O(T)$，空间复杂度为 $O(1)$。初始值为 0 时梯度为 0；学习率过大可能发散；结果可能是正根或负根。

## Q10 · 均匀随机数转换

已知一个可以均匀生成 0 到 4 整数的生成器，只允许调用它，设计一个函数均匀生成 0 到 6 的整数。

两次调用得到 v=5a+b，v 在 0 到 24 上均匀分布。只保留 0 到 20，再对 7 取模；21 到 24 拒绝并重新采样。

~~~python
def rand_0_to_6(rand_0_to_4):
    while True:
        value = 5 * rand_0_to_4() + rand_0_to_4()
        if value < 21:
            return value % 7
~~~

接受概率为 $21/25$，期望调用原始生成器次数为 $50/21$，额外空间复杂度为 $O(1)$。直接取模会造成概率不均匀。

## Q11 · 快速排序过程中计算 Pairwise AUC

### 题目要求

给定样本的真实排序和模型预测结果。真实排序给出每个样本的真实 rank，rank 越小表示真实顺序越靠前；模型预测结果是分数，分数越大表示模型认为越靠前。

要求使用快速排序按照预测分数排序，并在 partition 过程中同步计算 Pairwise AUC，不能排序完成后再单独扫描。预测顺序正确计 1 分，错误计 0 分，预测分数相同计 0.5 分：

$$
\mathrm{AUC}
=
\frac{\text{正确排序对数}+0.5\times\text{预测平局对数}}
{\text{总样本对数}}
$$

### 思路和代码

使用三路 partition 将样本分为 high、equal、low。partition 时计算 high/equal、high/low、equal/low 的跨区间贡献，equal 内部的样本对计 0.5 分，再递归处理 high 和 low。真实 rank 的跨区间统计用 Fenwick 树完成。

~~~python
class Fenwick:
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def add(self, index, value):
        index += 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index):
        result = 0
        index += 1
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result


def count_ordered_pairs(before, after, rank_count):
    bit = Fenwick(rank_count)
    for _, rank in before:
        bit.add(rank, 1)
    score = 0.0
    for _, rank in after:
        less = bit.prefix_sum(rank - 1) if rank > 0 else 0
        equal = bit.prefix_sum(rank) - less
        score += less + 0.5 * equal
    return score


def pairwise_auc_during_quicksort(true_order, predicted_scores):
    if len(true_order) < 2:
        return 1.0
    if set(true_order) != set(predicted_scores):
        raise ValueError("IDs must match")

    true_rank = {item: index for index, item in enumerate(true_order)}
    records = [(predicted_scores[item], true_rank[item])
               for item in true_order]
    rank_count = len(records)

    def solve(items):
        if len(items) < 2:
            return 0.0
        pivot = items[len(items) // 2][0]
        high = [item for item in items if item[0] > pivot]
        equal = [item for item in items if item[0] == pivot]
        low = [item for item in items if item[0] < pivot]

        score = 0.5 * len(equal) * (len(equal) - 1) / 2
        score += count_ordered_pairs(high, equal, rank_count)
        score += count_ordered_pairs(high, low, rank_count)
        score += count_ordered_pairs(equal, low, rank_count)
        return score + solve(high) + solve(low)

    numerator = solve(records)
    denominator = len(records) * (len(records) - 1) / 2
    return numerator / denominator
~~~

平均时间复杂度为 $O(n\log^2 n)$；快速排序最坏可能为 $O(n^2)$。如果只有正负标签，可以用正负样本累计计数实现更简单的标准 AUC 版本。

## Q12 · KNN

给定带标签训练样本和待预测样本，使用 KNN 找到距离最近的 K 个样本，并通过多数投票预测类别。默认使用欧氏距离和分类任务。

~~~python
from collections import Counter


def knn_predict(train_x, train_y, query, k):
    if not train_x or len(train_x) != len(train_y):
        raise ValueError("training data must be non-empty and aligned")
    if not 1 <= k <= len(train_x):
        raise ValueError("invalid k")

    distances = []
    for features, label in zip(train_x, train_y):
        distance = sum((a - b) ** 2
                       for a, b in zip(features, query))
        distances.append((distance, label))
    distances.sort(key=lambda pair: pair[0])

    nearest = distances[:k]
    counts = Counter(label for _, label in nearest)
    best = max(counts.values())
    candidates = {label for label, count in counts.items()
                  if count == best}
    for _, label in nearest:
        if label in candidates:
            return label
    raise RuntimeError("unreachable")
~~~

距离计算为 $O(nd)$，全排序为 $O(n\log n)$，额外空间为 $O(n)$。特征通常需要标准化，平票规则必须明确。

---

## 四、机器学习 / 大模型手撕

## Q13 · 多头注意力机制

给定输入张量 $X\in\mathbb{R}^{B\times L\times d_{\mathrm{model}}}$，实现 Q/K/V 投影、拆分多个 head、缩放点积注意力、可选 causal mask、拼接各 head 和输出投影。

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_{\mathrm{head}}}}
\right)V
$$

完整 PyTorch 实现见 [LLM Attention 手撕](../llm-coding/01-attention.md)，本章只保留面试题和要点。时间复杂度为 $O(BL^2d_{\mathrm{model}})$，注意力分数矩阵空间复杂度为 $O(BhL^2)$。

## Q14 · 手写 GRPO Loss

给定同一 prompt 下的一组回答、每条回答的奖励、old/current/reference policy 的 log-prob 和 token mask，只写出 GRPO loss 公式，不要求实现代码。

组内相对优势：

$$
A_i=
\frac{
r_i-\mathrm{mean}(r_1,\ldots,r_G)
}{
\mathrm{std}(r_1,\ldots,r_G)+\epsilon
}
$$

重要性比率：

$$
\rho_{i,t}(\theta)
=
\exp\left(
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
-
\log\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})
\right)
$$

带 clipping 和 KL 正则的常见形式：

$$
\mathcal{L}_{\mathrm{GRPO}}
=
-\mathbb{E}_{i,t}
\left[
\min\left(
\rho_{i,t}A_i,
\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
\right]
+
\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
$$

只对有效生成 token 计算策略损失，padding 和不参与训练的上下文位置通过 mask 排除。old policy 用于 ratio 分母，reference policy 用于 KL 约束；advantage 通常需要 detach。GSPO 使用 sequence-level ratio，不能直接替换标准 GRPO 的 token-level ratio。

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
