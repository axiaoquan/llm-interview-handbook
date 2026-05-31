# 05 · StackQueue 栈与队列

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 用栈实现队列](#q01--用栈实现队列)
- [Q02 · 用队列实现栈](#q02--用队列实现栈)
- [Q03 · 有效的括号](#q03--有效的括号)
- [Q04 · 删除字符串中的所有相邻重复项](#q04--删除字符串中的所有相邻重复项)
- [Q05 · 逆波兰表达式求值](#q05--逆波兰表达式求值)
- [Q06 · 滑动窗口最大值](#q06--滑动窗口最大值)
- [Q07 · 前 K 个高频元素](#q07--前-k-个高频元素)

---

## 基础知识

- **互相实现**：双栈实现队列 / 单队列实现栈
- **括号匹配**：栈的经典应用
- **单调队列**：滑动窗口最大值
- **优先队列（堆）**：Top-K 问题

---

## Q01 · 用栈实现队列

> [LeetCode 232](https://leetcode.cn/problems/implement-queue-using-stacks/) · 难度：⭐⭐

### 🎯 思路

两个栈：

- `stack_in`：负责入队（push）
- `stack_out`：负责出队（pop / peek）

pop 时如果 `stack_out` 为空，就把 `stack_in` 全部倒过去，再 pop。

### 🛠 代码

```python
class MyQueue:
    def __init__(self):
        self.s_in, self.s_out = [], []

    def push(self, x):
        self.s_in.append(x)

    def pop(self):
        if not self.s_out:
            while self.s_in:
                self.s_out.append(self.s_in.pop())
        return self.s_out.pop()

    def peek(self):
        ans = self.pop()
        self.s_out.append(ans)
        return ans

    def empty(self):
        return not self.s_in and not self.s_out
```

---

## Q02 · 用队列实现栈

> [LeetCode 225](https://leetcode.cn/problems/implement-stack-using-queues/) · 难度：⭐⭐

### 🎯 思路（一个队列即可）

pop 时把队列**前 n-1 个**出队再入队，最后一个就是栈顶。

```python
from collections import deque

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)

    def pop(self):
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
        return self.q.popleft()

    def top(self):
        return self.q[-1]

    def empty(self):
        return not self.q
```

---

## Q03 · 有效的括号

> [LeetCode 20](https://leetcode.cn/problems/valid-parentheses/) · 难度：⭐⭐

### 🎯 思路

遇到左括号入栈；遇到右括号检查栈顶是否匹配，匹配则出栈，不匹配返回 False。最后栈为空才有效。

```python
class Solution:
    def isValid(self, s):
        pairs = {')': '(', ']': '[', '}': '{'}
        stack = []
        for c in s:
            if c in '([{':
                stack.append(c)
            else:
                if not stack or stack.pop() != pairs[c]:
                    return False
        return not stack
```

### 🪤 易错点

- 不能只看长度奇偶 → 必须用栈匹配
- 最后**栈必须空**（不然有未闭合的左括号）

### 🔗 同类题

- [删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

---

## Q04 · 删除字符串中的所有相邻重复项

> [LeetCode 1047](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/) · 难度：⭐⭐

### 🎯 思路

栈：当前字符 == 栈顶 → pop；否则 push。

```python
class Solution:
    def removeDuplicates(self, s):
        stack = []
        for c in s:
            if stack and stack[-1] == c:
                stack.pop()
            else:
                stack.append(c)
        return ''.join(stack)
```

---

## Q05 · 逆波兰表达式求值

> [LeetCode 150](https://leetcode.cn/problems/evaluate-reverse-polish-notation/) · 难度：⭐⭐

### 🎯 思路

逆波兰（后缀）：运算符在后。
遍历 token：数字入栈；运算符弹出两个数字算完再入栈。

```python
class Solution:
    def evalRPN(self, tokens):
        stack = []
        for t in tokens:
            if t in '+-*/':
                b = stack.pop()
                a = stack.pop()
                if t == '+': stack.append(a + b)
                elif t == '-': stack.append(a - b)
                elif t == '*': stack.append(a * b)
                else: stack.append(int(a / b))   # 注意 Python 负数除法
            else:
                stack.append(int(t))
        return stack[0]
```

### 🪤 易错点

- 弹出顺序：**先弹的是 b（右操作数），后弹的是 a**
- 整数除法：`int(a/b)` **而非 `a // b`**（后者向下取整，对负数结果不同）

---

## Q06 · 滑动窗口最大值

> [LeetCode 239](https://leetcode.cn/problems/sliding-window-maximum/) · 难度：⭐⭐⭐ · 标签：单调队列

### 🎯 思路

**单调递减队列**：只保留窗口中可能成为最大值的元素。

- 新元素入队前，从队尾弹出所有比它小的（它们不可能再当最大）
- 队首元素出窗口（i - k 等于队首元素索引）时弹出
- 队首始终是当前窗口最大

### 🛠 代码

```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums, k):
        q = deque()    # 存的是下标
        ans = []
        for i, x in enumerate(nums):
            while q and nums[q[-1]] < x:
                q.pop()
            q.append(i)
            if q[0] <= i - k:        # 出窗口
                q.popleft()
            if i >= k - 1:
                ans.append(nums[q[0]])
        return ans
```

### 📊 复杂度

- 时间：**O(n)**（每个元素最多进出一次）
- 空间：O(k)

### 🪤 易错点

- 队列存的是**下标**而非值（方便判断是否出窗口）
- 单调递减：所以入队前要弹出所有更小的

---

## Q07 · 前 K 个高频元素

> [LeetCode 347](https://leetcode.cn/problems/top-k-frequent-elements/) · 难度：⭐⭐⭐ · 标签：堆

### 🎯 思路

1. 用 map 统计频率
2. 小顶堆维护 Top-K（超过 k 个就 pop 堆顶）
3. 取出堆中所有元素

### 🛠 代码

```python
import heapq
from collections import Counter

class Solution:
    def topKFrequent(self, nums, k):
        cnt = Counter(nums)
        heap = []
        for x, freq in cnt.items():
            heapq.heappush(heap, (freq, x))
            if len(heap) > k:
                heapq.heappop(heap)
        return [x for _, x in heap]
```

### 🪤 为什么用小顶堆？

求 Top-K **大**值用**小顶堆**——堆顶是当前 K 个里最小的，新元素比它大才有资格进。

### 📊 复杂度

- 时间：**O(n log k)**
- 空间：O(n + k)

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
