# 02 · LinkedList 链表

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 移除链表元素](#q01--移除链表元素)
- [Q02 · 设计链表](#q02--设计链表)
- [Q03 · 反转链表](#q03--反转链表)
- [Q04 · 两两交换链表中的结点](#q04--两两交换链表中的结点)
- [Q05 · 删除链表的倒数第 N 个结点](#q05--删除链表的倒数第-n-个结点)
- [Q06 · 链表相交](#q06--链表相交)
- [Q07 · 环形链表 II](#q07--环形链表-ii)

---

## 基础知识

### 链表类型

| 类型 | 特点 |
|---|---|
| 单链表 | 只能向前查询 |
| 双链表 | 可前可后 |
| 循环链表 | 首尾相连 |

### 储存方式

链表在内存中**不连续**，靠指针串联。

### 链表定义

```python
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
```

### 三个核心技巧

- **虚拟头结点**：统一头结点和其他结点的处理
- **双指针**：快慢指针 / 距离差指针
- **递归**：链表很多操作天然递归

---

## Q01 · 移除链表元素

> [LeetCode 203](https://leetcode.cn/problems/remove-linked-list-elements/) · 难度：⭐⭐

### 🎯 思路

#### 解法一：虚拟头结点

头结点和其他结点的删除逻辑不同，**加一个虚拟头**统一处理：

```
dummy → head → ... → tail
        ↑ 删除 head 也变得简单
```

#### 解法二：递归

- 空 → 返回空
- 头结点值 == val → 直接返回 `removeElements(head.next, val)`
- 头结点值 != val → `head.next = removeElements(head.next, val)`，返回 head

### 🛠 代码

```python
# 虚拟头结点
class Solution:
    def removeElements(self, head, val):
        dummy = ListNode(0, head)
        cur = dummy
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next       # 跳过
            else:
                cur = cur.next
        return dummy.next
```

```python
# 递归
class Solution:
    def removeElements(self, head, val):
        if head is None:
            return None
        if head.val == val:
            return self.removeElements(head.next, val)
        head.next = self.removeElements(head.next, val)
        return head
```

### 📊 复杂度

- 时间：O(n)
- 空间：迭代 O(1) / 递归 O(n)

### 🪤 易错点

- 不能直接 `cur = cur.next`，需要判断 cur.next.val 才能决定是否跳过
- 递归终止条件别漏 `head is None`

---

## Q02 · 设计链表

> [LeetCode 707](https://leetcode.cn/problems/design-linked-list/) · 难度：⭐⭐⭐

### 🎯 思路

**双向链表**应同时维护：
- `head`（头结点）
- `tail`（尾结点）
- `size`（结点总数）

确保增删查 O(1)（边界）/ O(n)（中间）。

### 📖 关键操作要点

| 操作 | 注意点 |
|---|---|
| 初始化空链表 | head=None, tail=None, size=0 |
| 头部插入 | 空链表时同时更新 tail；非空在 head 前插入 |
| 尾部插入 | 空链表时同时更新 head；非空在 tail 后插入 |
| 指定位置插入 | 边界用头/尾函数（**记得 return 防止再走中间逻辑**） |
| 删除头结点 | 判断新头是否存在，否则 tail = None |
| 删除尾结点 | 判断新尾是否存在，否则 head = None |

### 🪤 总结

1. **前后指针要同时更新**
2. 所有操作考虑**空链表 / 单节点**特殊情况
3. **记得 return**，防止逻辑错乱

---

## Q03 · 反转链表

> [LeetCode 206](https://leetcode.cn/problems/reverse-linked-list/) · 难度：⭐⭐ · 标签：双指针 / 递归

### 🎯 思路

**双指针法**：cur 是当前结点，pre 是前一个（初始 None）。

每步：
1. 暂存 cur.next
2. cur.next = pre（翻转）
3. pre = cur, cur = tmp（移动指针）

最终返回 pre。

### 🛠 代码

```python
# 双指针
class Solution:
    def reverseList(self, head):
        pre, cur = None, head
        while cur:
            tmp = cur.next     # 先暂存
            cur.next = pre     # 翻转
            pre, cur = cur, tmp
        return pre
```

```python
# 递归（同样逻辑）
class Solution:
    def reverseList(self, head):
        return self.reverse(head, None)

    def reverse(self, cur, pre):
        if cur is None:
            return pre
        tmp = cur.next
        cur.next = pre
        return self.reverse(tmp, cur)
```

### 📊 复杂度

- 时间：O(n)
- 空间：迭代 O(1) / 递归 O(n)

### 🪤 易错点

- **必须先暂存 cur.next**，否则修改 cur.next 后丢失后续
- 返回的是 **pre**（而不是 cur，cur 最后是 None）

---

## Q04 · 两两交换链表中的结点

> [LeetCode 24](https://leetcode.cn/problems/swap-nodes-in-pairs/) · 难度：⭐⭐ · 标签：虚拟头

### 🎯 思路

用**虚拟头结点**简化，循环条件 `cur.next` 和 `cur.next.next` 都不为空。

每次需要**提前保存结点 1 和结点 3**：

```
cur → 1 → 2 → 3 → ...     变成     cur → 2 → 1 → 3 → ...
```

### 🛠 代码

```python
class Solution:
    def swapPairs(self, head):
        dummy = ListNode(0, head)
        cur = dummy
        while cur.next and cur.next.next:
            n1 = cur.next
            n2 = cur.next.next
            n3 = n2.next        # 提前存

            cur.next = n2
            n2.next = n1
            n1.next = n3

            cur = n1            # 移动到 n1（已经是新 pair 的尾）
        return dummy.next
```

### 📊 复杂度

- 时间：O(n)
- 空间：O(1)

---

## Q05 · 删除链表的倒数第 N 个结点

> [LeetCode 19](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) · 难度：⭐⭐ · 标签：快慢指针

### 🎯 思路

**快慢指针**构造一个长度为 n 的"滑动窗口"：

1. fast 先走 n 步
2. fast、slow 同时走，直到 fast 指向链表末尾
3. 此时 slow 指向**待删除结点的前一个**（用虚拟头方便统一）

```python
class Solution:
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(0, head)
        fast = slow = dummy
        for _ in range(n + 1):    # n+1 让 slow 停在前一个
            fast = fast.next
        while fast:
            fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return dummy.next
```

### 🪤 易错点

- 用虚拟头时 fast 走 n+1 步（让 slow 停在待删除结点的**前一个**）
- 不用虚拟头时 fast 先走 n 步，slow 停在待删除结点本身（不好处理头结点删除）

---

## Q06 · 链表相交

> [面试题 02.07](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/) · 难度：⭐⭐

### 🎯 思路

**长度差 + 双指针**：

1. 求两个链表的长度，计算差 n
2. 长链表先走 n 步
3. 两指针同时走，相等就是交点

### 🛠 代码

```python
class Solution:
    def getIntersectionNode(self, headA, headB):
        # 求长度
        def length(h):
            n = 0
            while h: n += 1; h = h.next
            return n

        la, lb = length(headA), length(headB)
        if la < lb:
            headA, headB = headB, headA
            la, lb = lb, la

        # 长链表先走差距步
        for _ in range(la - lb):
            headA = headA.next

        while headA and headA != headB:
            headA, headB = headA.next, headB.next
        return headA
```

### 📖 优雅解法（双指针拼接）

```python
def getIntersectionNode(headA, headB):
    a, b = headA, headB
    while a is not b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a
```

走完自己的链就走对方的链——两人最终相遇点要么是交点，要么是 None。

---

## Q07 · 环形链表 II

> [LeetCode 142](https://leetcode.cn/problems/linked-list-cycle-ii/) · 难度：⭐⭐⭐ · 标签：快慢指针 / 数学推导

### 🎯 拆成两个问题

#### 1. 判断是否有环

快慢指针：fast 速度 2，slow 速度 1。
若有环，**两者必相遇**（快指针在环里相对慢指针速度 1 追，不会跳过）。

#### 2. 求环的入口

设：
- 头到环入口距离 = $x$
- 入口到相遇点 = $y$
- 相遇点到入口 = $z$

相遇时：

- slow 走过 $x + y$
- fast 走过 $x + y + n(y + z)$，n 是 fast 在环里的圈数（必 ≥ 1）

由 $2 \cdot \text{slow} = \text{fast}$：
$$
2(x + y) = x + y + n(y+z) \implies x = (n-1)(y+z) + z
$$

**关键**：当 $n = 1$ 时 $x = z$。意味着：
> 假设一个指针从头出发，另一个从相遇点出发，**速度都为 1**，必定在**环的入口**相遇。

### 🛠 代码

```python
class Solution:
    def detectCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:                  # 找到相遇点
                p = head
                while p is not slow:
                    p, slow = p.next, slow.next
                return p
        return None
```

### 🪤 为什么 slow 不会在环里走超过一圈才相遇？

- slow 进环时 fast 已经在环里
- 假设 slow 走完一圈，fast 走了两圈 → 必定追上或刚好追上 slow
- 所以 slow 进环 → 必在一圈内被追上

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
