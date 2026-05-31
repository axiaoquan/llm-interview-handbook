# 03 · HashTable 哈希表

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 有效的字母异位词](#q01--有效的字母异位词)
- [Q02 · 查找常用字符](#q02--查找常用字符)
- [Q03 · 两个数组的交集](#q03--两个数组的交集)
- [Q04 · 快乐数](#q04--快乐数)
- [Q05 · 两数之和](#q05--两数之和)
- [Q06 · 四数相加 II](#q06--四数相加-ii)
- [Q07 · 三数之和](#q07--三数之和)
- [Q08 · 四数之和](#q08--四数之和)

---

## 基础知识

### 哈希表是什么

哈希表用**关键码作为数组下标**直接访问元素。最常用场景：**快速判断元素是否在集合中**。

### 哈希函数

通过 hashCode 把任意类型转化为数组下标：

```
"小李" → hashCode → 1
"小王" → hashCode → 1     ← 哈希碰撞
```

### 哈希碰撞处理

| 方法 | 思想 |
|---|---|
| **拉链法** | 冲突的元素存在链表中 |
| **线性探测法** | 顺序找下一个空位（要求 tablesize > datasize） |

### 三种哈希结构选型

| 结构 | 适用 | 缺点 |
|---|---|---|
| **数组** | 数值范围有限（如 26 字母） | 范围大时浪费 |
| **set** | 只关心是否存在 | 不能存额外信息 |
| **map** | 需要 key→value 关联 | 实现稍重 |

> **核心思路**：哈希法是**空间换时间**——用额外数组 / set / map 存放数据，换取快速查找。

---

## Q01 · 有效的字母异位词

> [LeetCode 242](https://leetcode.cn/problems/valid-anagram/) · 难度：⭐⭐

### 🎯 思路

字符是 26 个小写字母 → 用**长度 26 的数组**作为哈希表。

s 中字符 +1，t 中字符 -1，最后数组**全为 0** 即 anagram。

### 🛠 代码

```python
class Solution:
    def isAnagram(self, s, t):
        if len(s) != len(t):
            return False
        record = [0] * 26
        for c in s:
            record[ord(c) - ord('a')] += 1
        for c in t:
            record[ord(c) - ord('a')] -= 1
        return all(x == 0 for x in record)
```

### 📊 复杂度

- 时间：**O(n)**
- 空间：O(1)（常数 26）

### 🔗 同类题

- [赎金信](https://leetcode.cn/problems/ransom-note/)

---

## Q02 · 查找常用字符

> [LeetCode 1002](https://leetcode.cn/problems/find-common-characters/) · 难度：⭐⭐

### 🎯 思路

用第一个字符串初始化哈希表，对后续每个字符串计算其哈希表，**与现有哈希表逐位取最小**。

### 🛠 代码

```python
class Solution:
    def commonChars(self, words):
        common = [float('inf')] * 26
        for w in words:
            cnt = [0] * 26
            for c in w:
                cnt[ord(c) - ord('a')] += 1
            for i in range(26):
                common[i] = min(common[i], cnt[i])
        ans = []
        for i in range(26):
            ans += [chr(i + ord('a'))] * common[i]
        return ans
```

---

## Q03 · 两个数组的交集

> [LeetCode 349](https://leetcode.cn/problems/intersection-of-two-arrays/) · 难度：⭐⭐

### 🎯 思路

数值有限范围（题目限定 0~1000）→ 数组哈希。

```python
class Solution:
    def intersection(self, nums1, nums2):
        h1, h2 = [0] * 1001, [0] * 1001
        for x in nums1: h1[x] = 1
        for x in nums2: h2[x] = 1
        return [i for i in range(1001) if h1[i] and h2[i]]
```

如果数值无限制，用 `set`：

```python
return list(set(nums1) & set(nums2))
```

### 🔗 同类题

- [快乐数](https://leetcode.cn/problems/happy-number/)（用 set 判断是否进入循环）

---

## Q04 · 快乐数

> [LeetCode 202](https://leetcode.cn/problems/happy-number/) · 难度：⭐⭐

### 🎯 思路

求各位平方和，若进入循环就不是 happy number。**用 set 检测重复数字**。

```python
class Solution:
    def isHappy(self, n):
        def squares_sum(x):
            s = 0
            while x:
                s += (x % 10) ** 2
                x //= 10
            return s

        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = squares_sum(n)
        return n == 1
```

---

## Q05 · 两数之和

> [LeetCode 1](https://leetcode.cn/problems/two-sum/) · 难度：⭐⭐ · 标签：哈希表

### 🎯 思路

遍历到某元素时，需要查询**之前所有元素**是否有 `target - x` → **哈希表 O(n)**。

哈希表结构：`{value: index}`（既要值又要下标 → **map**）。

### 🛠 代码

```python
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, x in enumerate(nums):
            if target - x in seen:
                return [seen[target - x], i]
            seen[x] = i
```

### 🪤 易错点

- **先查后存**，否则同一元素自己和自己相加可能算一对
- 题目保证一定有解 → 不用考虑 not found

---

## Q06 · 四数相加 II

> [LeetCode 454](https://leetcode.cn/problems/4sum-ii/) · 难度：⭐⭐⭐

### 🎯 思路

把 4 个数组拆成 **(A, B) + (C, D)** 两组：

1. 遍历 A、B，统计 `a+b` 的出现次数 → 哈希表 1
2. 遍历 C、D，查询 `-(c+d)` 在哈希表 1 中的次数累加

→ 时间从 $O(n^4)$ 降到 $O(n^2)$。

### 🛠 代码

```python
from collections import defaultdict

class Solution:
    def fourSumCount(self, A, B, C, D):
        d1 = defaultdict(int)
        for a in A:
            for b in B:
                d1[a + b] += 1
        ans = 0
        for c in C:
            for d in D:
                ans += d1[-(c + d)]
        return ans
```

---

## Q07 · 三数之和

> [LeetCode 15](https://leetcode.cn/problems/3sum/) · 难度：⭐⭐⭐ · 标签：双指针

### 🎯 为什么不用哈希？

三元组**不能重复**，去重在哈希里很麻烦 → 用**排序 + 双指针**。

### 📖 算法

1. 数组排序
2. 遍历 i：
   - `nums[i] > 0` → 直接 break（后面都更大）
   - `nums[i] == nums[i-1]` → continue（**i 的去重**）
3. left = i+1, right = n-1
   - 三数和 > 0 → right--
   - 三数和 < 0 → left++
   - 三数和 == 0 → 记录，**同时移动 left/right 并去重**

### 🛠 代码

```python
class Solution:
    def threeSum(self, nums):
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i + 1, n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > 0:
                    r -= 1
                elif s < 0:
                    l += 1
                else:
                    ans.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]: l += 1   # 去重
                    while l < r and nums[r] == nums[r-1]: r -= 1
                    l, r = l + 1, r - 1
        return ans
```

### 📊 复杂度

- 时间：**O(n²)**
- 空间：O(1)（不计输出）

### 🪤 易错点（三处去重）

1. 外层 i 去重：`if i > 0 and nums[i] == nums[i-1]`
2. 找到一组解后，l 去重：`while l < r and nums[l] == nums[l+1]: l+=1`
3. 同上，r 去重

---

## Q08 · 四数之和

> [LeetCode 18](https://leetcode.cn/problems/4sum/) · 难度：⭐⭐⭐

### 🎯 思路

四数之和 = 两层循环 + 双指针，套娃在三数之和外。

**剪枝注意**：因为目标 target 可能为负，简单的 `nums[i] > target` **不能直接 break**。
正确剪枝：`if nums[i] > target and target >= 0` 或者 `if nums[i] > target and nums[i] >= 0`。

### 📊 复杂度

- 时间：**O(n³)**
- 空间：O(1)

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
