# 01 · Array 数组

## 📑 本章目录

### 章节核心
- [基础知识](#基础知识)

### 题目
- [Q01 · 二分查找](#q01--二分查找) ✅
- [Q02 · 移除元素（双指针）](#q02--移除元素双指针) ✅
- [Q03 · 长度最小的子数组（滑动窗口）](#q03--长度最小的子数组滑动窗口) ⏳
- [Q04 · 区间和（前缀和）](#q04--区间和前缀和) ⏳

### 推荐练习
- [螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)
- [有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)
- [开发商购买土地](https://kamacoder.com/problempage.php?pid=1044)

---

## 基础知识

数组是最基础的数据结构。这一章重点掌握：
- **二分查找** —— 区间定义（左闭右闭 vs 左闭右开）
- **双指针** —— 快慢指针，常用于原地修改
- **滑动窗口** —— 子数组类问题
- **前缀和** —— 区间和 O(1) 查询

要点：

- 数组下标从 0 开始
- 数组内存空间地址连续 → **删除/插入需要移动其他元素，O(n)**
- C++ 中二维数组在地址空间上是连续的

---

## Q01 · 二分查找

> [LeetCode 704](https://leetcode.cn/problems/binary-search/) · 难度：⭐⭐ · 标签：二分查找

### 🎯 思路

二分查找一般用于**无重复元素的有序数组**（若有重复，下标可能不唯一）。

写二分前必须明确两件事，否则一定写错：

1. **边界条件**：`left < right` 还是 `left <= right`
2. **区间定义（不变量）**：

   | 区间 | 含义 | 循环条件 | right 更新 |
   |---|---|---|---|
   | **左闭右闭** `[left, right]` | `left == right` 仍合法 | `left <= right` | `right = mid - 1` |
   | **左闭右开** `[left, right)` | `left == right` 已无意义 | `left < right` | `right = mid` |

只要在每次循环中**始终维持当初定义的不变量**，就不会出问题。

### 🛠 代码

#### 左闭右闭写法

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1   # [left, right]
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```

#### 左闭右开写法

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums)        # [left, right)
        while left < right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid                # 注意是 mid，不是 mid - 1
        return -1
```

### 📊 复杂度

- 时间：**O(log n)**
- 空间：O(1)

### 🪤 易错点

- `mid = (left + right) // 2` 在某些语言可能溢出，更稳的写法 `left + (right - left) // 2`
- 边界条件必须跟区间定义匹配，**不能两种写法混搭**
- 数组**有重复元素**时，二分返回的下标不一定是你想要的那一个

---

## Q02 · 移除元素（双指针）

> [LeetCode 27](https://leetcode.cn/problems/remove-element/) · 难度：⭐⭐ · 标签：双指针 / 数组

### 🎯 思路

数组的"删除"不是真删，而是**把后续元素往前移**，直接做法是 O(n²)。

**双指针法 O(n)** 的核心想法：
- **快指针**：扫描原数组，跳过要删除的元素
- **慢指针**：指向新数组的"待写入位置"

```
[1, 3, 2, 3, 4]    要删除 3
 ↑slow ↑fast
 fast 找到非 3 的元素 → 写入 slow → slow++
```

### 🛠 代码

```python
class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

### 📊 复杂度

- 时间：**O(n)**（暴力是 O(n²)）
- 空间：O(1)（原地修改）

### 🪤 易错点

- 慢指针的 `slow` 是**最终长度**，所以 return 它而不是数组
- 快慢指针不要写反方向
- 双指针法对**保持元素顺序**、**原地操作**这两个需求很合适

---

## Q03 · 长度最小的子数组（滑动窗口）

> [LeetCode 209](https://leetcode.cn/problems/minimum-size-subarray-sum/) · 难度：⭐⭐ · 标签：滑动窗口

### 🎯 思路

经典**滑动窗口**：右指针不断右移扩张窗口；当 `sum >= target` 时，左指针右移收缩窗口找最短长度。

> ⚠️ **必须注意**：右指针先右移并把元素加进 sum 之后再判断（这时还无法立刻知道是否符合）。

### 🛠 代码

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: list[int]) -> int:
        n = len(nums)
        left, total, ans = 0, 0, n + 1
        for right in range(n):
            total += nums[right]
            while total >= target:        # 满足条件，左指针收缩
                ans = min(ans, right - left + 1)
                total -= nums[left]
                left += 1
        return 0 if ans == n + 1 else ans
```

### 📊 复杂度

- 时间：**O(n)**（每个元素最多进出窗口一次）
- 空间：O(1)

### 🪤 易错点

- 初值 `ans` 必须比所有可能答案都大，最后判断"是否找到"
- 滑动窗口适用前提：**单调性**（扩张时单调增、收缩时单调减），否则不能用（如包含负数的数组求"和为 K"，必须用前缀和+哈希）

---

## Q04 · 区间和（前缀和）

> [Kamacoder 1070](https://kamacoder.com/problempage.php?pid=1070) · 难度：⭐⭐ · 标签：前缀和

### 🎯 思路

要算区间 $[l, r]$ 的和，**朴素做法**是 $O(n)$ 累加。

**前缀和**：先用 $O(n)$ 算出 `prefix[i] = sum(nums[0..i-1])`，区间和变成 $O(1)$ 查询。

```
nums   : [1, 2, 3, 3, 2, 1]
prefix : [0, 1, 3, 6, 9, 11, 12]   ← 多开一位 prefix[0]=0
```

求区间 $[2, 5]$ 的和：`prefix[6] - prefix[2] = 12 - 3 = 9`。

> ⚠️ 注意 `prefix[r+1] - prefix[l]`，要把"l 之前"减掉。

### 🛠 代码

```python
n = int(input())
nums = list(map(int, input().split()))

# 构造 prefix（多开一位避免越界）
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i + 1] = prefix[i] + nums[i]

# 查询区间 [l, r]（0-indexed，闭区间）
def range_sum(l: int, r: int) -> int:
    return prefix[r + 1] - prefix[l]
```

### 📊 复杂度

- 预处理：O(n)
- 单次查询：**O(1)**
- 多次查询时优势明显（暴力是 O(nq)）

### 🪤 易错点

- **下标对齐**：`prefix[i]` 表示前 i 个元素之和，长度比 nums 大 1
- 二维前缀和：`prefix[i+1][j+1] - prefix[i][j+1] - prefix[i+1][j] + prefix[i][j]`（容斥原理）

### 🔗 同类题

- [开发商购买土地](https://kamacoder.com/problempage.php?pid=1044)
- [和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)（前缀和 + 哈希表，含负数）

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
