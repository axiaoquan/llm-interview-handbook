# 08 · MonotonicStack 单调栈

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 每日温度](#q01--每日温度)
- [Q02 · 下一个更大元素 I](#q02--下一个更大元素-i)
- [Q03 · 下一个更大元素 II](#q03--下一个更大元素-ii)
- [Q04 · 接雨水](#q04--接雨水)
- [Q05 · 柱状图中最大的矩形](#q05--柱状图中最大的矩形)

---

## 基础知识

### 套路

**核心场景**：求"下一个更大 / 更小元素"。

**栈中保存的是下标**（不是值），遇到比栈顶大（或小）的元素就**弹出栈顶并处理**。

### 单调递增 vs 递减

- 求**下一个更大** → 单调**递减**栈（遇到更大就弹）
- 求**下一个更小** → 单调**递增**栈（遇到更小就弹）

---

## Q01 · 每日温度

> [LeetCode 739](https://leetcode.cn/problems/daily-temperatures/) · 难度：⭐⭐

### 🎯 思路

求每个温度等待几天后才有更高温度。
**栈中存日期下标**，遇到更高温度时弹出栈顶并算差值。

### 🛠 代码

```python
class Solution:
    def dailyTemperatures(self, T):
        ans = [0] * len(T)
        stack = []                 # 存下标
        for i, t in enumerate(T):
            while stack and T[stack[-1]] < t:
                j = stack.pop()
                ans[j] = i - j
            stack.append(i)
        return ans
```

### 📊 复杂度

- 时间：**O(n)**（每元素最多进出一次）
- 空间：O(n)

---

## Q02 · 下一个更大元素 I

> [LeetCode 496](https://leetcode.cn/problems/next-greater-element-i/) · 难度：⭐⭐

### 🎯 思路

先用单调栈求 nums2 中每个元素的"下一个更大"，存入哈希表；再遍历 nums1 直接查。

```python
class Solution:
    def nextGreaterElement(self, nums1, nums2):
        nxt = {}
        stack = []
        for x in nums2:
            while stack and stack[-1] < x:
                nxt[stack.pop()] = x
            stack.append(x)
        return [nxt.get(x, -1) for x in nums1]
```

---

## Q03 · 下一个更大元素 II

> [LeetCode 503](https://leetcode.cn/problems/next-greater-element-ii/) · 难度：⭐⭐

### 🎯 思路

**环形数组** → 把数组拼接两倍长（用 `i % n` 模），再跑单调栈。

```python
class Solution:
    def nextGreaterElements(self, nums):
        n = len(nums)
        ans = [-1] * n
        stack = []
        for i in range(2 * n):
            x = nums[i % n]
            while stack and nums[stack[-1]] < x:
                ans[stack.pop()] = x
            if i < n:
                stack.append(i)
        return ans
```

---

## Q04 · 接雨水

> [LeetCode 42](https://leetcode.cn/problems/trapping-rain-water/) · 难度：⭐⭐⭐⭐

### 🎯 三种解法

#### 1) 双指针（最优）

每个位置的盛水量 = `min(左最大, 右最大) - 当前高度`。

双指针从两端向内：哪边短板**已确定**就处理哪边。

```python
class Solution:
    def trap(self, height):
        l, r = 0, len(height) - 1
        lmax = rmax = 0
        ans = 0
        while l <= r:
            if height[l] <= height[r]:
                lmax = max(lmax, height[l])
                ans += lmax - height[l]
                l += 1
            else:
                rmax = max(rmax, height[r])
                ans += rmax - height[r]
                r -= 1
        return ans
```

**时间 O(n)，空间 O(1)。**

#### 2) 两数组（更直观，空间 O(n)）

预处理每个位置的左最大和右最大数组。

#### 3) 单调栈

遇到更高的柱子就计算"凹槽"的雨水。

---

## Q05 · 柱状图中最大的矩形

> [LeetCode 84](https://leetcode.cn/problems/largest-rectangle-in-histogram/) · 难度：⭐⭐⭐⭐

### 🎯 思路

**单调递增栈**：栈顶 ↘ 栈底单调递减；遇到更小元素弹栈顶时，把它当矩形的高 h，弹后栈顶为左边界，当前 i 为右边界，面积 `h * (right - left - 1)`。

> ⚠️ **数组两侧补 0** 简化边界处理。

### 🛠 代码

```python
class Solution:
    def largestRectangleArea(self, heights):
        heights = [0] + heights + [0]    # 两侧补 0
        stack = []
        ans = 0
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                top = stack.pop()
                left = stack[-1]         # 弹后栈顶
                ans = max(ans, heights[top] * (i - left - 1))
            stack.append(i)
        return ans
```

### 📊 复杂度

- 时间：**O(n)**
- 空间：O(n)

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
