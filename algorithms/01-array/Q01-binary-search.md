# Q01 · 二分查找

> [LeetCode 704](https://leetcode.cn/problems/binary-search/) · 难度：⭐⭐ · 标签：二分查找

## 🎯 思路

二分查找一般用于**无重复元素的有序数组**（若有重复，下标可能不唯一）。

写二分前必须明确两件事，否则一定写错：

1. **边界条件**：`left < right` 还是 `left <= right`
2. **区间定义（不变量）**：

   | 区间 | 含义 | 循环条件 | right 更新 |
   |---|---|---|---|
   | **左闭右闭** `[left, right]` | `left == right` 仍合法 | `left <= right` | `right = mid - 1` |
   | **左闭右开** `[left, right)` | `left == right` 已无意义 | `left < right` | `right = mid` |

只要在每次循环中**始终维持当初定义的不变量**，就不会出问题。

## 🛠 代码

### 左闭右闭写法

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

### 左闭右开写法

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

## 📊 复杂度

- 时间：**O(log n)**
- 空间：O(1)

## 🪤 易错点

- `mid = (left + right) // 2` 在某些语言可能溢出，更稳的写法 `left + (right - left) // 2`
- 边界条件必须跟区间定义匹配，**不能两种写法混搭**
- 数组**有重复元素**时，二分返回的下标不一定是你想要的那一个

## 🔗 同类题

- [螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)
- [搜索插入位置](https://leetcode.cn/problems/search-insert-position/)
- [在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)
