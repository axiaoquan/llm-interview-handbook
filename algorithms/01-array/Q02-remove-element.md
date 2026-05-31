# Q02 · 移除元素（双指针）

> [LeetCode 27](https://leetcode.cn/problems/remove-element/) · 难度：⭐⭐ · 标签：双指针 / 数组

## 🎯 思路

数组的"删除"不是真删，而是**把后续元素往前移**，直接做法是 O(n²)。

**双指针法 O(n)** 的核心想法：
- **快指针**：扫描原数组，跳过要删除的元素
- **慢指针**：指向新数组的"待写入位置"

```
[1, 3, 2, 3, 4]    要删除 3
 ↑slow ↑fast
 fast 找到非 3 的元素 → 写入 slow → slow++
```

## 🛠 代码

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

## 📊 复杂度

- 时间：**O(n)**（暴力是 O(n²)）
- 空间：O(1)（原地修改）

## 🪤 易错点

- 慢指针的 `slow` 是**最终长度**，所以 return 它而不是数组
- 快慢指针不要写反方向
- 双指针法对**保持元素顺序**、**原地操作**这两个需求很合适

## 🔗 同类题

- [有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)
- [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)
- [移动零](https://leetcode.cn/problems/move-zeroes/)
