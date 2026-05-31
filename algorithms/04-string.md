# 04 · String 字符串

## 📑 本章目录

- [基础知识](#基础知识)
- [Q01 · 反转字符串](#q01--反转字符串)
- [Q02 · 反转字符串 II](#q02--反转字符串-ii)
- [Q03 · 替换数字](#q03--替换数字)
- [Q04 · 翻转字符串里的单词](#q04--翻转字符串里的单词)
- [Q05 · 右旋字符串](#q05--右旋字符串)
- [Q06 · KMP 算法](#q06--kmp-算法)
- [Q07 · 重复的子字符串](#q07--重复的子字符串)

---

## 基础知识

- **原地反转**：双指针交换
- **KMP 算法**：利用前缀表跳过已匹配部分
- **重复子串**：`(s+s)[1:-1]` 包含 s，或 KMP 前缀表

---

## Q01 · 反转字符串

> [LeetCode 344](https://leetcode.cn/problems/reverse-string/) · 难度：⭐

```python
class Solution:
    def reverseString(self, s):
        l, r = 0, len(s) - 1
        while l < r:
            s[l], s[r] = s[r], s[l]
            l, r = l + 1, r - 1
```

---

## Q02 · 反转字符串 II

> [LeetCode 541](https://leetcode.cn/problems/reverse-string-ii/) · 难度：⭐⭐

每 2k 字符一组，反转前 k 个。Python 用切片：

```python
class Solution:
    def reverseStr(self, s, k):
        s = list(s)
        for i in range(0, len(s), 2 * k):
            s[i:i+k] = reversed(s[i:i+k])
        return ''.join(s)
```

---

## Q03 · 替换数字

> [Kamacoder 1064](https://kamacoder.com/problempage.php?pid=1064) · 难度：⭐⭐

把字符串中的数字替换为 "number"。

**关键技巧（双指针 from back）**：先扩容，从后往前填，避免 O(n²)。

```python
s = list(input())
# 数字数量
cnt = sum(1 for c in s if c.isdigit())
# 每个数字 → number 多 5 个字符
s += [''] * (cnt * 5)
i, j = len(s) - cnt * 5 - 1, len(s) - 1
while i >= 0:
    if s[i].isdigit():
        for c in 'rebmun':   # 反向填 "number"
            s[j] = c
            j -= 1
    else:
        s[j] = s[i]
        j -= 1
    i -= 1
print(''.join(s))
```

---

## Q04 · 翻转字符串里的单词

> [LeetCode 151](https://leetcode.cn/problems/reverse-words-in-a-string/) · 难度：⭐⭐

```python
class Solution:
    def reverseWords(self, s):
        return ' '.join(reversed(s.split()))
```

**进阶（不用 split）**：

1. 去除多余空格
2. 整体反转
3. 每个单词单独反转

---

## Q05 · 右旋字符串

> [Kamacoder 1065](https://kamacoder.com/problempage.php?pid=1065) · 难度：⭐⭐

把后 k 个字符移到前面。

**优雅做法**：整体反转 → 前 k 反转 → 后 (n-k) 反转。

```python
s = list(input())
k = int(input())
n = len(s)
s.reverse()
s[:k] = reversed(s[:k])
s[k:] = reversed(s[k:])
print(''.join(s))
```

---

## Q06 · KMP 算法

> [LeetCode 28](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/) · 难度：⭐⭐⭐

### 🎯 核心思想

字符不匹配时，**根据已匹配的文本跳过**部分位置，避免重复匹配。

利用**前缀表 `next`** 记录子串的**最长相等前后缀长度**。

```
needle:  a a b a a f
next:    0 1 0 1 2 0
```

匹配到 f（下标 5）失败 → 跳到 next[4] = 2 的位置继续。

### 📖 求前缀表

**核心**：先判断要不要回退（while），再判断字符是否相等。

```python
def build_next(needle):
    nxt = [0] * len(needle)
    j = 0       # 前缀末尾
    for i in range(1, len(needle)):
        while j > 0 and needle[i] != needle[j]:
            j = nxt[j - 1]                 # 回退
        if needle[i] == needle[j]:
            j += 1
        nxt[i] = j
    return nxt
```

### 📖 KMP 匹配

```python
class Solution:
    def strStr(self, haystack, needle):
        if not needle:
            return 0
        nxt = build_next(needle)
        j = 0
        for i in range(len(haystack)):
            while j > 0 and haystack[i] != needle[j]:
                j = nxt[j - 1]
            if haystack[i] == needle[j]:
                j += 1
            if j == len(needle):
                return i - j + 1
        return -1
```

### 📊 复杂度

- 时间：**O(n + m)**（朴素是 O(nm)）
- 空间：O(m)

### 🪤 易错点

- `j = nxt[j-1]`，**不是 nxt[j]**
- while 是回退多次，不是 if

---

## Q07 · 重复的子字符串

> [LeetCode 459](https://leetcode.cn/problems/repeated-substring-pattern/) · 难度：⭐⭐

### 🎯 解法一：移动匹配

`(s + s)[1:-1]` 中如果还能找到 s，说明 s 由重复子串组成。

```python
class Solution:
    def repeatedSubstringPattern(self, s):
        return s in (s + s)[1:-1]
```

### 🎯 解法二：KMP 前缀表

如果 s 由重复子串组成，**最长相等前后缀不包含的部分**一定是最小重复子串。

```python
class Solution:
    def repeatedSubstringPattern(self, s):
        nxt = build_next(s)
        n = len(s)
        if nxt[-1] != 0 and n % (n - nxt[-1]) == 0:
            return True
        return False
```

---

[⬅ 回到 algorithms](README.md) · [⬅ 回到首页](../README.md)
