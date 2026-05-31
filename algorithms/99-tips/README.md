# 99 · Tips · Python & 工程小贴士

刷题过程中踩过的坑、好用的 Python 习惯。

---

## 1. 输入读取

### 一次性读全部输入（OJ 常用）

```python
import sys
data = sys.stdin.read().split()
# 输入 "1 2 3\n4 5" → ["1", "2", "3", "4", "5"]（注意是字符串）
```

读完一定要按需 `int(x)` 或 `float(x)` 转类型。

---

## 2. 二维数组创建陷阱 ⚠️

```python
# ❌ 错误写法：n 个子列表是同一个对象的引用
result = [[0] * n] * n
result[0][0] = 1
# 此时所有行的第一列都变成了 1！

# ✅ 正确写法：列表推导式，每行独立
result = [[0] * n for _ in range(n)]
```

---

## 3. list 操作的时间复杂度

| 操作 | 复杂度 |
|---|---|
| `list.append(x)` | O(1) 摊销 |
| `list.pop()` （末尾） | **O(1)** |
| `list.pop(i)` （指定下标） | **O(n)** |
| `list[a:b]`（切片） | **O(n)** |
| `x in list`（线性查找） | O(n) |

刷题时如果频繁 `list.pop(0)` → 用 `collections.deque` 改成 O(1)。

---

## 4. 判断字符类型

```python
s = "a1B"
"a".isdigit()    # False
"1".isdigit()    # True
"a".isalpha()    # True，大小写都算
"A".isalpha()    # True
"A".isupper()    # True
"a".islower()    # True
```

---

## 5. 字典 / 哈希常用招式

```python
# 计数器 (推荐)
from collections import Counter
cnt = Counter("hello")  # {'h':1, 'e':1, 'l':2, 'o':1}

# 默认值
from collections import defaultdict
d = defaultdict(int)    # 不存在的 key 默认值 0
d[k] += 1               # 不需要先判断 k in d

# get(k, default)
d.get(k, 0)
```

---

## 6. 集合不能存 list 但能存 tuple

```python
# 字母异位词分组的常见技巧
key = tuple(sorted(s))   # 把字符串/list 转成 tuple 才能做 dict key
groups[key].append(s)
```

---

## 7. 字符串拼接性能

```python
# ❌ 慢（每次都生成新字符串）
s = ""
for c in lst:
    s += c

# ✅ 快
"".join(lst)
```

---

## 8. 二分查找标准库

```python
import bisect
bisect.bisect_left(arr, x)   # 第一个 >= x 的位置
bisect.bisect_right(arr, x)  # 第一个 > x 的位置
bisect.insort_left(arr, x)   # 插入并保持有序
```

很多场景比手写二分省事且不容易翻车。

---

## 9. 堆 (优先队列)

```python
import heapq
h = []
heapq.heappush(h, x)
top = heapq.heappop(h)   # 默认是小顶堆

# 大顶堆 → 取负数
heapq.heappush(h, -x)
```

---

## 10. 不可变默认参数陷阱 ⚠️

```python
# ❌ 错误：默认值是同一个 list 实例！
def f(x, lst=[]):
    lst.append(x)
    return lst
f(1)  # [1]
f(2)  # [1, 2]   ← 共享！

# ✅ 正确
def f(x, lst=None):
    if lst is None:
        lst = []
    lst.append(x)
    return lst
```

---

[⬅ 回到 algorithms](../README.md)
