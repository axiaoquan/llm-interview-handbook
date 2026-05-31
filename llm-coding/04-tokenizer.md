# 04 · Tokenizer 手撕

BPE（Byte-Pair Encoding）是现代 LLM 分词器的基石（GPT/LLaMA/Qwen 都是 BPE 变种）。

## 📑 本章目录

- [Q01 · BPE 训练（学合并规则）](#q01--bpe-训练学合并规则)
- [Q02 · BPE 编码（用规则切词）](#q02--bpe-编码用规则切词)
- [Q03 · 词频统计与 vocab 构建](#q03--词频统计与-vocab-构建)

---

## Q01 · BPE 训练（学合并规则）

### 🎯 目标

从语料里学出"出现频率最高的字符对依次合并"的规则列表。

```
初始：每个词拆成单字符序列  →  "low" → ["l","o","w","</w>"]
迭代：找到最高频的相邻 pair 合并  →  ("l","o") 高频 → 合并成 "lo"
重复 k 次得到 k 条合并规则
```

### 🛠 代码（教学版）

```python
from collections import Counter
from typing import List, Tuple

def get_pair_stats(vocab: dict) -> Counter:
    """统计所有相邻字符对的频次。
    vocab: { 'l o w </w>': 5, 'l o w e r </w>': 2, ... }
    """
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair: Tuple[str, str], vocab: dict) -> dict:
    """把所有 word 中出现的 pair 合并成一个新符号"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab


def train_bpe(corpus: List[str], num_merges: int = 1000):
    """
    corpus: 词列表（建议先按空格切并加 </w> 词尾标记）
    num_merges: 合并多少次
    """
    # 1. 初始化：每个词拆成字符序列
    vocab = Counter()
    for word in corpus:
        # "low" → "l o w </w>"
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] += 1

    merges = []  # 学到的合并规则
    for i in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)        # 频次最高的 pair
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)

    return merges, vocab


# === 用法示例 ===
corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
merges, final_vocab = train_bpe(corpus, num_merges=10)
print("学到的合并规则：")
for pair in merges:
    print(' + '.join(pair))
```

### 🪤 易错点

- **`</w>` 词尾标记**：区分 "est" 在词中（`lowest`）和词尾（`newest</w>`），让模型知道边界
- **GPT-2 的优化**：直接在 byte 级别做 BPE（256 个起始符号），完美处理任何 Unicode 字符
- **训练时间**：朴素实现 O(num_merges × n_words × n_pairs)，工业级要用 priority queue 优化（HuggingFace tokenizers 是 Rust 写的）

---

## Q02 · BPE 编码（用规则切词）

### 🎯 目标

给定输入字符串，根据训练好的 merges 列表把它切成 token。

### 🛠 代码

```python
def encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """编码单个词"""
    # 1. 拆字符
    tokens = list(word) + ['</w>']

    # 2. 反复应用 merges（按规则学习的顺序，先学的优先级高）
    for pair in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            # 如果当前位置 + 1 匹配当前规则，合并
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


def encode(text: str, merges: List[Tuple[str, str]]) -> List[str]:
    """编码整个文本"""
    tokens = []
    for word in text.split():
        tokens.extend(encode_word(word, merges))
    return tokens


# === 用法 ===
result = encode("lowest newest", merges)
# 可能输出：['low', 'est</w>', 'new', 'est</w>']
```

### 🪤 高效实现：贪心 + priority queue

朴素实现是 O(n × num_merges)，工业级用 priority queue：

```python
def encode_word_fast(word, merge_ranks: dict):
    """
    merge_ranks: {('l','o'): 0, ('l','o','w'): 1, ...}  ← 规则的优先级（越早学的 rank 越小）
    """
    tokens = list(word) + ['</w>']
    while len(tokens) >= 2:
        # 找当前所有相邻 pair 中 rank 最小的
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        valid_pairs = [(merge_ranks.get(p, float('inf')), i, p) for i, p in enumerate(pairs)]
        rank, idx, best_pair = min(valid_pairs)
        if rank == float('inf'):    # 没有可合并的规则
            break
        # 合并 idx 和 idx+1
        tokens = tokens[:idx] + [tokens[idx] + tokens[idx+1]] + tokens[idx+2:]
    return tokens
```

### 🪤 易错点

1. **合并顺序很重要**：先学的优先级更高（rank 小），不能乱序应用
2. **未知字符**：用 byte-level BPE 就不会有 OOV
3. **空白处理**：GPT-2 把空格当作 word 前缀的一部分（`Ġworld`），保留空白信息

---

## Q03 · 词频统计与 vocab 构建

### 🛠 完整流程

```python
def build_vocab(corpus_file: str, num_merges: int = 30000):
    # 1. 读语料 + 词频统计
    word_counter = Counter()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                word_counter[word] += 1

    # 2. 训练 BPE
    word_list = list(word_counter.elements())
    merges, _ = train_bpe(word_list, num_merges)

    # 3. 构建 token → id 映射
    all_tokens = set()
    for word in word_counter:
        for tok in encode_word(word, merges):
            all_tokens.add(tok)

    token_to_id = {tok: i for i, tok in enumerate(sorted(all_tokens))}
    # 加特殊 token
    for special in ['<pad>', '<unk>', '<bos>', '<eos>']:
        token_to_id[special] = len(token_to_id)

    return merges, token_to_id
```

### 📊 主流模型 tokenizer 对比

| 模型 | 算法 | vocab size | 中文友好 |
|---|---|---|---|
| GPT-2 | byte-level BPE | 50257 | 差（一个汉字 ≈ 3 token） |
| LLaMA-2 | SentencePiece BPE | 32000 | 差 |
| LLaMA-3 | tiktoken (BPE) | 128000 | 较好 |
| Qwen-2 | BPE + 中文优化 | 151936 | 好（一个汉字 ≈ 1 token） |
| Tiktoken (GPT-4) | BPE | 100256 | 较好 |

### 🪤 中文 LLM 的特殊处理

- **预分词器**：先用 jieba / sentencepiece 做粗切分，再 BPE 细化
- **byte fallback**：未在 vocab 里的字符回退到 byte 级（保证 100% 编码成功）
- **vocab 扩展**：基础模型 + 行业词表，比如医学名词、代码符号

---

[⬅ 回到 llm-coding](README.md) · [⬅ 回到首页](../README.md)
