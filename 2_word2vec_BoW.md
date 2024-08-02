## Bag of Words

词袋模型（Bag of Words, BoW）是将文本转换为词的向量最基本的方法之一。  

其主要通过统计每个单词在文本中出现的次数，将文本转换为固定长度的向量表示，其中每个向量元素对应词汇表中的一个单词。

### 1. 主要概念
`词汇表`：词袋模型会为整个语料库中的所有不同单词创建一个词汇表。  
`词频`：词袋模型会计算每个单词在每个文档中出现的次数。  

### 2. 主要步骤
2.1 文本预处理：  
- 去除标点符号和特殊字符
- 转换为小写
- 分词（将文本拆分为单词）
- 去除停用词（如 "the", "is", "in" 等）

2.2 创建词汇表：  
对整个语料库中的所有文档进行分词，创建一个包含所有不同单词的词汇表。  

### 3. 示例

假设我们有以下几个简单的文本：
1. "I love machine learning."
2. "Machine learning is great."
3. "I love learning."
4. "I love machine learning and i am great."

经过移除标点符号、转换为小写以及分词后，创建词汇表：  
`['i', 'love', 'machine', 'learning', 'and', 'am', 'great', 'is']`  

根据词汇表，将句子 "I love machine learning and i am great" 转换为词频向量：`[2, 1, 1, 1, 1, 1, 1, 0]`.  

解释：  
- 'i' 出现了 2 次  
- 'love' 出现了 1 次  
- 'machine' 出现了 1 次  
- 'learning' 出现了 1 次  
- 'and' 出现了 1 次  
- 'am' 出现了 1 次  
- 'great' 出现了 1 次  
- 'is' 出现了 0 次  
