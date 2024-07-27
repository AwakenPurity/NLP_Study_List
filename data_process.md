## 1.数据预处理  
在自然语言处理任务中，我们首先需要考虑的是：计算机是无法直接识别任务文字的（包括:中文、英文、德文等等），**"词如何在计算机中表示"** 是任何自然语言处理任务中的 **"入门券"**。

### 1.1 One-hot 编码  
One-hot 编码，又叫做独热编码，其是一种将数据转换为数值数据的技术；每个类别用一个位图来表示，该位图中只有一个位置的值为1，其余位置的值为0。  

比如：“我”，“爱”，“你”这三个单词，One-hot 编码表示如下：   

| 我      | 1  | 0  | 0  |
|---------|----|----|----|
| 爱      | 0  | 1  | 0  |
| 你      | 0  | 0  | 1  |

但这样做，One-hot的维度是由词库（单词的数量）来决定的，如果是10000个单词，One-hot维度就须是10000维，  
这往往会导致两个问题：  
- 维度灾难，随着数据维度的增加，导致数据变得非常稀疏；这使得寻找相似点或进行聚类等变得困难。
- 无法度量每个词语之间的相似度，就比如："Peking University"和"北京大学"表示的都是同一个意思，如果用One-hot 编码，并不能够表示出两个词汇之间的相似度，无法度量。


### 1.2 Word2Vec 词嵌入

为了解决One-hot 编码的困境，谷歌团队提出了Word2Vec的方法。  
**Word2Vec** 是一种用于将单词转换为向量表示的方法，该方法旨在通过从大量文本数据中学习，生成能够捕捉单词语义信息的向量，使得在向量空间中，**相似意义的词距离更近**。  

简单地说，12维度的数据，在One-hot 编码中只能表示12个单词，而在Word2Vec中，可以表示无数个单词。 
通常，数据的维度越高，能提供的信息量也就越大，一般数据的维度我们控制在 **`[50, 100]`** 之间。

要准确地将单词进行向量化表示，则需要考虑两件事：
- 单词 (单个token) 本身的含义
- 单词和其他单词之间的关联或关系或者相似度

单词本身的含义一般是很容易就可以做到的，难点在于如何将一个单词和其他单词进行关联；Word2Vec主要是提出了两种方案：  
- Continuous Bag-of-Words (CBOW) : 根据上下文词预测中心词
- Skip-Gram: 根据中心词预测上下文词  


**CBOW的流程**：  
- 数据集的构建：  
  `example: I tried to update my Scipy library version but it was still the same`
   
  **一、数据预处理**
  1. 从文本数据中去除标点符号、特殊字符等
  2. 标记化（Tokenization）： 将文本数据分割成单词序列
  3. 创建词汇表（Vocabulary）： 建立一个包含所有单词的词汇表，并为每个单词分配一个唯一的索引  
  
  **二、构建上下文窗口**
  1. 定义窗口大小：在模型训练之前，我们需要定义一个单词周围的哪几个单词作为其上下文；例如,窗口大小为 `2` ，这意味着在目标词的前后各取2个单词作为上下文。  
     | 1  |  2   |    3   |    4   |  5 |
     |---|-------|--------|--------|----|
     | I | tried | **to** | update | my |
     
     `to` 是需要被预测的中心词，`I`, `tried`, `update` 和 `my` 就是其上下文。  

  **三、生成上下文-目标对**  
  对于每个目标词，提取其前后的上下文词。
  例如，对于目标词 `to`，其上下文词为 `I`, `tried`, `update`, `my`。  
  则一个 `数据对` 可以表示为 {  
  x: `I`, `tried`, `update`, `my`;  
  y: `to`  
  }

  **四、训练过程**  
  如果一个语料库大一点，比如，有4W个单词，那我们在模型最有一层 `softmax 层` 计算的时候，计算量将会很大，十分耗时。

  - **初始方案**：
  输入两个单词，判断他们是不是前后对应的输入和输出，可以看成是一个二分类任务。  
  （输入`I`, `tried`，判断 `tried` 是 `I` 后面单词的可能性是多少？）  
  但这种情况下，构建出来的数据集的标签全为1，无法进行较好的训练。

  - **负采样方案**：  
  正样本选择：对于一个给定的上下文窗口，选择实际出现的目标词作为正样本。  
  负样本选择：从词汇表中随机选择若干个词作为负样本，**这些词不应该出现在当前上下文中**。


      | input | output | target |
      |-------|--------|--------|
      |   I   |  tried |   1    |
      |   I   | version|   0    |
      |   I   |   but  |   0    |

      负采样的个数一般在 `5` 个左右（经验值）。

 ### 1.3 繁体字转为简体字
 通常，在使用维基百科数据集的时候，往往会遇到很多繁体字，需要使用 `opencc` 这个包来进行转换。  
 '''  
 import opencc   
 converter = opencc.OpenCC('t2s.json')    
 converter.convert('汉字')  # 漢字    
 '''  

 参考链接：https://github.com/BYVoid/OpenCC?tab=readme-ov-file  
 维基百科数据集连接：https://dumps.wikimedia.org/zhwiki/latest/


 ### 1.4 代码  
 1. 利用 `gensim` 包来对进行词嵌入
 ```
 from gensim.models import word2vec  
 import logging
 logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
 
 # 随便以一句话为例
 raw_sentences = ["I tried to update my Scipy library version but it was still the same", " I deprecated in the last Scipy version"]  
 
 # 分词
 sentences = [s.split() for s in raw_sentences]
 
 # 词嵌入模型训练
 '''
  sentences: 输入数据，
  min_count: 过滤词频小于等于1的单词，
  Gensim中的Word2Vec模型默认使用CBOW方法，如果你想明确指定使用CBOW或Skip-gram，可以通过设置参数sg来选择：  
    sg=0 使用CBOW方法（默认值）  
    sg=1 使用Skip-gram方法  
 '''
 model = word2vec.Word2Vec(sentences, min_count=1)
 
 # 判断两个词的相似程度
 model.wv.similarity('tried', 'last')
 ```
