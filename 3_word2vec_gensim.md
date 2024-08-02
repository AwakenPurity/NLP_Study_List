## Word Embedding之word2vec分类任务（使用gensim包

之前在第一节的时候，简单介绍过`Word2Vec模型`，`Word2Vec模型`和`词袋模型`都是用于词向量化的模型：  
1. `词袋模型`主要是基于词频统计来将单词转为向量的
2. `Word2Vec模型`通过将单词嵌入到一个连续向量空间中，使得在语义上相似的单词在向量空间中距离较近，在第一节中，我们也提到过Word2Vec 有两种主要的训练方法：CBOW（Continuous Bag of Words）和 Skip-gram。

