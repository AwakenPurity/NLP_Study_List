## 3. Word Embedding之word2vec分类任务（使用gensim包

之前在第一节的时候，简单介绍过`Word2Vec模型`，`Word2Vec模型`和`词袋模型`都是用于词向量化的模型：  
1. `词袋模型`主要是基于词频统计来将单词转为向量的。
2. `Word2Vec模型`通过将单词嵌入到一个连续向量空间中，使得在语义上相似的单词在向量空间中距离较近，在第一节中，我们也提到过Word2Vec 有两种主要的训练方法：CBOW（Continuous Bag of Words）和 Skip-gram。

不同于词袋模型， `Word2Vec模型`具有以下优势：  
1. 捕捉语义关系：Word2Vec 模型可以捕捉到单词之间的语义关系，相似的单词在向量空间中距离较近。
2. 相比于其他方法，Word2Vec 训练较为高效，尤其是使用负采样和分层 Softmax 技术时。
3. 泛化能力：训练好的词向量可以用于各种下游任务，如文本分类、聚类、命名实体识别等。

数据集准备：
[labeledTrainData.tsv](https://ww0.lanzout.com/iRXun26aiihc): 包含电影评论及其对应的情感标签（如正面或负面）  
[unlabeledTrainData.tsv](https://ww0.lanzout.com/i2bFV26aiiid): 没有对应的情感标的电影评论
[stopwords.txt](https://ww0.lanzout.com/iASta26aiite): 常用的停用词表，包含一组停用词，这些词在文本处理中会被忽略，因为它们对于文本的主题或意义贡献不大

### 3.1 利用没有对应的情感标的电影评论的数据集来进行词嵌入

```python
# 读取书就
df = pd.read_csv('./unlabeledTrainData.tsv', sep='\t', escapechar='\\')
print("Number of reviews: {}".format(len(df)))
df.head()
```
Number of reviews: 50000
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9999_0</td>
      <td>Watching Time Chasers, it obvious that it was ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45057_0</td>
      <td>I saw this film about 20 years ago and remembe...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15561_0</td>
      <td>Minor Spoilers&lt;br /&gt;&lt;br /&gt;In New York, Joan Ba...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7161_0</td>
      <td>I went to see this film with a great deal of e...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43971_0</td>
      <td>Yes, I agree with everyone on this site this m...</td>
    </tr>
  </tbody>
</table>
</div>

```python
stopwords = {}.fromkeys([ line.rstrip() for line in open('./stopwords.txt', encoding='utf-8')])
eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]

    return ' '.join(words)

df["clean_review"] = df.review.apply(clean_text)
df.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentiment</th>
      <th>review</th>
      <th>clean_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5814_8</td>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
      <td>stuff moment mj i ve started listening music w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2381_9</td>
      <td>1</td>
      <td>"The Classic War of the Worlds" by Timothy Hin...</td>
      <td>classic war worlds timothy hines a entertainin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7759_3</td>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
      <td>film starts a manager nicholas bell giving inv...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3630_4</td>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
      <td>assumed praised film greatest filmed opera did...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9495_8</td>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
      <td>superbly trashy wondrously unpretentious s exp...</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

```python

```

```python

```

```python

```

```python

```
