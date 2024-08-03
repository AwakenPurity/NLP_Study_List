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
[unlabeledTrainData.tsv](https://ww0.lanzout.com/i2bFV26aiiid): 没有对应的情感标签的电影评论
[stopwords.txt](https://ww0.lanzout.com/iASta26aiite): 常用的停用词表，包含一组停用词，这些词在文本处理中会被忽略，因为它们对于文本的主题或意义贡献不大

### 3.1 利用`没有对应的情感标签`的电影评论的数据集来进行词嵌入

```python
# 读取数据
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
      <th>review</th>
      <th>clean_review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9999_0</td>
      <td>Watching Time Chasers, it obvious that it was ...</td>
      <td>watching time chasers obvious made a bunch fri...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45057_0</td>
      <td>I saw this film about 20 years ago and remembe...</td>
      <td>i film years ago remember nasty i based a true...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15561_0</td>
      <td>Minor Spoilers&lt;br /&gt;&lt;br /&gt;In New York, Joan Ba...</td>
      <td>minor spoilersin york joan barnard elvire audr...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7161_0</td>
      <td>I went to see this film with a great deal of e...</td>
      <td>i film a great deal excitement i school direct...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43971_0</td>
      <td>Yes, I agree with everyone on this site this m...</td>
      <td>i agree site movie bad call a movie insult mov...</td>
    </tr>
  </tbody>
</table>
</div>

```python
review_part = df["clean_review"]
review_part.shape
```
(50000,)  

```python
# 忽略程序运行时产生的所有警告warnings信息
import warnings
warnings.filterwarnings("ignore")
```

```python
# 'punkt'是一个分词器模型，用于将句子拆分成单词或标点符号
nltk.download('punkt')
# 加载 punkt 英语句子分割器
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
```

```python
def split_sentence(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [ clean_text(s) for s in raw_sentences if s]
    return sentences

sentences = sum(review_part.apply(split_sentence), [])
print('{} reviews -> {} sentences'.format(len(review_part), len(sentences)))
```
50000 reviews -> 50000 sentences

```python
sentences[0]
```
'watching time chasers obvious made a bunch friends sitting day film school hey s pool money make a bad movie ended making a bad movie dull story bad script lame acting poor cinematography bottom barrel stock music corners cut prevented film s release life s'  

```python
sentences_list = []
for line in sentences:
    sentences_list.append(nltk.word_tokenize(line))
```

```python
# 设定词向量训练的参数
num_features = 300 # 词的特征维度
min_word_count = 40 # 最小词频
num_workers = 4 # 
context = 10 # 上下文词汇大小
model_name = "{}features_{}minwords_{}context.model".format(num_features, min_word_count, context)
```

利用gensim来训练word2vec模型
```python
from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences_list, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context)
```

```python
model.init_sims(replace=True) # 优化模型以节省内存
model.save(model_name)
```

```python
# 从一组单词中找出那个最不匹配的单词，换句话说，就是找出与其他单词在语义上最不相关的单词。
model.wv.doesnt_match(['kitchen', 'man', 'women']) 
```
'kitchen'

```python
# 查找和某个单词最相似的方法
model.wv.most_similar("boy")
```
[('girl', 0.7107567191123962),
 ('astro', 0.6462628841400146),
 ('orphan', 0.6284143924713135),
 ('teenager', 0.6098208427429199),
 ('kid', 0.5947464108467102),
 ('lad', 0.5718541145324707),
 ('teenage', 0.5633155703544617),
 ('brat', 0.5550559759140015),
 ('child', 0.5489398837089539),
 ('yr', 0.5479920506477356)]

利用gensim训练得到的word2vec模型来对有标签的数据进行词嵌入，再进行分类
 ```python
df = pd.read_csv('./labeledTrainData.tsv', sep='\t', escapechar='\\')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5814_8</td>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2381_9</td>
      <td>1</td>
      <td>"The Classic War of the Worlds" by Timothy Hin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7759_3</td>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3630_4</td>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9495_8</td>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
    </tr>
  </tbody>
</table>
</div>  


 ```python
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text() # 去除html标签
    text = re.sub(r'[^a-zA-Z]', ' ', text) # 所有非字母字符（即不是英文字母的字符）替换为空格
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

def to_review_vector(review):
    global word_vec

    review = clean_text(review, remove_stopwords=True)
    word_vec = np.zeros((1,300))
    for word in review:
        if word in model.wv: #  # 使用 model.wv 检查单词是否在模型中
            word_vec += np.array([model.wv[word]])
    
    return pd.Series(word_vec.mean(axis=0))

train_data_features = df.review.apply(to_review_vector)
train_data_features.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>290</th>
      <th>291</th>
      <th>292</th>
      <th>293</th>
      <th>294</th>
      <th>295</th>
      <th>296</th>
      <th>297</th>
      <th>298</th>
      <th>299</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.394048</td>
      <td>-0.852723</td>
      <td>1.114327</td>
      <td>-1.505018</td>
      <td>-0.655734</td>
      <td>2.151611</td>
      <td>2.536792</td>
      <td>-0.098964</td>
      <td>-0.149527</td>
      <td>-0.503505</td>
      <td>...</td>
      <td>0.094960</td>
      <td>1.030062</td>
      <td>-0.048680</td>
      <td>2.427370</td>
      <td>0.630168</td>
      <td>0.723000</td>
      <td>1.841443</td>
      <td>-0.788298</td>
      <td>-0.573784</td>
      <td>-3.397351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.242756</td>
      <td>-1.239282</td>
      <td>-1.382195</td>
      <td>-2.138052</td>
      <td>-0.778870</td>
      <td>3.038621</td>
      <td>-0.577753</td>
      <td>-1.344181</td>
      <td>-1.513765</td>
      <td>-0.144517</td>
      <td>...</td>
      <td>1.542312</td>
      <td>1.745327</td>
      <td>0.796945</td>
      <td>3.184882</td>
      <td>2.471293</td>
      <td>1.997546</td>
      <td>0.592304</td>
      <td>-2.263047</td>
      <td>-1.541600</td>
      <td>-3.333471</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.295596</td>
      <td>1.155815</td>
      <td>1.159109</td>
      <td>5.577098</td>
      <td>3.805943</td>
      <td>-5.697885</td>
      <td>0.753577</td>
      <td>3.266692</td>
      <td>-0.616527</td>
      <td>2.054443</td>
      <td>...</td>
      <td>-1.087373</td>
      <td>3.034952</td>
      <td>1.233940</td>
      <td>-1.618164</td>
      <td>-0.456538</td>
      <td>1.089408</td>
      <td>2.019972</td>
      <td>-1.931761</td>
      <td>-1.460601</td>
      <td>-0.773434</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.244674</td>
      <td>-0.311246</td>
      <td>-0.945215</td>
      <td>-2.912349</td>
      <td>-0.258065</td>
      <td>-0.505598</td>
      <td>-0.997139</td>
      <td>2.164532</td>
      <td>-0.790215</td>
      <td>-3.752288</td>
      <td>...</td>
      <td>2.493686</td>
      <td>2.543949</td>
      <td>0.564080</td>
      <td>-0.145979</td>
      <td>0.855271</td>
      <td>0.518375</td>
      <td>-0.916275</td>
      <td>-1.522183</td>
      <td>-0.908431</td>
      <td>-4.771239</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.542964</td>
      <td>1.511702</td>
      <td>2.476547</td>
      <td>2.734181</td>
      <td>-0.850201</td>
      <td>-3.805721</td>
      <td>3.280828</td>
      <td>2.702045</td>
      <td>2.409029</td>
      <td>1.172359</td>
      <td>...</td>
      <td>0.143987</td>
      <td>0.729355</td>
      <td>0.966916</td>
      <td>-0.529388</td>
      <td>-0.037723</td>
      <td>3.727327</td>
      <td>1.128880</td>
      <td>-1.660366</td>
      <td>-0.895423</td>
      <td>0.237743</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 300 columns</p>
</div>

 ```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_features, df.sentiment, test_size=0.2, random_state=0)
```

 ```python
LR_model = LogisticRegression()
LR_model = LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

print("Recall metric in the testing data: ", cnf_matrix[1,1] / (cnf_matrix[1,0]+cnf_matrix[1,1]))
print("acc metric in the testing data: ", (cnf_matrix[1,1] + cnf_matrix[0,0]) / cnf_matrix.sum())

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=class_names,
                      title="Confusion Matrix")
plt.show()
```
Recall metric in the testing data:  0.8739804241435563
acc metric in the testing data:  0.865
![image](https://github.com/user-attachments/assets/03c97e58-ee8c-46c1-94a9-d12f70805a55)

