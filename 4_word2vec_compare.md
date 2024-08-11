## 3. 通过词袋模型、TF-IDF模型、词向量模型、深度学习模型分别进行词嵌入，来查看他们的区别。

本节里所用的数据集和文件均可在下面获取：  
[socialmedia_relevant_cols_clean.csv](https://ww0.lanzout.com/iU6Ux276hpeh)  
[cleaned_data.csv](https://ww0.lanzout.com/icAWk276hvuj)


### 3.1 数据预处理
```python
import keras
import nltk
import pandas as pd
import numpy as np
import re
import codecs
```
```python
questions = pd.read_csv("socialmedia_relevant_cols_clean.csv")
questions.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>choose_one</th>
      <th>class_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>just happened a terrible car crash</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>our deeds are the reason of this  earthquake m...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>heard about  earthquake is different cities, s...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>there is a forest fire at spot pond, geese are...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>forest fire near la ronge sask  canada</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
questions.describe()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>class_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10876.000000</td>
      <td>10876.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5437.500000</td>
      <td>0.432604</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3139.775098</td>
      <td>0.498420</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2718.750000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5437.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8156.250000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10875.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>

**数据清洗，去掉无用字符**
```python
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^a-zA-Z0-9(),!?@\'\`\"\_\n]", "")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

questions = standardize_text(questions, "text")

questions.to_csv("cleaned_data.csv")
questions.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>choose_one</th>
      <th>class_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>just happened a terrible car crash</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>our deeds are the reason of this  earthquake m...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>heard about  earthquake is different cities, s...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>there is a forest fire at spot pond, geese are...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>forest fire near la ronge sask  canada</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
clean_data = pd.read_csv("cleaned_data.csv")
clean_data.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>text</th>
      <th>choose_one</th>
      <th>class_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>just happened a terrible car crash</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>our deeds are the reason of this  earthquake m...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>heard about  earthquake is different cities, s...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>there is a forest fire at spot pond, geese are...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>forest fire near la ronge sask  canada</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 上面可以看出有些列重复了，进行删除
clean_data = clean_data.drop(columns=['Unnamed: 0.1'])
clean_data = clean_data.drop(columns=['Unnamed: 0'])

clean_data.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>choose_one</th>
      <th>class_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>just happened a terrible car crash</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>our deeds are the reason of this  earthquake m...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heard about  earthquake is different cities, s...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>there is a forest fire at spot pond, geese are...</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>forest fire near la ronge sask  canada</td>
      <td>Relevant</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

**查看数据分布情况: 是否不平衡**
```python
clean_data.groupby("class_label").count()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>choose_one</th>
    </tr>
    <tr>
      <th>class_label</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6187</td>
      <td>6187</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4673</td>
      <td>4673</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>

**分词、数据集划分**
```python
# 分词
# 使用了NLTK库中的RegexpTokenizer来对文本进行分词
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_data["tokens"] = clean_data["text"].apply(tokenizer.tokenize)
clean_data.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>choose_one</th>
      <th>class_label</th>
      <th>tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>just happened a terrible car crash</td>
      <td>Relevant</td>
      <td>1</td>
      <td>[just, happened, a, terrible, car, crash]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>our deeds are the reason of this  earthquake m...</td>
      <td>Relevant</td>
      <td>1</td>
      <td>[our, deeds, are, the, reason, of, this, earth...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>heard about  earthquake is different cities, s...</td>
      <td>Relevant</td>
      <td>1</td>
      <td>[heard, about, earthquake, is, different, citi...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>there is a forest fire at spot pond, geese are...</td>
      <td>Relevant</td>
      <td>1</td>
      <td>[there, is, a, forest, fire, at, spot, pond, g...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>forest fire near la ronge sask  canada</td>
      <td>Relevant</td>
      <td>1</td>
      <td>[forest, fire, near, la, ronge, sask, canada]</td>
    </tr>
  </tbody>
</table>
</div>

**语料库情况**
```python
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
```

```python
# 一共有多少个单词
all_words = [word for tokens in clean_data["tokens"] for word in tokens]
# 每个句子的长度
sequences_lengths = [len(tokens) for tokens in clean_data["tokens"]]
# 语料库包含多少个不同的单词
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabluary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sequences_lengths))
```
154724 words total, with a vocabluary size of 18101  
Max sentence length is 34  

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
plt.xlabel("Sentence length")
plt.ylabel("Number of sentences")
plt.hist(sequences_lengths)
plt.show()
```
![image](https://github.com/user-attachments/assets/b3529fe7-623d-4e70-9a3d-01a7b9ec13ef)

下面通过词袋模型、TF-IDF模型、词向量模型、深度学习模型分别进行词嵌入，来查看他们的区别。

### 3.2 词袋模型 Bag of Words Counts 
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv(data):
    count_vectirizer = CountVectorizer()
    emb = count_vectirizer.fit_transform(data)
    return emb, count_vectirizer

# 将clean_data["text"]所有内容都放在一个列表中
list_corpus = clean_data["text"].tolist()
list_labels = clean_data["class_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=32)

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)
```

**PCA展示Bag of Words**
```python
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches

def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'blue']
    if plot:
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

fig = plt.figure(figsize=(16, 16))
plot_LSA(X_train_counts, y_train)
plt.show()
```
![image](https://github.com/user-attachments/assets/c413e3c8-3a39-4243-8b69-364f4cf4ce30)

**可以看来PCA效果并不是很好**

**逻辑回归分类看一下结果**
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=30.0, class_weight='balanced', solver="newton-cg",
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_count = clf.predict(X_test_counts)
```

```python
# 评估
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_count)
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" % (accuracy, precision, recall, f1))
```
accuracy: 0.769, precision: 0.771, recall: 0.769, f1: 0.769  

```python
# 混淆矩阵
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, 
                          classes, 
                          title="Confusion matrix", 
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.xticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
```

```python
cm = confusion_matrix(y_test, y_predicted_count)
fig = plt.figure(figsize=(10,10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], title="Confusion Matrix")
plt.show()
```
![image](https://github.com/user-attachments/assets/54778841-f3d7-4241-bd03-6aebfe3f759d)

```python
print(cm)
```
[[1001  210    5]
 [ 283  673    2]
 [   2    0    0]]

**进一步检查模型的关注点  **
```python
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom': bottom
        }
        return classes

importance = get_most_important_features(count_vectorizer, clf, 10)
importance
```
{0: {'tops': [(2.115824123340634, 'windy'),
   (2.1799153731543583, 'weighs'),
   (2.2054055038586355, 'ice'),
   (2.215906647191191, 'age'),
   (2.259999163387926, 'breaks'),
   (2.2744918042282802, 'christmas'),
   (2.2827571787082026, 'poll'),
   (2.5208192943590193, 'swell'),
   (3.1090845052883997, 'finally'),
   (3.545381511454468, 'po')],
  'bottom': [(-2.8219688769947275, 'x1392'),
   (-3.0023705607413107, 'typhoon'),
   (-3.021015416493898, 'sunburned'),
   (-3.080252298531729, 'derailment'),
   (-3.091500326132359, 'distance'),
   (-3.2005696202249267, 'ryans'),
   (-3.53937836422686, 'stake'),
   (-3.5881606223436653, 'hiroshima'),
   (-4.747463021900055, 'deaths'),
   (-4.884968504327644, 'storm')]}}  

```python
def plot_important_words(top_scores,top_words,bottom_scores,bottom_words,name):
    y_pos = np.arange(len(top_words))
    top_pairs =[(a,b)for a,b in zip(top_words, top_scores)]
    top_pairs=sorted(top_pairs, key=lambda x:x[1])
    
    bottom_pairs =[(a,b)for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs=sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
   
    top_words = [a[0] for a in top_pairs]
    top_scores =[a[1] for a in top_pairs]

    bottom_words =[a[0] for a in bottom_pairs]
    bottom_scores =[a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words' , fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt. subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Disaster', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance' , fontsize=20)
```

```python
top_scores = [a[0] for a in importance[0]['tops']]
top_words = [a[1] for a in importance[0]['tops']]

bottom_scores = [a[0] for a in importance[0]['bottom']]
bottom_words = [a[1] for a in importance[0]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")
```
![image](https://github.com/user-attachments/assets/0c71f6d1-7109-4b70-9450-10cde97ade5d)


```python
import numpy as np

def get_precision_recall(pred_labels, ground_labels):
    pred_labels = np.reshape(pred_labels, (-1, 1))
    ground_labels = np.reshape(ground_labels, (-1, 1))
    classification = np.sum(np.equal(pred_labels, ground_labels)) / (
                ground_labels.shape[0] * ground_labels.shape[1]) * 100
    true_results = pred_labels[ground_labels == 1]

    ########################################
    TP = np.sum(true_results)
    epsilon = 1e-7  # Small value to avoid division by zero
    precision = TP / (np.sum(pred_labels[pred_labels == 1]) + epsilon) * 100
    recall = TP / (np.sum(ground_labels[ground_labels == 1]) + epsilon) * 100

    ###########################################

    print('accu:%.2f, precision:%.2f, recall=%.2f' % (classification, precision, recall))
    return classification, precision, recall

# Example usage
pred_labels = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
ground_labels = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
get_precision_recall(pred_labels, ground_labels)
```
accu:77.78, precision:83.33, recall=83.33  
(77.77777777777779, 83.33333194444447, 83.33333194444447)


### 3.3 TF-IDF Bag of Words

TF-IDF 是一种用于信息检索与文本挖掘的常用加权技术，用来评估一个词对于一个文档集或语料库中某个文档的重要程度。    
 
例如：《中国的蜜蜂养殖》这本书，经过数据统计后，中国、蜜蜂、养殖这三个词的数量最多，那么谁能代表这篇文章的重要性呢？    
中国一般是比较常见的，那么是“蜜蜂”还是“养殖”能够代表这本书的特征呢？

$词频(TF) = \frac{某个词 t 在文章 d 中出现的次数}{文章 d 的总词数}$  
$逆文档频率(IDF) = log \frac{语料库 D 的文档总数}{包含词 t 的文档数量 + 1}$  
$ TF-IDF = TF \cdot IDF$，  

解释：当词频 TF 数量越大，且 IDF 越大， 则说明这个词汇的重要性程度越高；IDF 越大，就说明包含该词的文档数量越少、越稀有，就说明该词越珍贵！！

```python
def tfidf(data):
    tfidf_vevtorizer = TfidfVectorizer() # 它用于将文本转换为TF-IDF特征矩阵。
    '''
    fit_transform 方法对 data 进行学习（即根据文本数据中的词汇构建词汇表），
    然后将文本数据转换为TF-IDF特征矩阵。返回的 train 是一个稀疏矩阵，包含了文本数据的TF-IDF特征。
    '''
    train = tfidf_vevtorizer.fit_transform(data) 
    return train, tfidf_vevtorizer

# 对训练数据 X_train 进行TF-IDF特征提取，并得到相应的特征矩阵 X_train_tfidf
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
'''
使用相同的TF-IDF向量化器 tfidf_vectorizer 对测试数据 X_test 进行转换，得到测试数据的特征矩阵 X_test_tfidf。
这样可以确保训练数据和测试数据在相同的词汇表和转换方式下得到特征表示。
'''
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

```python
fig = plt.figure(figsize=(16, 16))
plot_LSA(X_train_tfidf, y_train)
plt.show()
```
![image](https://github.com/user-attachments/assets/8e4253c0-1f6d-465c-b819-d3f0dd42a06a)

**和词袋模型相比，TF-IDF的效果看上去会稍微好一点**

```python
clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver="newton-cg",
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf.fit(X_train_tfidf, y_train)

y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
```

```python
accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" % (accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))
```
accuracy: 0.773, precision: 0.773, recall: 0.773, f1: 0.773  

```python
cm = confusion_matrix(y_test, y_predicted_tfidf)
fig = plt.figure(figsize=(10,10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], title="Confusion Matrix")
plt.show()
```
![image](https://github.com/user-attachments/assets/5d8f8245-d06a-44a1-b47f-9b1cb48cb586)

```python
importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
```

```python
top_scores = [a[0] for a in importance_tfidf[0]['tops']]
top_words = [a[1] for a in importance_tfidf[0]['tops']]

bottom_scores = [a[0] for a in importance_tfidf[0]['bottom']]
bottom_words = [a[1] for a in importance_tfidf[0]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")
```
![image](https://github.com/user-attachments/assets/909b1970-1fdf-4f8e-a640-8ea13c589c9d)

**但是，以上方法都是考虑每一个基于频率的情况，如果在新的测试环境下有些词发生变化了怎么办？**  

**比如：有一些单词从来都没有在之前的文档里出现过**    
**或者**  
**类似于bad和worse的单词表达的意义差不多但是长得不一样**

**词频统计的方法就无法或者很难捕捉到这些特征**

### 3.4 word2vec
```python
import gensim

# https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/tree/main
word2vec_path = "GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
```

```python
def get_average_word2vec(token_list, vector, generate_missing=False, k=300):
    if len(token_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in token_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in token_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_question, generate_missing=False):
    embeddings = clean_question['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)
```

```python
embeddings = get_word2vec_embeddings(word2vec, clean_data)

X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
                                                                                        test_size=0.2, random_state=40)
```

```python
X_train_word2vec[2]
```
array([-1.67142428e-03,  5.19831731e-02, -4.97671274e-04,  9.55153245e-02,
       -1.56419020e-01, -1.42728365e-03,  4.40603403e-02, -9.99380258e-02,
        3.44191331e-02,  5.22085337e-02, -1.76591140e-02, -1.42493615e-01,
       -1.30690355e-01,  9.65951773e-02, -9.96610201e-02,  1.47047776e-01,
        1.26600999e-01,  1.73612154e-01,  3.83347731e-03, -3.90413724e-02,
       -8.86922983e-02, -6.96927584e-02,  8.54926476e-02, -4.00108924e-02,
        8.65689791e-02, -2.40478516e-02, -1.22312106e-01, -2.58166973e-02,
       -5.26310847e-03, -7.98105093e-02, -3.84005033e-02,  2.73719201e-02,
       -1.58835191e-02, -3.35364709e-02, -4.36589168e-02,  1.90523588e-02,
       -4.92351239e-02,  8.82943960e-02,  1.04041466e-01,  9.55528846e-02,
        1.38624925e-01, -2.42991814e-02,  1.56306340e-01,  4.38983624e-04,
        4.14804312e-02, -5.94012921e-02,  4.49523926e-02, -3.12969501e-02,
       -6.20962290e-02,  2.44891827e-02, -1.09238845e-01,  7.57915790e-02,
       -6.23497596e-03,  6.58176129e-02,  1.60287710e-02, -1.07046274e-02,
       -1.47141677e-02, -8.01579402e-02,  5.68002554e-02, -1.18774414e-01,
        7.04251803e-03,  1.09637921e-01, -6.26314603e-02, -5.73120117e-02,
        2.55995530e-02, -3.46773588e-02, -2.95973558e-02,  1.12135667e-01,
       -1.24771118e-01,  6.16924579e-02,  9.30880033e-02,  5.36452073e-02,
        3.64285983e-02, -9.43392240e-02, -1.51986929e-01, -8.02189754e-02,
        1.04764498e-01,  1.33020254e-01, -1.31695087e-02,  8.83225661e-02,
       -9.19893705e-02, -3.18861741e-02, -4.56859882e-02, -2.59164663e-02,
        5.09502704e-02, -8.04725060e-03, -1.69208233e-02,  1.70917218e-01,
        2.82639724e-02,  1.57552866e-02,  1.88504733e-02,  5.26281504e-02,
       -5.15717726e-02, -1.13628681e-01, -2.30806791e-02, -5.21756686e-02,
        7.92518029e-02,  5.03751315e-02,  8.04631160e-02,  2.58225661e-03,
       -4.53256460e-02,  6.01431040e-03, -3.62267127e-02,  5.01990685e-02,
       -9.67665452e-02,  3.82024325e-02, -3.22171725e-02, -1.56250000e-02,
        7.71484375e-02, -3.07218111e-02, -1.22483474e-01, -9.05879094e-02,
       -3.23814979e-02,  1.17351825e-02,  6.29841731e-02,  5.60044509e-02,
        3.09236967e-02, -3.36092435e-02,  2.68625113e-02,  7.35048147e-02,
       -4.62458684e-02,  2.31370192e-02, -6.49367112e-02,  6.74907978e-02,
        3.49872296e-02, -6.85644883e-02, -3.53346605e-02, -1.73715445e-04,
        4.26025391e-02, -2.24750225e-02,  1.23502291e-02, -5.67908654e-02,
       -6.55235877e-02, -1.71227088e-02, -4.63547340e-02, -3.40441190e-02,
        1.25098595e-02,  7.15989333e-03,  9.31161734e-02,  7.35567533e-02,
        1.33526142e-01, -1.33573092e-01,  8.24303260e-02,  8.55431190e-03,
        4.03864934e-02,  1.00848858e-02, -2.84517728e-02, -9.41678561e-02,
        2.37567608e-02, -8.44632662e-03,  3.24049730e-02, -1.96932279e-02,
       -1.09628531e-01, -5.23024339e-03, -6.00022536e-03,  2.62920673e-03,
       -8.03833008e-02,  4.89220252e-02, -9.26595835e-02,  3.14237154e-02,
       -4.43443885e-02,  5.74481671e-02,  1.29676232e-02, -1.17375300e-03,
        3.23333740e-02, -1.45014836e-01,  6.45094651e-03,  2.05735427e-02,
       -1.76062951e-03,  5.49081656e-03, -1.90129207e-01, -7.14956430e-02,
       -5.39926382e-02, -6.09036959e-02, -4.35791016e-02, -5.30771109e-02,
        9.03038612e-02, -4.26518367e-02,  2.79001089e-02,  3.19824219e-02,
       -2.84611629e-02, -1.90570538e-02, -9.09775954e-03,  5.87064303e-02,
       -3.32782452e-02,  1.26953125e-02, -3.67330404e-02,  1.74842248e-02,
       -2.76536208e-03,  9.14705717e-02,  6.97490986e-02,  2.19257061e-02,
       -9.62101863e-02,  1.88856858e-02,  1.33361816e-02, -5.05418044e-02,
       -2.40672185e-02, -7.64817458e-02, -7.65568660e-02, -9.56373948e-02,
       -3.26655461e-02,  6.78147536e-02, -9.52148438e-03, -7.49323918e-03,
        2.35126202e-02,  1.79349459e-02, -3.15692608e-02, -3.87056791e-02,
       -2.27391170e-02,  2.40196815e-02, -1.13912729e-03,  4.28372897e-02,
       -4.05555138e-02, -2.44985727e-02, -9.10175030e-02, -1.79842435e-02,
        1.35047326e-01, -5.96782978e-02, -1.57883864e-01,  4.77201022e-02,
       -1.91744291e-02,  3.54942909e-02, -6.94298377e-02, -2.39633413e-02,
        5.86876502e-02, -2.69024189e-03,  8.16826454e-02,  9.09893329e-02,
       -3.54121282e-02,  2.68930288e-02,  3.04729755e-02, -4.98985877e-02,
        9.03555063e-03,  7.77306190e-02,  9.14685176e-02, -4.23306685e-02,
        6.25751202e-02, -2.02073317e-02,  8.46604567e-02,  4.39805251e-03,
        5.88003305e-02,  2.05218975e-02,  7.59183444e-03, -6.61973220e-02,
        2.18881460e-02,  7.80193622e-03,  1.47000826e-02,  6.03355995e-02,
        4.22175481e-02,  1.35028546e-02,  7.33971229e-02,  3.12500000e-02,
        8.58927507e-02,  6.22136043e-02,  5.06403996e-02, -1.01662856e-01,
        4.24382136e-02,  6.73452524e-02, -2.33060397e-02, -4.59641677e-02,
        4.77670523e-02, -3.11654898e-02, -1.66485126e-02,  3.61738939e-02,
       -1.52587891e-03,  1.18326040e-01, -3.67901142e-02,  1.15039532e-02,
        9.46514423e-03, -6.43967849e-02,  3.15129207e-02,  4.30344802e-02,
        8.95104041e-02,  7.76461088e-02,  8.36111215e-02, -1.35286771e-02,
       -7.02655499e-02, -1.11309345e-01,  1.11553486e-02, -3.89216496e-03,
       -2.73061899e-02,  2.83766526e-02, -2.83437876e-02,  3.96728516e-03,
        6.20563214e-03,  3.27524038e-02, -5.62239427e-02,  2.61981671e-03,
        1.04135367e-02,  3.98137019e-03, -9.35058594e-02,  3.24472281e-02,
       -1.69800978e-02,  2.03904372e-02, -6.92702073e-02,  4.45087139e-03,
       -9.13813664e-02, -7.45286208e-02, -8.67550190e-03, -7.15519832e-03])

```python
len(X_train_word2vec[2])
```
300

```python
fig = plt.figure(figsize=(16,16))
plot_LSA(embeddings, list_labels)
plt.show()
```
![image](https://github.com/user-attachments/assets/4a400724-bd4e-4a75-8bb9-13b528928409)

**从可视化图上来看，似乎变得线性可分了。**

```python
clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver="newton-cg",
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_w2v.fit(X_train_word2vec, y_train_word2vec)

y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)
```

```python
accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)
print("accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f" % (accuracy_word2vec, 
                                                                   precision_word2vec, 
                                                                   recall_word2vec, 
                                                                   f1_word2vec))
```
accuracy: 0.778, precision: 0.777, recall: 0.778, f1: 0.777  

```python
cm = confusion_matrix(y_test_word2vec, y_predicted_word2vec)
fig = plt.figure(figsize=(10,10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant', 'Disaster', 'Unsure'], title="Confusion Matrix")
plt.show()
```
![image](https://github.com/user-attachments/assets/3b84a8af-7f6a-42ce-85cb-84ca9f02c146)

**词嵌入在分类的时候，是直接将一句话中的单词的词向量进行平均然后再进行分类，并没有考虑一句话中单词之间的序列（前后顺序）**

### 3.5 基于深度学习的词嵌入（CNN or RNN）
```python
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
```

```python
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 35
VOCAB_SIZE = len(VOCAB)
```

```python
VALIDATION_SPLIT = .2
# 基于词频，频率最高的单词会排在前面，将文本数据转换为整数序列
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(clean_data["text"].tolist())
sequences = tokenizer.texts_to_sequences(clean_data["text"].tolist())
```

```python
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))
```
Found 19097 unique tokens.  

```python
cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(clean_data["class_label"]))
```

```python
indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data = cnn_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
for word,index in word_index.items():
    embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(embedding_weights.shape)
```
(19098, 300)  

```python
from keras.layers import Dense, Input, Flatten, Dropout, Concatenate  # 使用 Concatenate 替代 Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding  # 注意 MaxPooling1D 的大小写
from keras.layers import LSTM, Bidirectional
from keras.models import Model  # Model 应该从 keras.models 导入
```

```python
def ConvNet(embeddings, max_sequence_length,
            num_words, embedding_dim, lables_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)
    sequences_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequences_input)

    convs = []
    filter_size = [3,4,5]
    
    for filter_size in filter_size:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)

    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv == True:
        x = Dropout(0.5)(l_merge)
    else:
        x = Dropout(0.5)(pool)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    preds = Dense(lables_index, activation='softmax')(x)

    model = Model(sequences_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    return model
```

```python
x_train = cnn_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = cnn_data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM,
                len(list(clean_data["class_label"].unique())),
                False)
```

```python
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=128)
```
