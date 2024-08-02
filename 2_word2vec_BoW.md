
## 2. Word Embedding之词袋模型分类任务（Bag of Words

词袋模型（Bag of Words, BoW）是将文本转换为词的向量最基本的方法之一。  

其主要通过统计每个单词在文本中出现的次数，将文本转换为固定长度的向量表示，其中每个向量元素对应词汇表中的一个单词。

### 2.1. 主要概念
`词汇表`：词袋模型会为整个语料库中的所有不同单词创建一个词汇表。  
`词频`：词袋模型会计算每个单词在每个文档中出现的次数。  

### 2.2. 主要步骤
2.1 文本预处理：  
- 去除标点符号和特殊字符
- 转换为小写
- 分词（将文本拆分为单词）
- 去除停用词（如 "the", "is", "in" 等）

2.2 创建词汇表：  
对整个语料库中的所有文档进行分词，创建一个包含所有不同单词的词汇表。  

### 2.3. 示例

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

### 2.4. 基于词袋模型来实验一个分类任务  

1. 数据集准备  
[labeledTrainData.tsv](https://ww0.lanzout.com/iRXun26aiihc): 包含电影评论及其对应的情感标签（如正面或负面）  
[stopwords.txt](https://ww0.lanzout.com/iASta26aiite): 常用的停用词表，包含一组停用词，这些词在文本处理中会被忽略，因为它们对于文本的主题或意义贡献不大

2. 代码
```python
import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.corpus import stopwords
```
```python
nltk.download()
```
```python
df = pd.read_csv('./labeledTrainData.tsv', sep='\t', escapechar='\\')
print(len(df))
```
25000
```python
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

对影评数据进行数据预处理：   
1. 因为这些数据是从网上爬取的，需要去掉一些html标签  
2. 移除标点符号  
3. 分词 token  
4. 去掉停用词  
5. 重组为新的句子  
```python
df['review'][999]
```
'This move was on TV last night. I guess as a time filler, because it sucked bad! The movie is just an excuse to show some tits and ass at the start and somewhere about half way. (Not bad tits and ass though). But the story is too ridiculous for words. The "wolf", if that is what you can call it, is hardly shown fully save his teeth. When it is fully in view, you can clearly see they had some interns working on the CGI, because the wolf runs like he\'s running in a treadmill, and the CGI fur looks like it\'s been waxed, all shiny :)<br /><br />The movie is full of gore and blood, and you can easily spot who is going to get killed/slashed/eaten next. Even if you like these kind of splatter movies you will be disappointed, they didn\'t do a good job at it.<br /><br />Don\'t even get me started on the actors... Very corny lines and the girls scream at everything about every 5 seconds. But then again, if someone asked me to do bad acting just to give me a few bucks, then hey, where do I sign up?<br /><br />Overall boring and laughable horror.'  

```python
# 去掉HTML标签的数据
example = BeautifulSoup(df['review'][999], 'html.parser').get_text()
example
```
'This move was on TV last night. I guess as a time filler, because it sucked bad! The movie is just an excuse to show some tits and ass at the start and somewhere about half way. (Not bad tits and ass though). But the story is too ridiculous for words. The "wolf", if that is what you can call it, is hardly shown fully save his teeth. When it is fully in view, you can clearly see they had some interns working on the CGI, because the wolf runs like he\'s running in a treadmill, and the CGI fur looks like it\'s been waxed, all shiny :)The movie is full of gore and blood, and you can easily spot who is going to get killed/slashed/eaten next. Even if you like these kind of splatter movies you will be disappointed, they didn\'t do a good job at it.Don\'t even get me started on the actors... Very corny lines and the girls scream at everything about every 5 seconds. But then again, if someone asked me to do bad acting just to give me a few bucks, then hey, where do I sign up?Overall boring and laughable horror.'  

```python
# 去掉标点符号
example_letters = re.sub(r'[^a-zA-Z]', ' ',example)
example_letters
```
'This move was on TV last night  I guess as a time filler  because it sucked bad  The movie is just an excuse to show some tits and ass at the start and somewhere about half way   Not bad tits and ass though   But the story is too ridiculous for words  The  wolf   if that is what you can call it  is hardly shown fully save his teeth  When it is fully in view  you can clearly see they had some interns working on the CGI  because the wolf runs like he s running in a treadmill  and the CGI fur looks like it s been waxed  all shiny   The movie is full of gore and blood  and you can easily spot who is going to get killed slashed eaten next  Even if you like these kind of splatter movies you will be disappointed  they didn t do a good job at it Don t even get me started on the actors    Very corny lines and the girls scream at everything about every   seconds  But then again  if someone asked me to do bad acting just to give me a few bucks  then hey  where do I sign up Overall boring and laughable horror '  

```python
# 都转为小写
words = example_letters.lower().split()
words
```
['this',
 'move',
 'was',
 'on',
 'tv',
 'last',
 'night',
 'i',
 'guess',
 'as',
 'a',
...
 'sign',
 'up',
 'overall',
 'boring',
 'and',
 'laughable',
 'horror']  
  
 去除停用词  

首先需要使用 `nltk.download()` 检查停用词库 `stopwords` 是否下载, 若显示 `installed`, 代表已经安装完成。  

停用词，可以使用nltk中自带的停用词表，也可以使用网上的一些停用词库；    

这里我们以网上的一个停用词库为例。  

```python
# nltk.download()
```
```python
# 去停用词
stopwords = {}.fromkeys([ line.rstrip() for line in open('./stopwords.txt', encoding='utf-8')])
words_nostop = [w for w in words if w not in stopwords]
```
```python
# 将以上方法写成一个函数

'''
这是前面创建的字典，其中每个键是一个停用词，值是 None。
set() 函数将字典的键转换为一个集合。集合是一种无序且不重复的数据结构。
当你将一个字典传递给 set() 函数时，实际上只会使用字典的键来创建集合。
'''
eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]

    return ' '.join(words)
```
```python
df['review'][1000]
```
"I watched this movie really late last night and usually if it's late then I'm pretty forgiving of movies. Although I tried, I just could not stand this movie at all, it kept getting worse and worse as the movie went on. Although I know it's suppose to be a comedy but I didn't find it very funny. It was also an especially unrealistic, and jaded portrayal of rural life. In case this is what any of you think country life is like, it's definitely not. I do have to agree that some of the guy cast members were cute, but the french guy was really fake. I do have to agree that it tried to have a good lesson in the story, but overall my recommendation is that no one over 8 watch it, it's just too annoying."  
```python
clean_text(df['review'][1000])
```
'i watched movie late night s late i m pretty forgiving movies i i stand movie worse worse movie i s suppose a comedy i didn t find funny unrealistic jaded portrayal rural life case country life s i agree guy cast members cute french guy fake i agree a good lesson story recommendation watch s annoying'   

清洗数据到dataframe中  
```python
df['clean_review'] = df.review.apply(clean_text)
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

抽取bag of words特征（用sklearn的CountVectorizer）  

用于将文本数据转换为特征向量，适合于自然语言处理（NLP）和机器学习任务。   
具体来说，CountVectorizer 可以将文本数据转化为词频矩阵（Bag of Words模型），其中每一行表示一个文本样本，每一列表示一个特征（单词或n-gram），单元格的值表示该特征在该样本中出现的次数。  

简单地说，CountVectorizer主要是通过对词频进行统计来将token进行向量化。  
```python
vectorizer = CountVectorizer(max_features=5000)
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
train_data_features.shape
```
```python
from sklearn.model_selection import train_test_split

'''
train_data_features: 这是特征数据集，通常是一个数组或矩阵，每一行表示一个样本，每一列表示一个特征。在这个例子中，train_data_features 包含所有样本的特征。

df.sentiment:这是标签数据集，通常是一个数组或列表，每个元素表示对应样本的标签。在这个例子中，df.sentiment 包含所有样本的标签。

test_size=0.2: 测试集所占比例，可以是浮点数（0 到 1 之间）或整数。这里的 0.2 表示 20% 的数据用于测试集，剩余的 80% 用于训练集。

random_state=0:随机数种子，用于确保结果的可重复性。相同的 random_state 会产生相同的划分。设置 random_state 可以使得每次运行代码时，数据集划分方式相同，从而便于调试和结果复现
'''

X_train, X_test, y_train, y_test = train_test_split(train_data_features, df.sentiment, test_size=0.2, random_state=0)
```
```python
import matplotlib.pyplot as plt
import itertools

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
Recall metric in the testing data:  0.8531810766721044  
acc metric in the testing data:  0.8492  
![image](https://github.com/user-attachments/assets/e050c61e-c86a-4257-aec9-515551162e59)


