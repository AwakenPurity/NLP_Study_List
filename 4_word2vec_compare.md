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

数据清洗，去掉无用字符
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

查看数据分布情况: 是否不平衡  
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

分词、数据集划分  
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

语料库情况  
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

```

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

