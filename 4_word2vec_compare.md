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

```python

```

```python

```

```python

```

```python

```

