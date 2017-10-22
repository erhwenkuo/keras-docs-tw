# 序列預處理

## 填充序列pad_sequences
```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='pre', value=0.)
```
將長為```nb_samples```的序列（標量序列）轉化為形如```(nb_samples,nb_timesteps)```2D numpy array。如果提供了參數```maxlen```，```nb_timesteps=maxlen```，否則其值為最長序列的長度。其他短於該長度的序列都會在後部填充0以達到該長度。長於`nb_timesteps`的序列將會被截斷，以使其匹配目標長度。 padding和截斷發生的位置分別取決於`padding`和`truncating`.

### 參數

* sequences：浮點數或整數構​​成的兩層嵌套列表

* maxlen：None或整數，為序列的最大長度。大於此長度的序列將被截短，小於此長度的序列將在後部填0.

* dtype：返回的numpy array的數據類型

* padding：‘pre’或‘post’，確定當需要補0時，在序列的起始還是結尾補

* truncating：‘pre’或‘post’，確定當需要截斷序列時，從起始還是結尾截斷

* value：浮點數，此值將在填充時代替預設的填充值0

### 返回值

返回形如```(nb_samples,nb_timesteps)```的2D張量

***

## 跳字skipgrams
```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size,
    window_size=4, negative_samples=1., shuffle=True,
    categorical=False, sampling_table=None)
```
skipgrams將一個詞向量下標的序列轉化為下面的一對tuple：

* 對於正樣本，轉化為（word，word in the same window）

* 對於負樣本，轉化為（word，random word from the vocabulary）

【Tips】根據維基百科，n-gram代表在給定序列中產生連續的n項，當序列句子時，每項就是單詞，此時n-gram也稱為shingles。而skip-gram的推廣，skip-gram產生的n項子序列中，各個項在原序列中不連續，而是跳了k個字。例如，對於句子：

“the rain in Spain falls mainly on the plain”

其 2-grams為子序列集合：

the rain，rain in，in Spain，Spain falls，falls mainly，mainly on，on the，the plain

其 1-skip-2-grams為子序列集合：

the in, rain Spain, in falls, Spain mainly, falls on, mainly the, on plain.

更多詳情請參考[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

### 參數

* sequence：下標的列表，如果使用sampling_tabel，則某個詞的下標應該為它在數據庫中的順序。 （從1開始）

* vocabulary_size：整數，字典大小

* window_size：整數，正樣本對之間的最大距離

* negative_samples：大於0的浮點數，等於0代表沒有負樣本，等於1代表負樣本與正樣本數目相同，以此類推（即負樣本的數目是正樣本的```negative_samples```倍）

* shuffle：布爾值，確定是否隨機打亂樣本

* categorical：布爾值，確定是否要使得返回的標籤具有確定類別

* sampling_table：形如```(vocabulary_size,)```的numpy array，其中```sampling_table[i]```代表沒有負樣本或隨機負樣本。等於1為與正樣本的數目相同
採樣到該下標為i的單詞的概率（假定該單詞是數據庫中第i常見的單詞）

### 輸出

函數的輸出是一個```(couples,labels)```的元組，其中：

* ```couples```是一個長為2的整數列表：```[word_index,other_word_index]```

* ```labels```是一個僅由0和1構成的列表，1代表```other_word_index```在```word_index```的窗口，0代表```other_word_index```是詞典裡的隨機單詞。

* 如果設置```categorical```為```True```，則標籤將以one-hot的方式給出，即1變為\[0,1\]，0變為\[1, 0\]

***

## 獲取採樣表make_sampling_table
```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-5)
```
該函數用以產生```skipgrams```中所需要的參數```sampling_table```。這是一個長為```size```的向量，```sampling_table[i]```代表採樣到數據集中第i常見的詞的概率（為平衡期起見，對於越經常出現的詞，要以越低的概率採到它）

### 參數

* size：詞典的大小

* sampling_factor：此值越低，則代表採樣時更緩慢的概率衰減（即常用的詞會被以更低的概率被採到），如果設置為1，則代表不進行下採樣，即所有樣本被採樣到的概率都是1。