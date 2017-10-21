# 文本預處理

## 句子分割text_to_word_sequence
```python
keras.preprocessing.text.text_to_word_sequence(text,
                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
```
本函數將一個句子拆分成單詞構成的列表

### 參數

* text：字符串，待處理的文本

* filters：需要濾除的字符的列表或連接形成的字符串，例如標點符號。預設值為'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'，包含標點符號，製表符和換行符等

* lower：布爾值，是否將序列設為小寫形式

* split：字符串，單詞的分隔符，如空格

### 返回值

字符串列表

***

## one-hot編碼
```python
keras.preprocessing.text.one_hot(text,
                                 n,
                                 filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" ")
```
本函數將一段文本編碼為one-hot形式的碼，即僅記錄詞在詞典中的下標。


【Tips】
從定義上，當字典長為n時，每個單詞應形成一個長為n的向量，其中僅有單詞本身在字典中下標的位置為1，其餘均為0，這稱為one-hot。

為了方便起見，函數在這裡僅把“1”的位置，即字典中詞的下標記錄下來。

### 參數

* n：整數，字典長度

### 返回值

整數列表，每個整數是\[1,n\]之間的值，代表一個單詞（不保證唯一性，即如果詞典長度不夠，不同的單詞可能會被編為同一個碼）。

***

## 特徵哈希hashing_trick
```python
keras.preprocessing.text.hashing_trick(text,
                                       n,
                                       hash_function=None,
                                       filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                       lower=True,
                                       split=' ')
```
將文本轉換為固定大小的哈希空間中的索引序列

### 參數

* n: 哈希空間的維度

* hash_function: 預設為python `hash` 函數, 可以是'md5' 或任何接受輸入字符串, 並返回int 的函數. 注意`hash` 不是一個穩定的哈希函數, 因此在不同執行環境下會產生不同的結果, 作為對比, 'md5' 是一個穩定的哈希函數.

### 返回值

整數列表

## 分詞器Tokenizer
```python
keras.preprocessing.text.Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
```
Tokenizer是一個用於向量化文本，或將文本轉換為序列（即單詞在字典中的下標構成的列表，從1算起）的類。

### 構造參數

* 與```text_to_word_sequence```同名參數含義相同

* num_words：None或整數，處理的最大單詞數量。若被設置為整數，則分詞器將被限制為待處理數據集中最常見的```num_words```個單詞

* char_level: 如果為 True, 每個字符將被視為一個標記

### 類方法

* fit_on_texts(texts)

* texts：要用以訓練的文本列表

* texts_to_sequences(texts)

* texts：待轉為序列的文本列表

* 返回值：序列的列表，列表中每個序列對應於一段輸入文本

* texts_to_sequences_generator(texts)

* 本函數是```texts_to_sequences```的生成器函數版

* texts：待轉為序列的文本列表

* 返回值：每次調用返回對應於一段輸入文本的序列

* texts_to_matrix(texts, mode)：

* texts：待向量化的文本列表

* mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，預設為‘binary’

* 返回值：形如```(len(texts), nb_words)```的numpy array

* fit_on_sequences(sequences):

* sequences：要用以訓練的序列列表

* sequences_to_matrix(sequences):

* sequences：待向量化的序列列表

* mode：‘binary’，‘count’，‘tfidf’，‘freq’之一，預設為‘binary’

* 返回值：形如```(len(sequences), nb_words)```的numpy array

### 屬性
* word_counts:字典，將單詞（字符串）映射為它們在訓練期間出現的次數。僅在調用fit_on_texts之後設置。
* word_docs: 字典，將單詞（字符串）映射為它們在訓練期間所出現的文檔或文本的數量。僅在調用fit_on_texts之後設置。
* word_index: 字典，將單詞（字符串）映射為它們的排名或者索引。僅在調用fit_on_texts之後設置。
* document_count: 整數。分詞器被訓練的文檔（文本或者序列）數量。僅在調用fit_on_texts或fit_on_sequences之後設置。