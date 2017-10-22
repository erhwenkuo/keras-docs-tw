# 常用數據庫

## CIFAR10 小圖片分類數據集

該數據庫具有50,000個32*32的彩色圖片作為訓練集，10,000個圖片作為測試集。圖片一共有10個類別。

### 使用方法
```python
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

### 返回值：

兩個Tuple

```X_train```和```X_test```是形如（nb_samples, 3, 32, 32）的RGB三通道圖像數據，數據類型是無符號8位整形（uint8）

```Y_train```和 ```Y_test```是形如（nb_samples,）標籤數據，標籤的範圍是0~9

***

## CIFAR100 小圖片分類數據庫

該數據庫具有50,000個32*32的彩色圖片作為訓練集，10,000個圖片作為測試集。圖片一共有100個類別，每個類別有600張圖片。這100個類別又分為20個大類。

### 使用方法
```python
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
```

### 參數

* label_mode：為‘fine’或‘coarse’之一，控制標籤的精細度，‘fine’獲得的標籤是100個小類的標籤，‘coarse’獲得的標籤是大類的標籤

### 返回值

兩個Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：是形如（nb_samples, 3, 32, 32）的RGB三通道圖像數據，數據類型是無符號8位整形（uint8）

* y_train和y_test：是形如（nb_samples,）標籤數據，標籤的範圍是0~9

***

## IMDB影評傾向分類

本數據庫含有來自IMDB的25,000條影評，被標記為正面/負面兩種評價。影評已被預處理為詞下標構成的[<font color='#FF0000'>序列</font>](../preprocessing/sequence)。方便起見，單詞的下標基於它在數據集中出現的頻率標定，例如整數3所編碼的詞為數據集中第3常出現的詞。這樣的組織方法使得用戶可以快速完成諸如“只考慮最常出現的10,000個詞，但不考慮最常出現的20個詞”這樣的操作

按照慣例，0不代表任何特定的詞，而用來編碼任何未知單詞

### 使用方法
```python
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      test_split=0.1)
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```

### 參數

* path：如果你在本機上已有此數據集（位於```'~/.keras/datasets/'+path```），則載入。否則數據將下載到該目錄下

* nb_words：整數或None，要考慮的最常見的單詞數，序列中任何出現頻率更低的單詞將會被編碼為`oov_char`的值。

* skip_top：整數，忽略最常出現的若干單詞，這些單詞將會被編碼為`oov_char`的值

* maxlen：整數，最大序列長度，任何長度大於此值的序列將被截斷

* seed：整數，用於數據重排的隨機數種子

* start_char：字符，序列的起始將以該字符標記，預設為1因為0通常用作padding

* oov_char：整數，因```nb_words```或```skip_top```限製而cut掉的單詞將被該字符代替

* index_from：整數，真實的單詞（而不是類似於```start_char```的特殊佔位符）將從這個下標開始

### 返回值

兩個Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：序列的列表，每個序列都是詞下標的列表。如果指定了```nb_words```，則序列中可能的最大下標為```nb_words-1```。如果指定了```maxlen```，則序列的最大可能長度為```maxlen```

* y_train和y_test：為序列的標籤，是一個二值list

***

## 路透社新聞主題分類

本數據庫包含來自路透社的11,228條新聞，分為了46個主題。與IMDB庫一樣，每條新聞被編碼為一個詞下標的序列。

### 使用方法
```python
from keras.datasets import reuters


(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         nb_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

參數的含義與IMDB同名參數相同，唯一多的參數是：
```test_split```，用於指定從原數據中分割出作為測試集的比例。該數據庫支持獲取用於編碼序列的詞下標：
```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```
上面代碼的返回值是一個以單詞為關鍵字，以其下標為值的字典。例如，```word_index['giraffe']```的值可能為```1234```

### 參數

* path：如果你在本機上已有此數據集（位於```'~/.keras/datasets/'+path```），則載入。否則數據將下載到該目錄下

***

## MNIST手寫數字識別

本數據庫有60,000個用於訓練的28*28的灰度手寫數字圖片，10,000個測試圖片

### 使用方法
```python
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
### 參數

* path：如果你在本機上已有此數據集（位於```'~/.keras/datasets/'+path```），則載入。否則數據將下載到該目錄下

### 返回值

兩個Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：是形如（nb_samples, 28, 28）的灰度圖片數據，數據類型是無符號8位整形（uint8）

* y_train和y_test：是形如（nb_samples,）標籤數據，標籤的範圍是0~9

數據庫將會被下載到```'~/.keras/datasets/'+path```

***

## Boston房屋價格回歸數據集

本數據集由StatLib庫取得，由CMU維護。每個樣本都是1970s晚期波士頓郊區的不同位置，每條數據含有13個屬性，目標值是該位置房子的房價中位數（千dollar）。


### 使用方法
```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

### 參數

* path：數據存放位置，預設```'~/.keras/datasets/'+path```

* seed：隨機數種子

* test_split：分割測試集的比例

### 返回值

兩個Tuple,```(X_train, y_train), (X_test, y_test)```