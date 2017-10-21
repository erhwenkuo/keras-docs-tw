#Merge層

Merge層提供了一系列用於融合兩個層或兩個張量的層對象和方法。以大寫首字母開頭的是Layer類，以小寫字母開頭的是張量的函數。小寫字母開頭的張量函數在內部實際上是調用了大寫字母開頭的層。

## Add
```python
keras.layers.merge.Add()
```
將Layer that adds a list of inputs.

該層接收一個列表的同shape張量，並返回它們的和，shape不變。

## Multiply
``python
keras.layers.merge.Multiply()
```
該層接收一個列表的同shape張量，並返回它們的逐元素積的張量，shape不變。

## Average
```python
keras.layers.merge.Average()
```
該層接收一個列表的同shape張量，並返回它們的逐元素均值，shape不變。


## Maximum
```python
keras.layers.merge.Maximum()
```
該層接收一個列表的同shape張量，並返回它們的逐元素最大值，shape不變。

## Concatenate
```python
keras.layers.merge.Concatenate(axis=-1)
```
該層接收一個列表的同shape張量，並返回它們的按照給定軸相接構成的向量。

### 參數

* axis: 想接的軸
* **kwargs: 普通的Layer關鍵字參數

## Dot
```python
keras.layers.merge.Dot(axes, normalize=False)
```
計算兩個tensor中樣本的張量乘積。例如，如果兩個張量```a```和```b```的shape都為（batch_size, n），則輸出為形如（batch_size,1）的張量，結果張量每個batch的數據都是a[i,:]和b[i,:]的矩陣（向量）點積。


### 參數

* axes: 整數或整數的tuple，執行乘法的軸。
* normalize: 布爾值，是否沿執行成績的軸做L2規範化，如果設為True，那麼乘積的輸出是兩個樣本的餘弦相似性。
* **kwargs: 普通的Layer關鍵字參數


## add
```python
add(inputs)
```
Add層的函數式包裝

###參數：

* inputs: 長度至少為2的張量列表A
* **kwargs: 普通的Layer關鍵字參數
###返回值

輸入列表張量之和

## multiply
``python
multiply(inputs)
```
Multiply的函數包裝

###參數：

* inputs: 長度至少為2的張量列表
* **kwargs: 普通的Layer關鍵字參數
###返回值

輸入列表張量之逐元素積

## average
```python
average(inputs)
```
Average的函數包裝

###參數：

* inputs: 長度至少為2的張量列表
* **kwargs: 普通的Layer關鍵字參數
###返回值

輸入列表張量之逐元素均值

## maximum
```python
maximum(inputs)
```
Maximum的函數包裝

###參數：

* inputs: 長度至少為2的張量列表
* **kwargs: 普通的Layer關鍵字參數
###返回值

輸入列表張量之逐元素均值


## concatenate
```python
concatenate(inputs, axis=-1))
```
Concatenate的函數包裝

### 參數
* inputs: 長度至少為2的張量列
* axis: 相接的軸
* **kwargs: 普通的Layer關鍵字參數

## dot
```python
dot(inputs, axes, normalize=False)
```
Dot的函數包裝


### 參數
* inputs: 長度至少為2的張量列
* axes: 整數或整數的tuple，執行乘法的軸。
* normalize: 布爾值，是否沿執行成績的軸做L2規範化，如果設為True，那麼乘積的輸出是兩個樣本的餘弦相似性。
* **kwargs: 普通的Layer關鍵字參數