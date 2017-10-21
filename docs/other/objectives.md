# 目標函數objectives

目標函數，或稱損失函數，是編譯一個模型必須的兩個參數之一：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

可以通過傳遞預定義目標函數名字指定目標函數，也可以傳遞一個Theano/TensroFlow的符號函數作為目標函數，該函數對每個數據點應該只返回一個標量值，並以下列兩個參數為參數：

* y_true：真實的數據標籤，Theano/TensorFlow張量

* y_pred：預測值，與y_true相同shape的Theano/TensorFlow張量
```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```
真實的優化目標函數是在各個數據點得到的損失函數值之和的均值

請參考[目標實現代碼](https://github.com/fchollet/keras/blob/master/keras/objectives.py)獲取更多信息

## 可用的目標函數

* mean_squared_error或mse

* mean_absolute_error或mae

* mean_absolute_percentage_error或mape

* mean_squared_logarithmic_error或msle

* squared_hinge

* hinge

* categorical_hinge

* binary_crossentropy（亦稱作對數損失，logloss）

* logcosh

* categorical_crossentropy：亦稱作多類的對數損失，注意使用該目標函數時，需要將標籤轉化為形如```(nb_samples, nb_classes)```的二值序列

* sparse_categorical_crossentrop：如上，但接受稀疏標籤。注意，使用該函數時仍然需要你的標籤與輸出值的維度相同，你可能需要在標籤數據上增加一個維度：```np.expand_dims(y,-1)```

* kullback_leibler_divergence:從預測值概率分佈Q到真值概率分佈P的信息增益,用以度量兩個分佈的差異.

* poisson：即```(predictions - targets * log(predictions))```的均值

* cosine_proximity：即預測值與真實標籤的餘弦距離平均值的相反數


**注意**: 當使用"categorical_crossentropy"作為目標函數時,標籤應該為多類模式,即one-hot編碼的向量,而不是單個數值. 可以使用工具中的`to_categorical`函數完成該轉換.示例如下:
```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```