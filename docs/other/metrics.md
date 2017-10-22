# 性能評估

## 使用方法

性能評估模塊提供了一系列用於模型性能評估的函數,這些函數在模型編譯時由`metrics`關鍵字設置

性能評估函數類似與[目標函數](objectives.md), 只不過該性能的評估結果講不會用於訓練.

可以通過字符串來使用域定義的性能評估函數
```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```
也可以自定義一個Theano/TensorFlow函數並使用之
```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```
### 參數

* y_true:真實標籤,theano/tensorflow張量
* y_pred:預測值, 與y_true形式相同的theano/tensorflow張量

### 返回值

單個用以代表輸出各個數據點上均值的值

## 可用預定義張量

除fbeta_score額外擁有預設參數beta=1外,其他各個性能指標的參數均為y_true和y_pred

* binary_accuracy: 對二分類問題,計算在所有預測值上的平均正確率
* categorical_accuracy:對多分類問題,計算再所有預測值上的平均正確率
* sparse_categorical_accuracy:與`categorical_accuracy`相同,在對稀疏的目標值預測時有用
* top_k_categorical_accracy: 計算top-k正確率,當預測值的前k個值中存在目標類別即認為預測正確
* sparse_top_k_categorical_accuracy：與top_k_categorical_accracy作用相同，但適用於稀疏情況

## 定制評估函數

定制的評估函數可以在模型編譯時傳入,該函數應該以`(y_true, y_pred)`為參數,並返回單個張量,或從`metric_name`映射到`metric_value`的字典,下面是一個示例:

```python
(y_true, y_pred) as arguments and return a single tensor value.

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

```