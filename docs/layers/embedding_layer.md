# 嵌入層 Embedding

## Embedding層

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```
嵌入層將正整數（下標）轉換為具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]

Embedding層只能作為模型的第一層

### 參數

* input_dim：大或等於0的整數，字典長度，即輸入數據最大下標+1

* output_dim：大於0的整數，代表全連接嵌入的維度

* embeddings_initializer: 嵌入矩陣的初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* embeddings_regularizer: 嵌入矩陣的正則項，為[Regularizer](../other/regularizers)對象

* embeddings_constraint: 嵌入矩陣的約束項，為[Constraints](../other/constraints)對象

* mask_zero：布爾值，確定是否將輸入中的‘0’看作是應該被忽略的‘填充’（padding）值，該參數在使用[遞歸層](recurrent_layer)處理變長輸入時有用。設置為```True```的話，模型中後續的層必須都支持masking，否則會拋出異常。如果該值為True，則下標0在字典中不可用，input_dim應設置為|vocabulary| + 1。

* input_length：當輸入序列的長度固定時，該值為其長度。如果要在該層後接```Flatten```層，然後接```Dense```層，則必須指定該參數，否則```Dense```層的輸出維度無法自動推斷。


### 輸入shape

形如（samples，sequence_length）的2D張量

### 輸出shape

形如(samples, sequence_length, output_dim)的3D張量

### 例子
```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

### 參考文獻

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)