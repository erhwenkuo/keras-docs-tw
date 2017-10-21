# 包裝器Wrapper

## TimeDistributed包裝器
```python
keras.layers.wrappers.TimeDistributed(layer)
```
該包裝器可以把一個層應用到輸入的每一個時間步上

### 參數

* layer：Keras層對象

輸入至少為3D張量，下標為1的維度將被認為是時間維

例如，考慮一個含有32個樣本的batch，每個樣本都是10個向量組成的序列，每個向量長為16，則其輸入維度為```(32,10,16)```，其不包含batch大小的```input_shape```為```(10,16)```

我們可以使用包裝器```TimeDistributed```包裝```Dense```，以產生針對各個時間步信號的獨立全連接：

```python
# as the first layer in a model
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

# subsequent layers: no need for input_shape
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

程序的輸出數據shape為```(32,10,8)```

使用```TimeDistributed```包裝```Dense```嚴格等價於```layers.TimeDistribuedDense```。不同的是包裝器```TimeDistribued```還可以對別的層進行包裝，如這裡對```Convolution2D```包裝：

```python
model = Sequential()
model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
```

## Bidirectional包裝器
```python
keras.layers.wrappers.Bidirectional(layer, merge_mode='concat', weights=None)
```
雙向RNN包裝器

### 參數

* layer：```Recurrent```對象
* merge_mode：前向和後向RNN輸出的結合方式，為```sum```,```mul```,```concat```,```ave```和``` None```之一，若設為None，則返回值不結合，而是以列表的形式返回

### 例子
```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```