# 常用層

常用層對應於core模塊，core內部定義了一系列常用的網絡層，包括全連接、激活層等

## Dense層
```python
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
Dense就是常用的全連接層，所實現的運算是```output = activation(dot(input, kernel)+bias)```。其中```activation```是逐元素計算的激活函數，```kernel```是本層的權值矩陣，```bias```為偏置向量，只有當```use_bias= True```才會添加。

如果本層的輸入數據的維度大於2，則會先被壓為與```kernel```相匹配的大小。

這裡是一個使用示例：

```python
# as first layer in a sequential model:
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

### 參數：

* units：大於0的整數，代表該層的輸出維度。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* use_bias: 布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象


### 輸入

形如(batch_size, ..., input_dim)的nD張量，最常見的情況為(batch_size, input_dim)的2D張量

### 輸出

形如(batch_size, ..., units)的nD張量，最常見的情況為(batch_size, units)的2D張量

***

<a name='activation'>
<font color='#404040'>
## Activation層
</font></a>
```python
keras.layers.core.Activation(activation)
```
激活層對一個層的輸出施加激活函數

### 參數

* activation：將要使用的激活函數，為預定義激活函數名或一個Tensorflow/Theano的函數。參考[激活函數](../other/activations)

### 輸入shape

任意，當使用激活層作為第一層時，要指定```input_shape```

### 輸出shape

與輸入shape相同

***
</a name='dropout'>
<font color='#404040'>
## Dropout層
</font></a>
```python
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
```
為輸入數據施加Dropout。 Dropout將在訓練過程中每次更新參數時按一定概率（rate）隨機斷開輸入神經元，Dropout層用於防止過擬合。

### 參數

* rate：0~1的浮點數，控制需要斷開的神經元的比例

* noise_shape：整數張量，為將要應用在輸入上的二值Dropout mask的shape，例如你的輸入為(batch_size, timesteps, features)，並且你希望在各個時間步上的Dropout mask都相同，則可傳入noise_shape=(batch_size, 1, features)。

* seed：整數，使用的隨機數種子

### 參考文獻

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

***


## Flatten層
```python
keras.layers.core.Flatten()
```
Flatten層用來將輸入“壓平”，即把多維的輸入一維化，常用在從卷積層到全連接層的過渡。 Flatten不影響batch的大小。

### 例子
```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

***

## Reshape層
```python
keras.layers.core.Reshape(target_shape)
```
Reshape層用來將輸入shape轉換為特定的shape

### 參數

* target_shape：目標shape，為整數的tuple，不包含樣本數目的維度（batch大小）

### 輸入shape

任意，但輸入的shape必須固定。當使用該層為模型首層時，需要指定```input_shape```參數

### 輸出shape

```(batch_size,)+target_shape```

### 例子

```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
```

***

## Permute層
```python
keras.layers.core.Permute(dims)
```
Permute層將輸入的維度按照給定模式進行重排，例如，當需要將RNN和CNN網絡連接時，可能會用到該層。

### 參數

* dims：整數tuple，指定重排的模式，不包含樣本數的維度。重拍模式的下標從1開始。例如（2，1）代表將輸入的第二個維度重拍到輸出的第一個維度，而將輸入的第一個維度重排到第二個維度

### 例子

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

### 輸入shape

任意，當使用激活層作為第一層時，要指定```input_shape```

### 輸出shape

與輸入相同，但是其維度按照指定的模式重新排列

***

## RepeatVector層
```python
keras.layers.core.RepeatVector(n)
```
RepeatVector層將輸入重複n次

### 參數

* n：整數，重複的次數

### 輸入shape

形如（nb_samples, features）的2D張量

### 輸出shape

形如（nb_samples, n, features）的3D張量

### 例子
```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)

```

***
## Lambda層
```python
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
```
本函數用以對上一層的輸出施以任何Theano/TensorFlow表達式

### 參數

* function：要實現的函數，該函數僅接受一個變量，即上一層的輸出

* output_shape：函數應該返回的值的shape，可以是一個tuple，也可以是一個根據輸入shape計算輸出shape的函數

* mask: 掩膜

* arguments：可選，字典，用來記錄向函數中傳遞的其他關鍵字參數

### 例子
```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```

```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2 # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
         output_shape=antirectifier_output_shape))
```
### 輸入shape

任意，當使用該層作為第一層時，要指定```input_shape```

### 輸出shape

由```output_shape```參數指定的輸出shape，當使用tensorflow時可自動推斷

***

## ActivityRegularizer層
```python
keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0)
```

經過本層的數據不會有任何變化，但會基於其激活值更新損失函數值

### 參數

* l1：1範數正則因子（正浮點數）

* l2：2範數正則因子（正浮點數）

### 輸入shape

任意，當使用該層作為第一層時，要指定```input_shape```

### 輸出shape

與輸入shape相同

***

## Masking層
```python
keras.layers.core.Masking(mask_value=0.0)
```

使用給定的值對輸入的序列信號進行“屏蔽”，用以定位需要跳過的時間步

對於輸入張量的時間步，即輸入張量的第1維度（維度從0開始算，見例子），如果輸入張量在該時間步上都等於```mask_value```，則該時間步將在模型接下來的所有層（只要支持masking）被跳過（屏蔽）。

如果模型接下來的一些層不支持masking，卻接受到masking過的數據，則拋出異常。

### 例子

考慮輸入數據```x```是一個形如(samples,timesteps,features)的張量，現將其送入LSTM層。因為你缺少時間步為3和5的信號，所以你希望將其掩蓋。這時候應該：

* 賦值```x[:,3,:] = 0.```，```x[:,5,:] = 0.```

* 在LSTM層之前插入```mask_value=0.```的```Masking```層

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```