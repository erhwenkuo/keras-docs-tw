# 初始化方法

初始化方法定義了對Keras層設置初始化權重的方法

不同的層可能使用不同的關鍵字來傳遞初始化方法，一般來說指定初始化方法的關鍵字是```kernel_initializer``` 和 ```bias_initializer```，例如：
```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

一個初始化器可以由字符串指定（必須是下面的預定義初始化器之一），或一個callable的函數，例如
```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```

## Initializer

Initializer是所有初始化方法的父類，不能直接使用，如果想要定義自己的初始化方法，請繼承此類。

## 預定義初始化方法

### Zeros
```python
keras.initializers.Zeros()
```
全零初始化

### Ones
```python
keras.initializers.Ones()
```
全1初始化

### Constant
```python
keras.initializers.Constant(value=0)
```
初始化為固定值value

### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
```
正態分佈初始化

* mean：均值
* stddev：標準差
* seed：隨機數種子

### RandomUniform
```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```
均勻分佈初始化
* minval：均勻分佈下邊界
* maxval：均勻分佈上邊界
* seed：隨機數種子


### TruncatedNormal
```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```
截尾高斯分佈初始化，該初始化方法與RandomNormal類似，但位於均值兩個標準差以外的數據將會被丟棄並重新生成，形成截尾分佈。該分佈是神經網絡權重和濾波器的推薦初始化方法。

* mean：均值
* stddev：標準差
* seed：隨機數種子

### VarianceScaling
```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```


該初始化方法能夠自適應目標張量的shape。

當```distribution="normal"```時，樣本從0均值，標準差為sqrt(scale / n)的截尾正態分佈中產生。其中：

* 當```mode = "fan_in"```時，權重張量的輸入單元數。
* 當```mode = "fan_out"```時，權重張量的輸出單元數
* 當```mode = "fan_avg"```時，權重張量的輸入輸出單元數的均值

當```distribution="uniform"```時，權重從[-limit, limit]範圍內均勻採樣，其中limit = limit = sqrt(3 * scale / n)

* scale: 放縮因子，正浮點數
* mode: 字符串，“fan_in”，“fan_out”或“fan_avg”fan_in", "fan_out", "fan_avg".
* distribution: 字符串，“normal”或“uniform”.
* seed: 隨機數種子

### Orthogonal
```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

用隨機正交矩陣初始化

* gain: 正交矩陣的乘性係數
* seed：隨機數種子

參考文獻：[Saxe et al.](http://arxiv.org/abs/1312.6120)

### Identiy
```python
keras.initializers.Identity(gain=1.0)
```
使用單位矩陣初始化，僅適用於2D方陣

* gain：單位矩陣的乘性係數

### lecun_uniform
```python
lecun_uniform(seed=None)
```

LeCun均勻分佈初始化方法，參數由[-limit, limit]的區間中均勻採樣獲得，其中limit=sqrt(3 / fan_in), fin_in是權重向量的輸入單元數（扇入）

* seed：隨機數種子

參考文獻：[LeCun 98, Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### lecun_normal
```python
lecun_normal(seed=None)
```
LeCun正態分佈初始化方法，參數由0均值，標準差為stddev = sqrt(1 / fan_in)的正態分佈產生，其中fan_in和fan_out是權重張量的扇入扇出（即輸入和輸出單元數目）

* seed：隨機數種子

參考文獻：

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
[Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)



### glorot_normal
```python
glorot_normal(seed=None)
```

Glorot正態分佈初始化方法，也稱作Xavier正態分佈初始化，參數由0均值，標準差為sqrt(2 / (fan_in + fan_out))的正態分佈產生，其中fan_in和fan_out是權重張量的扇入扇出（即輸入和輸出單元數目）

* seed：隨機數種子

參考文獻：[Glorot & Bengio, AISTATS 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

###glorot_uniform

```python
glorot_uniform(seed=None)
```
Glorot均勻分佈初始化方法，又成Xavier均勻初始化，參數從[-limit, limit]的均勻分佈產生，其中limit為`sqrt(6 / (fan_in + fan_out))`。 fan_in為權值張量的輸入單元數，fan_out是權重張量的輸出單元數。

* seed：隨機數種子

參考文獻：[Glorot & Bengio, AISTATS 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

### he_normal
```python
he_normal(seed=None)
```

He正態分佈初始化方法，參數由0均值，標準差為sqrt(2 / fan_in) 的正態分佈產生，其中fan_in權重張量的扇入

* seed：隨機數種子

參考文獻：[He et al](http://arxiv.org/abs/1502.01852)


### he_uniform
```python
he_normal(seed=None)
```

LeCun均勻分佈初始化方法，參數由[-limit, limit]的區間中均勻採樣獲得，其中limit=sqrt(6 / fan_in), fin_in是權重向量的輸入單元數（扇入）

* seed：隨機數種子

參考文獻：[He et al](http://arxiv.org/abs/1502.01852)

## 自定義初始化器
如果需要傳遞自定義的初始化器，則該初始化器必須是callable的，並且接收```shape```（將被初始化的張量shape）和```dtype```（數據類型）兩個參數，並返回符合```shape```和```dtype```的張量。


```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, init=my_init))
```