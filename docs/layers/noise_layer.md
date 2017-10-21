# 噪聲層Noise

## GaussianNoise層
```python
keras.layers.noise.GaussianNoise(stddev)
```

為數據施加0均值，標準差為```stddev```的加性高斯噪聲。該層在克服過擬合時比較有用，你可以將它看作是隨機的數據提升。高斯噪聲是需要對輸入數據進行破壞時的自然選擇。


因為這是一個起正則化作用的層，該層只在訓練時才有效。

### 參數

* stddev：浮點數，代表要產生的高斯噪聲標準差

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

***

## GaussianDropout層
```python
keras.layers.noise.GaussianDropout(rate)
```
為層的輸入施加以1為均值，標準差為```sqrt(rate/(1-rate)```的乘性高斯噪聲

因為這是一個起正則化作用的層，該層只在訓練時才有效。

### 參數

* rate：浮點數，斷連概率，與[Dropout層](core_layer/#dropout)相同

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

### 參考文獻

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)



## AlphaDropout
```python
keras.layers.noise.AlphaDropout(rate, noise_shape=None, seed=None)
```
對輸入施加Alpha Dropout

Alpha Dropout是一種保持輸入均值和方差不變的Dropout，該層的作用是即使在dropout時也保持數據的自規範性。通過隨機對負的飽和值進行激活，Alphe Drpout與selu激活函數配合較好。


### 參數

* rate: 浮點數，類似Dropout的Drop比例。乘性mask的標準差將保證為`sqrt(rate / (1 - rate))`.
* seed: 隨機數種子

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

### 參考文獻

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)