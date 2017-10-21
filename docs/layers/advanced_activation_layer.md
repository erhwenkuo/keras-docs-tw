# 高級激活層Advanced Activation

## LeakyReLU層
```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```
LeakyRelU是修正線性單元（Rectified Linear Unit，ReLU）的特殊版本，當不激活時，LeakyReLU仍然會有非零輸出值，從而獲得一個小梯度，避免ReLU可能出現的神經元“死亡”現象。即，```f(x)=alpha * x for x < 0```, ```f(x) = x for x>=0```

### 參數

* alpha：大於0的浮點數，代表激活函數圖像中第三象限線段的斜率

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

### 參考文獻

[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

***

## PReLU層
```python
keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```
該層為參數化的ReLU（Parametric ReLU），表達式是：```f(x) = alpha * x for x < 0```, ```f(x) = x for x>=0` ``，此處的```alpha```為一個與xshape相同的可學習的參數向量。

### 參數

* alpha_initializer：alpha的初始化函數
* alpha_regularizer：alpha的正則項
* alpha_constraint：alpha的約束項
* shared_axes：該參數指定的軸將共享同一組科學系參數，例如假如輸入特徵圖是從2D卷積過來的，具有形如`(batch, height, width, channels)`這樣的shape，則或許你會希望在空域共享參數，這樣每個filter就只有一組參數，設定`shared_axes=[1,2]`可完成該目標

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

### 參考文獻

* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)

***

## ELU層
```python
keras.layers.advanced_activations.ELU(alpha=1.0)
```
ELU層是指數線性單元（Exponential Linera Unit），表達式為：
該層為參數化的ReLU（Parametric ReLU），表達式是：```f(x) = alpha * (exp(x) - 1.) for x < 0```, ```f(x) = x for x>=0```

### 參數

* alpha：控制負因子的參數

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

### 參考文獻

* [>Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)

***

## ThresholdedReLU層
```python
keras.layers.advanced_activations.ThresholdedReLU(theta=1.0)
```
該層是帶有門限的ReLU，表達式是：```f(x) = x for x > theta```,```f(x) = 0 otherwise```

### 參數

* theata：大或等於0的浮點數，激活門限位置

### 輸入shape

任意，當使用該層為模型首層時需指定```input_shape```參數

### 輸出shape

與輸入相同

### 參考文獻

* [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)

***