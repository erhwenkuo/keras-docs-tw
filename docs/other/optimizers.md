# 優化器optimizers

優化器是編譯Keras模型必要的兩個參數之一
```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

可以在調用```model.compile()```之前初始化一個優化器對象，然後傳入該函數（如上所示），也可以在調用```model.compile()```時傳遞一個預定義優化器名。在後者情形下，優化器的參數將使用預設值。
```python
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```
## 所有優化器都可用的參數
參數```clipnorm```和```clipvalue```是所有優化器都可以使用的參數,用於對梯度進行裁剪.示例如下:
```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```
```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

## SGD
```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```
隨機梯度下降法，支持動量參數，支持學習衰減率，支持Nesterov動量

### 參數

* lr：大或等於0的浮點數，學習率

* momentum：大或等於0的浮點數，動量參數

* decay：大或等於0的浮點數，每次更新後的學習率衰減值

* nesterov：布爾值，確定是否使用Nesterov動量

***

## RMSprop
```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
```
除學習率可調整外，建議保持優化器的其他預設參數不變

該優化器通常是面對遞歸神經網絡時的一個良好選擇

### 參數

* lr：大或等於0的浮點數，學習率

* rho：大或等於0的浮點數

* epsilon：大或等於0的小浮點數，防止除0錯誤

***

## Adagrad
```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
```
建議保持優化器的預設參數不變

### Adagrad

* lr：大或等於0的浮點數，學習率

* epsilon：大或等於0的小浮點數，防止除0錯誤

***

## Adadelta
```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
```
建議保持優化器的預設參數不變

### 參數

* lr：大或等於0的浮點數，學習率

* rho：大或等於0的浮點數

* epsilon：大或等於0的小浮點數，防止除0錯誤

### 參考文獻

***

* [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

## Adam
```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

該優化器的預設值來源於參考文獻

### 參數

* lr：大或等於0的浮點數，學習率

* beta_1/beta_2：浮點數， 0<beta<1，通常很接近1

* epsilon：大或等於0的小浮點數，防止除0錯誤

### 參考文獻

* [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

***

## Adamax
```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

Adamax優化器來自於Adam的論文的Section7，該方法是基於無窮範數的Adam方法的變體。

預設參數由論文提供

### 參數

* lr：大或等於0的浮點數，學習率

* beta_1/beta_2：浮點數， 0<beta<1，通常很接近1

* epsilon：大或等於0的小浮點數，防止除0錯誤

### 參考文獻

* [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

***

## Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
```

Nesterov Adam optimizer: Adam本質上像是帶有動量項的RMSprop，Nadam就是帶有Nesterov 動量的Adam RMSprop

預設參數來自於論文，推薦不要對預設參數進行更改。

### 參數

* lr：大或等於0的浮點數，學習率

* beta_1/beta_2：浮點數， 0<beta<1，通常很接近1

* epsilon：大或等於0的小浮點數，防止除0錯誤

### 參考文獻

* [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)

* [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

## TFOptimizer
```python
keras.optimizers.TFOptimizer(optimizer)
```
TF優化器的包裝器