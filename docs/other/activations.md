# 激活函數Activations

激活函數可以通過設置單獨的[激活層](../layers/core_layer/#activation)實現，也可以在構造層對象時通過傳遞```activation```參數實現。

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

等價於

```python
model.add(Dense(64, activation='tanh'))
```

也可以通過傳遞一個逐元素運算的Theano/TensorFlow/CNTK函數來作為激活函數：
```python
from keras import backend as K

def tanh(x):
    return K.tanh(x)

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh)
```

***

## 預定義激活函數

* softmax：對輸入數據的最後一維進行softmax，輸入數據應形如```(nb_samples, nb_timesteps, nb_dims)```或```(nb_samples,nb_dims)```

* elu

* selu: 可伸縮的指數線性單元（Scaled Exponential Linear Unit），參考[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

* softplus

* softsign

* relu

* tanh

* sigmoid

* hard_sigmoid

* linear

## 高級激活函數

對於簡單的Theano/TensorFlow/CNTK不能表達的複雜激活函數，如含有可學習參數的激活函數，可通過[高級激活函數](../layers/advanced_activation_layer)實現，如PReLU，LeakyReLU等