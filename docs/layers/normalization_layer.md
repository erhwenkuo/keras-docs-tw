# （批）規範化BatchNormalization

## BatchNormalization層
```python
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer=' ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```
該層在每個batch上將前一層的激活值重新規範化，即使得其輸出數據的均值接近0，其標準差接近1

### 參數


* axis: 整數，指定要規範化的軸，通常為特徵軸。例如在進行```data_format="channels_first```的2D卷積後，一般會設axis=1。
* momentum: 動態均值的動量
* epsilon：大於0的小浮點數，用於防止除0錯誤
* center: 若設為True，將會將beta作為偏置加上去，否則忽略參數beta
* scale: 若設為True，則會乘以gamma，否則不使用gamma。當下一層是線性的時，可以設False，因為scaling的操作將被下一層執行。
* beta_initializer：beta權重的初始方法
* gamma_initializer: gamma的初始化方法
* moving_mean_initializer: 動態均值的初始化方法
* moving_variance_initializer: 動態方差的初始化方法
* beta_regularizer: 可選的beta正則
* gamma_regularizer: 可選的gamma正則
* beta_constraint: 可選的beta約束
* gamma_constraint: 可選的gamma約束



### 輸入shape

任意，當使用本層為模型首層時，指定```input_shape```參數時有意義。

### 輸出shape

與輸入shape相同

### 參考文獻

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)

【Tips】BN層的作用

（1）加速收斂
（2）控製過擬合，可以少用或不用Dropout和正則
（3）降低網絡對初始化權重不敏感
（4）允許使用較大的學習率