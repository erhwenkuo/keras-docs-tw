# 正則項

正則項在優化過程中層的參數或層的激活值添加懲罰項，這些懲罰項將與損失函數一起作為網絡的最終優化目標

懲罰項基於層進行懲罰，目前懲罰項的接口與層有關，但```Dense, Conv1D, Conv2D, Conv3D```具有共同的接口。

這些層有三個關鍵字參數以施加正則項：

* ```kernel_regularizer```：施加在權重上的正則項，為```keras.regularizer.Regularizer```對象

* ```bias_regularizer```：施加在偏置向量上的正則項，為```keras.regularizer.Regularizer```對象

* ```activity_regularizer```：施加在輸出上的正則項，為```keras.regularizer.Regularizer```對象

## 例子
```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 可用正則項

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0.)
```

## 開發新的正則項

任何以權重矩陣作為輸入並返回單個數值的函數均可以作為正則項，示例：

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)
```

可參考源代碼[keras/regularizer.py](https://github.com/fchollet/keras/blob/master/keras/regularizers.py)