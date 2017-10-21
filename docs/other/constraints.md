# 約束項

來自```constraints```模塊的函數在優化過程中為網絡的參數施加約束

懲罰項基於層進行懲罰，目前懲罰項的接口與層有關，但```Dense, Conv1D, Conv2D, Conv3D```具有共同的接口。

這些層通過一下關鍵字施加約束項

* ```kernel_constraint```：對主權重矩陣進行約束

* ```bias_constraint```：對偏置向量進行約束

```python
from keras.constraints import maxnorm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 預定義約束項

* max_norm(m=2)：最大模約束

* non_neg()：非負性約束

* unit_norm()：單位範數約束, 強制矩陣沿最後一個軸擁有單位範數

* min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0): 最小/最大範數約束