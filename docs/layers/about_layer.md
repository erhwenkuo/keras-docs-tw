# 關於Keras的“層”（Layer）

所有的Keras層對像都有如下方法：

* ```layer.get_weights()```：返回層的權重（numpy array）

* ```layer.set_weights(weights)```：從numpy array中將權重加載到該層中，要求numpy array的形狀與* ```layer.get_weights()```的形狀相同

* ```layer.get_config()```：返回當前層配置信息的字典，層也可以藉由配置信息重構:
```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

或者：

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

如果層僅有一個計算節點（即該層不是共享層），則可以通過下列方法獲得輸入張量、輸出張量、輸入數據的形狀和輸出數據的形狀：

* ```layer.input```

* ```layer.output```

* ```layer.input_shape```

* ```layer.output_shape```

如果該層有多個計算節點（參考[層計算節點和共享層](../getting_started/functional_API/#node)）。可以使用下面的方法

* ```layer.get_input_at(node_index)```

* ```layer.get_output_at(node_index)```

* ```layer.get_input_shape_at(node_index)```

* ```layer.get_output_shape_at(node_index)```