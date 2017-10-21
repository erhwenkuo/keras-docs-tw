#編寫自己的層


對於簡單的定制操作，我們或許可以通過使用```layers.core.Lambda```層來完成。但對於任何具​​有可訓練權重的定制層，你應該自己來實現。

這裡是一個Keras2的層應該具有的框架結構(如果你的版本更舊請升級)，要定制自己的層，你需要實現下面三個方法

* ```build(input_shape)```：這是定義權重的方法，可訓練的權應該在這裡被加入列表````self.trainable_weights```中。其他的屬性還包括```self.non_trainabe_weights```（列表）和```self.updates```（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以參考```BatchNormalization```層的實現來學習如何使用上面兩個屬性。這個方法必須設置```self.built = True```，可通過調用```super([layer],self).build()```實現

* ```call(x)```：這是定義層功能的方法，除非你希望你寫的層支持masking，否則你只需要關心```call```的第一個參數：輸入張量

* ```compute_output_shape(input_shape)```：如果你的層修改了輸入數據的shape，你應該在這裡指定shape變化的方法，這個函數使得Keras可以做自動shape推斷

```python
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape) # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

現存的Keras層代碼可以為你的實現提供良好參考，閱讀源代碼吧！