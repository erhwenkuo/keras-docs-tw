# 局部連接層LocallyConnceted

## LocallyConnected1D層
```python
keras.layers.local.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
```LocallyConnected1D```層與```Conv1D```工作方式類似，唯一的區別是不進行權值共享。即施加在不同輸入位置的濾波器是不一樣的。

### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：整數或由單個整數構成的list/tuple，卷積核的空域或時域窗長度

* strides：整數或由單個整數構成的list/tuple，為卷積的步長。任何不為1的strides均與任何不為1的dilation_rata均不兼容

* padding：補0策略，目前僅支持`valid`（大小寫敏感），`same`可能會在將來支持。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* dilation_rate：整數或由單個整數構成的list/tuple，指定dilated convolution中的膨脹比例。任何不為1的dilation_rata均與任何不為1的strides均不兼容。

* use_bias:布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象

### 輸入shape

形如（samples，steps，input_dim）的3D張量

### 輸出shape

形如（samples，new_steps，nb_filter）的3D張量，因為有向量填充的原因，```steps```的值會改變


***

## LocallyConnected2D層
```python
keras.layers.local.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
```LocallyConnected2D```層與```Convolution2D```工作方式類似，唯一的區別是不進行權值共享。即施加在不同輸入patch的濾波器是不一樣的，當使用該層作為模型首層時，需要提供參數```input_dim```或```input_shape```參數。參數含義參考```Convolution2D```。

### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：單個整數或由兩個整數構成的list/tuple，卷積核的寬度和長度。如為單個整數，則表示在各個空間維度的相同長度。

* strides：單個整數或由兩個整數構成的list/tuple，為卷積的步長。如為單個整數，則表示在各個空間維度的相同步長。

* padding：補0策略，目前僅支持`valid`（大小寫敏感），`same`可能會在將來支持。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的默認值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

* use_bias:布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象

### 輸入shape

‘channels_first’模式下，輸入形如（samples,channels，rows，cols）的4D張量

‘channels_last’模式下，輸入形如（samples，rows，cols，channels）的4D張量

注意這裡的輸入shape指的是函數內部實現的輸入shape，而非函數接口應指定的```input_shape```，請參考下面提供的例子。

### 輸出shape

‘channels_first’模式下，為形如（samples，nb_filter, new_rows, new_cols）的4D張量

‘channels_last’模式下，為形如（samples，new_rows, new_cols，nb_filter）的4D張量

輸出的行列數可能會因為填充方法而改變

### 例子

```python
# apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image
# with `data_format="channels_last"`:
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# now model.output_shape == (None, 30, 30, 64)
# notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

# add a 3x3 unshared weights convolution on top, with 32 output filters:
model.add(LocallyConnected2D(32, (3, 3)))
# now model.output_shape == (None, 28, 28, 32)
```