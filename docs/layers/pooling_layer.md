# 池化層

## MaxPooling1D層
```python
keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```
對時域1D信號進行最大值池化

### 參數

* pool_size：整數，池化窗口大小

* strides：整數或None，下採樣因子，例如設2將會使得輸出shape為輸入的一半，若為None則預設值為pool_size。

* padding：‘valid’或者‘same’

### 輸入shape

* 形如（samples，steps，features）的3D張量

### 輸出shape

* 形如（samples，downsampled_steps，features）的3D張量

***

## MaxPooling2D層
```python
keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```
為空域信號施加最大值池化

### 參數

* pool_size：整數或長為2的整數tuple，代表在兩個方向（豎直，水平）上的下採樣因子，如取（2，2）將使圖片在兩個維度上均變為原長的一半。為整數意為各個維度值相同且為該數字。

* strides：整數或長為2的整數tuple，或者None，步長值。

* border_mode：‘valid’或者‘same’


* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples，channels, rows，cols）的4D張量

‘channels_last’模式下，為形如（samples，rows, cols，channels）的4D張量

### 輸出shape

‘channels_first’模式下，為形如（samples，channels, pooled_rows, pooled_cols）的4D張量

‘channels_last’模式下，為形如（samples，pooled_rows, pooled_cols，channels）的4D張量

***

## MaxPooling3D層
```python
keras.layers.pooling.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```
為3D信號（空域或時空域）施加最大值池化

本層目前只能在使用Theano為後端時可用

### 參數

* pool_size：整數或長為3的整數tuple，代表在三個維度上的下採樣因子，如取（2，2，2）將使信號在每個維度都變為原來的一半長。

* strides：整數或長為3的整數tuple，或者None，步長值。

* padding：‘valid’或者‘same’

* data_format：字符串，“channels_first”或“channels_last”之一，代表數據的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128x128的數據為例，“channels_first”應將數據組織為（3,128,128,128），而“channels_last”應將數據組織為（128,128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D張量

‘channels_last’模式下，為形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D張量

### 輸出shape

‘channels_first’模式下，為形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D張量

‘channels_last’模式下，為形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D張量

***

## AveragePooling1D層
```python
keras.layers.pooling.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```
對時域1D信號進行平均值池化

### 參數

* pool_size：整數，池化窗口大小

* strides：整數或None，下採樣因子，例如設2將會使得輸出shape為輸入的一半，若為None則預設值為pool_size。

* padding：‘valid’或者‘same’

### 輸入shape

* 形如（samples，steps，features）的3D張量

### 輸出shape

* 形如（samples，downsampled_steps，features）的3D張量

***

## AveragePooling2D層
```python
keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```
為空域信號施加平均值池化

### 參數

* pool_size：整數或長為2的整數tuple，代表在兩個方向（豎直，水平）上的下採樣因子，如取（2，2）將使圖片在兩個維度上均變為原長的一半。為整數意為各個維度值相同且為該數字。

* strides：整數或長為2的整數tuple，或者None，步長值。

* border_mode：‘valid’或者‘same’


* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples，channels, rows，cols）的4D張量

‘channels_last’模式下，為形如（samples，rows, cols，channels）的4D張量

### 輸出shape

‘channels_first’模式下，為形如（samples，channels, pooled_rows, pooled_cols）的4D張量

‘channels_last’模式下，為形如（samples，pooled_rows, pooled_cols，channels）的4D張量

***

## AveragePooling3D層
```python
keras.layers.pooling.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```
為3D信號（空域或時空域）施加平均值池化

本層目前只能在使用Theano為後端時可用

### 參數

* pool_size：整數或長為3的整數tuple，代表在三個維度上的下採樣因子，如取（2，2，2）將使信號在每個維度都變為原來的一半長。

* strides：整數或長為3的整數tuple，或者None，步長值。

* padding：‘valid’或者‘same’

* data_format：字符串，“channels_first”或“channels_last”之一，代表數據的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128x128的數據為例，“channels_first”應將數據組織為（3,128,128,128），而“channels_last”應將數據組織為（128,128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

‘channels_first’模式下，為形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D張量

‘channels_last’模式下，為形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D張量

### 輸出shape

‘channels_first’模式下，為形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D張量

‘channels_last’模式下，為形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D張量

***

## GlobalMaxPooling1D層
```python
keras.layers.pooling.GlobalMaxPooling1D()
```
對於時間信號的全局最大池化

### 輸入shape

* 形如（samples，steps，features）的3D張量

### 輸出shape

* 形如(samples, features)的2D張量

***

## GlobalAveragePooling1D層
```python
keras.layers.pooling.GlobalAveragePooling1D()
```
為時域信號施加全局平均值池化

### 輸入shape

* 形如（samples，steps，features）的3D張量
### 輸出shape

* 形如(samples, features)的2D張量

***

## GlobalMaxPooling2D層
```python
keras.layers.pooling.GlobalMaxPooling2D(dim_ordering='default')
```
為空域信號施加全局最大值池化

### 參數

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples，channels, rows，cols）的4D張量

‘channels_last’模式下，為形如（samples，rows, cols，channels）的4D張量

### 輸出shape

形如(nb_samples, channels)的2D張量

***

## GlobalAveragePooling2D層
```python
keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')
```
為空域信號施加全局平均值池化

### 參數

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples，channels, rows，cols）的4D張量

‘channels_last’模式下，為形如（samples，rows, cols，channels）的4D張量

### 輸出shape

形如(nb_samples, channels)的2D張量