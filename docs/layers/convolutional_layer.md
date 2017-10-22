# 卷積層

## Conv1D層
```python
keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

一維卷積層（即時域卷積），用以在一維輸入信號上進行鄰域濾波。當使用該層作為首層時，需要提供關鍵字參數```input_shape```。例如```(10,128)```代表一個長為10的序列，序列中每個信號為128向量。而```(None, 128)```代表變長的128維向量序列。

該層生成將輸入信號與卷積核按照單一的空域（或時域）方向進行卷積。如果```use_bias=True```，則還會加上一個偏置項，若```activation```不為None，則輸出為經過激活函數的輸出。


### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：整數或由單個整數構成的list/tuple，卷積核的空域或時域窗長度

* strides：整數或由單個整數構成的list/tuple，為卷積的步長。任何不為1的strides均與任何不為1的dilation_rate均不兼容

* padding：補0策略，為“valid”, “same” 或“causal”，“causal”將產生因果（膨脹的）卷積，即output[t]不依賴於input[t+1：]。當對不能違反時間順序的時序信號建模時有用。參考[WaveNet: A Generative Model for Raw Audio, section 2.1.](https://arxiv.org/abs/1609.03499)。 “valid”代表只進行有效的捲積，即對邊界數據不處理。 “same”代表保留邊界處的捲積結果，通常會導致輸出shape與輸入shape相同。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* dilation_rate：整數或由單個整數構成的list/tuple，指定dilated convolution中的膨脹比例。任何不為1的dilation_rate均與任何不為1的strides均不兼容。

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


【Tips】可以將Convolution1D看作Convolution2D的快捷版，對例子中（10，32）的信號進行1D卷積相當於對其進行卷積核為（filter_length, 32）的2D卷積。 【@3rduncle】

***
## Conv2D層
```python
keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform ', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
二維卷積層，即對圖像的空域卷積。該層對二維輸入進行滑動窗卷積，當使用該層作為第一層時，應提供```input_shape```參數。例如```input_shape = (128,128,3)```代表128*128的彩色RGB圖像（```data_format='channels_last'```）

### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：單個整數或由兩個整數構成的list/tuple，卷積核的寬度和長度。如為單個整數，則表示在各個空間維度的相同長度。

* strides：單個整數或由兩個整數構成的list/tuple，為卷積的步長。如為單個整數，則表示在各個空間維度的相同步長。任何不為1的strides均與任何不為1的dilation_rate均不兼容

* padding：補0策略，為“valid”, “same” 。 “valid”代表只進行有效的捲積，即對邊界數據不處理。 “same”代表保留邊界處的捲積結果，通常會導致輸出shape與輸入shape相同。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* dilation_rate：單個整數或由兩個個整數構成的list/tuple，指定dilated convolution中的膨脹比例。任何不為1的dilation_rate均與任何不為1的strides均不兼容。

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

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
***

## SeparableConv2D層
```python
keras.layers.convolutional.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer= 'glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```
該層是在深度方向上的可分離卷積。

可分離卷積首先按深度方向進行卷積（對每個輸入通道分別卷積），然後逐點進行卷積，將上一步的捲積結果混合到輸出通道中。參數```depth_multiplier```控制了在depthwise卷積（第一步）的過程中，每個輸入通道信號產生多少個輸出通道。

直觀來說，可分離卷積可以看做講一個卷積核分解為兩個小的捲積核，或看作Inception模塊的一種極端情況。

當使用該層作為第一層時，應提供```input_shape```參數。例如```input_shape = (3,128,128)```代表128*128的彩色RGB圖像


### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：單個整數或由兩個個整數構成的list/tuple，卷積核的寬度和長度。如為單個整數，則表示在各個空間維度的相同長度。

* strides：單個整數或由兩個整數構成的list/tuple，為卷積的步長。如為單個整數，則表示在各個空間維度的相同步長。任何不為1的strides均與任何不為1的dilation_rate均不兼容

* padding：補0策略，為“valid”, “same” 。 “valid”代表只進行有效的捲積，即對邊界數據不處理。 “same”代表保留邊界處的捲積結果，通常會導致輸出shape與輸入shape相同。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* dilation_rate：單個整數或由兩個個整數構成的list/tuple，指定dilated convolution中的膨脹比例。任何不為1的dilation_rate均與任何不為1的strides均不兼容。

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

* use_bias:布爾值，是否使用偏置項

* depth_multiplier：在按深度卷積的步驟中，每個輸入通道使用多少個輸出通道

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* depthwise_regularizer：施加在按深度卷積的權重上的正則項，為[Regularizer](../other/regularizers)對象

* pointwise_regularizer：施加在按點卷積的權重上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象

* depthwise_constraint：施加在按深度卷積權重上的約束項，為[Constraints](../other/constraints)對象

* pointwise_constraint施加在按點卷積權重的約束項，為[Constraints](../other/constraints)對象


### 輸入shape

‘channels_first’模式下，輸入形如（samples,channels，rows，cols）的4D張量

‘channels_last’模式下，輸入形如（samples，rows，cols，channels）的4D張量

注意這裡的輸入shape指的是函數內部實現的輸入shape，而非函數接口應指定的```input_shape```，請參考下面提供的例子。

### 輸出shape

‘channels_first’模式下，為形如（samples，nb_filter, new_rows, new_cols）的4D張量

‘channels_last’模式下，為形如（samples，new_rows, new_cols，nb_filter）的4D張量

輸出的行列數可能會因為填充方法而改變

***

## Conv2DTranspose層
```python
keras.layers.convolutional.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
該層是轉置的捲積操作（反捲積）。需要反捲積的情況通常發生在用戶想要對一個普通卷積的結果做反方向的變換。例如，將具有該卷積層輸出shape的tensor轉換為具有該卷積層輸入shape的tensor。同時保留與卷積層兼容的連接模式。

當使用該層作為第一層時，應提供```input_shape```參數。例如```input_shape = (3,128,128)```代表128*128的彩色RGB圖像

### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：單個整數或由兩個個整數構成的list/tuple，卷積核的寬度和長度。如為單個整數，則表示在各個空間維度的相同長度。

* strides：單個整數或由兩個整數構成的list/tuple，為卷積的步長。如為單個整數，則表示在各個空間維度的相同步長。任何不為1的strides均與任何不為1的dilation_rate均不兼容

* padding：補0策略，為“valid”, “same” 。 “valid”代表只進行有效的捲積，即對邊界數據不處理。 “same”代表保留邊界處的捲積結果，通常會導致輸出shape與輸入shape相同。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* dilation_rate：單個整數或由兩個個整數構成的list/tuple，指定dilated convolution中的膨脹比例。任何不為1的dilation_rate均與任何不為1的strides均不兼容。

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

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


### 參考文獻
* [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
* [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
* [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

***

## Conv3D層
```python
keras.layers.convolutional.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
三維卷積對三維的輸入進行滑動窗卷積，當使用該層作為第一層時，應提供```input_shape```參數。例如```input_shape = (3,10,128,128)```代表對10幀128*128的彩色RGB圖像進行卷積。數據的通道位置仍然有```data_format```參數指定。

### 參數

* filters：卷積核的數目（即輸出的維度）

* kernel_size：單個整數或由3個整數構成的list/tuple，卷積核的寬度和長度。如為單個整數，則表示在各個空間維度的相同長度。

* strides：單個整數或由3個整數構成的list/tuple，為卷積的步長。如為單個整數，則表示在各個空間維度的相同步長。任何不為1的strides均與任何不為1的dilation_rate均不兼容

* padding：補0策略，為“valid”, “same” 。 “valid”代表只進行有效的捲積，即對邊界數據不處理。 “same”代表保留邊界處的捲積結果，通常會導致輸出shape與輸入shape相同。

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)），或逐元素（element-wise）的Theano函數。如果不指定該參數，將不會使用任何激活函數（即使用線性激活函數：a(x)=x）

* dilation_rate：單個整數或由3個個整數構成的list/tuple，指定dilated convolution中的膨脹比例。任何不為1的dilation_rate均與任何不為1的strides均不兼容。

* data_format：字符串，“channels_first”或“channels_last”之一，代表數據的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128x128的數據為例，“channels_first”應將數據組織為（3,128,128,128），而“channels_last”應將數據組織為（128,128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

* use_bias:布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象



### 輸入shape

‘channels_first’模式下，輸入應為形如（samples，channels，input_dim1，input_dim2, input_dim3）的5D張量

‘channels_last’模式下，輸入應為形如（samples，input_dim1，input_dim2, input_dim3，channels）的5D張量

這裡的輸入shape指的是函數內部實現的輸入shape，而非函數接口應指定的```input_shape```。

***

## Cropping1D層
```python
keras.layers.convolutional.Cropping1D(cropping=(1, 1))
```
在時間軸（axis1）上對1D輸入（即時間序列）進行裁剪

### 參數

* cropping：長為2的tuple，指定在序列的首尾要裁剪掉多少個元素

### 輸入shape

* 形如（samples，axis_to_crop，features）的3D張量

### 輸出shape

* 形如（samples，cropped_axis，features）的3D張量

***
## Cropping2D層
```python
keras.layers.convolutional.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```
對2D輸入（圖像）進行裁剪，將在空域維度，即寬和高的方向上裁剪

### 參數

* cropping：長為2的整數tuple，分別為寬和高方向上頭部與尾部需要裁剪掉的元素數

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

形如（samples，depth, first_axis_to_crop, second_axis_to_crop）


### 輸出shape

形如(samples, depth, first_cropped_axis, second_cropped_axis)的4D張量

***
## Cropping3D層
```python
keras.layers.convolutional.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```
對2D輸入（圖像）進行裁剪

### 參數
* cropping：長為3的整數tuple，分別為三個方向上頭部與尾部需要裁剪掉的元素數

* data_format：字符串，“channels_first”或“channels_last”之一，代表數據的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128x128的數據為例，“channels_first”應將數據組織為（3,128,128,128），而“channels_last”應將數據組織為（128,128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

形如 (samples, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)的5D張量

### 輸出shape
形如(samples, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)的5D張量
***
## UpSampling1D層
```python
keras.layers.convolutional.UpSampling1D(size=2)
```
在時間軸上，將每個時間步重複```length```次

### 參數

* size：上採樣因子

### 輸入shape

* 形如（samples，steps，features）的3D張量

### 輸出shape

* 形如（samples，upsampled_steps，features）的3D張量

***

## UpSampling2D層
```python
keras.layers.convolutional.UpSampling2D(size=(2, 2), data_format=None)
```
將數據的行和列分別重複size\[0\]和size\[1\]次

### 參數

* size：整數tuple，分別為行和列上採樣因子

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples，channels, rows，cols）的4D張量

‘channels_last’模式下，為形如（samples，rows, cols，channels）的4D張量

### 輸出shape

‘channels_first’模式下，為形如（samples，channels, upsampled_rows, upsampled_cols）的4D張量

‘channels_last’模式下，為形如（samples，upsampled_rows, upsampled_cols，channels）的4D張量

***

## UpSampling3D層
```python
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), data_format=None)
```
將數據的三個維度上分別重複size\[0\]、size\[1\]和ize\[2\]次

本層目前只能在使用Theano為後端時可用

### 參數

* size：長為3的整數tuple，代表在三個維度上的上採樣因子

* data_format：字符串，“channels_first”或“channels_last”之一，代表數據的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128x128的數據為例，“channels_first”應將數據組織為（3,128,128,128），而“channels_last”應將數據組織為（128,128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D張量

‘channels_last’模式下，為形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D張量

### 輸出shape

‘channels_first’模式下，為形如（samples, channels, dim1, dim2, dim3）的5D張量

‘channels_last’模式下，為形如（samples, upsampled_dim1, upsampled_dim2, upsampled_dim3,channels,）的5D張量

***

## ZeroPadding1D層
```python
keras.layers.convolutional.ZeroPadding1D(padding=1)
```
對1D輸入的首尾端（如時域序列）填充0，以控製卷積以後向量的長度

### 參數

* padding：整數，表示在要填充的軸的起始和結束處填充0的數目，這裡要填充的軸是軸1（第1維，第0維是樣本數）

### 輸入shape

形如（samples，axis_to_pad，features）的3D張量

### 輸出shape

形如（samples，paded_axis，features）的3D張量

***

## ZeroPadding2D層
```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), data_format=None)
```
對2D輸入（如圖片）的邊界填充0，以控製卷積以後特徵圖的大小

### 參數

* padding：整數tuple，表示在要填充的軸的起始和結束處填充0的數目，這裡要填充的軸是軸3和軸4（即在'th'模式下圖像的行和列，在' channels_last'模式下要填充的則是軸2，3）

* data_format：字符串，“channels_first”或“channels_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128的RGB圖像為例，“channels_first”應將數據組織為（3,128,128），而“channels_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，形如（samples，channels，first_axis_to_pad，second_axis_to_pad）的4D張量

‘channels_last’模式下，形如（samples，first_axis_to_pad，second_axis_to_pad, channels）的4D張量

### 輸出shape

‘channels_first’模式下，形如（samples，channels，first_paded_axis，second_paded_axis）的4D張量

‘channels_last’模式下，形如（samples，first_paded_axis，second_paded_axis, channels）的4D張量

***

## ZeroPadding3D層
```python
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```
將數據的三個維度上填充0

本層目前只能在使用Theano為後端時可用

### 參數

padding：整數tuple，表示在要填充的軸的起始和結束處填充0的數目，這裡要填充的軸是軸3，軸4和軸5，‘channels_last’模式下則是軸2，3和4

* data_format：字符串，“channels_first”或“channels_last”之一，代表數據的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channels_last”對應原本的“tf”，“channels_first”對應原本的“th”。以128x128x128的數據為例，“channels_first”應將數據組織為（3,128,128,128），而“channels_last”應將數據組織為（128,128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channels_last”。

### 輸入shape

‘channels_first’模式下，為形如（samples, channels, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad,）的5D張量

‘channels_last’模式下，為形如（samples, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad, channels）的5D張量

### 輸出shape

‘channels_first’模式下，為形如（samples, channels, first_paded_axis，second_paded_axis, third_paded_axis,）的5D張量

‘channels_last’模式下，為形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D張量
