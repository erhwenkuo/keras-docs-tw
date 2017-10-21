# Keras後端

## 什麼是“後端”

Keras是一個模型級的庫，提供了快速構建深度學習網絡的模塊。 Keras並不處理如張量乘法、卷積等底層操作。這些操作依賴於某種特定的、優化良好的張量操作庫。 Keras依賴於處理張量的庫就稱為“後端引擎”。 Keras提供了三種後端引擎Theano/Tensorflow/CNTK，並將其函數統一封裝，使得用戶可以以同一個接口調用不同後端引擎的函數

* [Theano](http://deeplearning.net/software/theano/)是一個開源的符號主義張量操作框架，由蒙特利爾大學LISA/MILA實驗室開發。
* [TensorFlow](http://www.tensorflow.org/)是一個符號主義的張量操作框架，由Google開發。
* [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/)是一個由微軟開發的商業級工具包。

在未來，我們有可能要添加更多的後端選項。

## 切換後端

注意：Windows用戶請把`$Home`改為`%USERPROFILE%`

如果你至少運行過一次Keras，你將在下面的目錄下找到Keras的配置文件：

```$HOME/.keras/keras.json```

如果該目錄下沒有該文件，你可以手動創建一個

文件的默認配置如下：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

將```backend```字段的值改寫為你需要使用的後端：```theano```或```tensorflow```或者`CNTK`，即可完成後端的切換

我們也可以通過定義環境變量```KERAS_BACKEND```來覆蓋上面配置文件中定義的後端：

```python
KERAS_BACKEND=tensorflow python -c "from keras import backend;"
Using TensorFlow backend.
```

## keras.json 細節
```python
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
你可以更改以上`~/.keras/keras.json`中的配置

- `iamge_data_format`：字符串，"channels_last"或"channels_first"，該選項指定了Keras將要使用的維度順序，可通過`keras.backend.image_data_format()`來獲取當前的維度順序。對2D數據來說，"channels_last"假定維度順序為(rows,cols,channels)而"channels_first"假定維度順序為(channels, rows, cols)。對3D數據而言，"channels_last"假定(conv_dim1, conv_dim2, conv_dim3, channels)，"channels_first"則是(channels, conv_dim1, conv_dim2, conv_dim3)

- `epsilon`：浮點數，防止除0錯誤的小數字
- `floatx`：字符串，`"float16"`, `"float32"`, `"float64"`之一，為浮點數精度
- `backend`：字符串，所使用的後端，為"tensorflow"或"theano"



## 使用抽象的Keras後端來編寫代碼

如果你希望你編寫的Keras模塊能夠同時在Theano和TensorFlow兩個後端上使用，你可以通過Keras後端接口來編寫代碼，這裡是一個簡介：

```python
from keras import backend as K
```
下面的代碼實例化了一個輸入佔位符，等價於```tf.placeholder()``` ，```T.matrix()```，```T.tensor3()```等

```python
input = K.placeholder(shape=(2, 4, 5))
# also works:
input = K.placeholder(shape=(None, 4, 5))
# also works:
input = K.placeholder(ndim=3)
```
下面的代碼實例化了一個共享變量（shared），等價於```tf.variable()```或 ```theano.shared()```

```python
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

大多數你需要的張量操作都可以通過統一的Keras後端接口完成，而不關心具體執行這些操作的是Theano還是TensorFlow
```python
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=2)
a = K.softmax(b)
a = concatenate([b, c], axis=-1)
# etc...
```

## Kera後端函數

### backend
```python
backend()
```
返回當前後端

###epsilon
```python
epsilon()
```

以數值形式返回一個（一般來說很小的）數，用以防止除0錯誤

###set_epsilon
```python
set_epsilon(e)
```

設置在數值表達式中使用的fuzz factor，用於防止除0錯誤，該值應該是一個較小的浮點數，示例：
```python
>>> from keras import backend as K
>>> K.epsilon()
1e-08
>>> K.set_epsilon(1e-05)
>>> K.epsilon()
1e-05
```

### floatx
```python
floatx()
```
返回默認的浮點數數據類型，為字符串，如 'float16', 'float32', 'float64'

### set_floatx(floatx)
```python
floatx()
```
設置默認的浮點數數據類型，為字符串，如 'float16', 'float32', 'float64',示例：
```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> K.set_floatx('float16')
>>> K.floatx()
'float16'
```


### cast_to_floatx
```python
cast_to_floatx(x)
```
將numpy array轉換為默認的Keras floatx類​​型，x為numpy array，返回值也為numpy array但其數據類型變為floatx。示例：
```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> arr = numpy.array([1.0, 2.0], dtype='float64')
>>> arr.dtype
dtype('float64')
>>> new_arr = K.cast_to_floatx(arr)
>>> new_arr
array([ 1., 2.], dtype=float32)
>>> new_arr.dtype
dtype('float32')
```

### image_data_format
```python
image_data_format()
```
返回默認的圖像的維度順序（‘channels_last’或‘channels_first’）

### set_image_data_format
```python
set_image_data_format(data_format)
```
設置圖像的維度順序（‘tf’或‘th’）,示例：
>>> from keras import backend as K
>>> K.image_data_format()
>>> 'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
>>> 'channels_last'
```

### is_keras_tensor()
​```python
is_keras_tensor(x)
```
判斷x是否是keras tensor對象的謂詞函數

```python
>>> from keras import backend as K
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var)
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var) # A variable is not a Tensor.
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder) # A placeholder is a Tensor.
True
```

### get_uid
```python
get_uid(prefix='')
```
獲得默認計算圖的uid，依據給定的前綴提供一個唯一的UID，參數為表示前綴的字符串，返回值為整數.
### reset_uids
```python
reset_uids()
```
重置圖的標識符
### is_keras_tensor
```python
is_keras_tensor(x)
```

判斷x是否是一個Keras tensor，返回一個布爾值，示例
```python
>>> from keras import backend as K
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var)
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var) # A variable is not a Tensor.
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder) # A placeholder is a Tensor.
True
```

### clear_session
```python
clear_session()
```
結束當前的TF計算圖，並新建一個。有效的避免模型/層的混亂

### manual_variable_initialization
```python
manual_variable_initialization(value)
```
指出變量應該以其默認值被初始化還是由用戶手動初始化，參數value為布爾值，默認False代表變量由其默認值初始化

### learning_phase

```python
learning_phase()
```
返回訓練模式/測試模式的flag，該flag是一個用以傳入Keras模型的標記，以決定當前模型執行於訓練模式下還是測試模式下

### set_learning_phase

```python
set_learning_phase()
```
設置訓練模式/測試模式0或1

### is_sparse
```python
is_sparse(tensor)
```
判斷一個tensor是不是一個稀疏的tensor(稀不稀疏由tensor的類型決定，而不是tensor實際上有多稀疏)，返回值是一個布爾值，示例：
```python
>>> from keras import backend as K
>>> a = K.placeholder((2, 2), sparse=False)
>>> print(K.is_sparse(a))
False
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
```

### to_dense
```python
to_dense(tensor)
```
將一個稀疏tensor轉換一個不稀疏的tensor並返回之，示例：
```python
>>> from keras import backend as K
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
>>> c = K.to_dense(b)
>>> print(K.is_sparse(c))
False
```

### variable
```python
variable(value, dtype='float32', name=None)
```
實例化一個張量，返回之

參數：

* value：用來初始化張量的值
* dtype：張量數據類型
* name：張量的名字（可選）

示例：
```python
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val, dtype='float64', name='example_var')
>>> K.dtype(kvar)
'float64'
>>> print(kvar)
example_var
>>> kvar.eval()
array([[ 1., 2.],
   [ 3., 4.]])
```

### placeholder
```python
placeholder(shape=None, ndim=None, dtype='float32', name=None)
```
實例化一個佔位符，返回之

參數：

* shape：佔位符的shape（整數tuple，可能包含None）
* ndim: 佔位符張量的階數，要初始化一個佔位符，至少指定```shape```和```ndim```之一，如果都指定則使用```shape`` `
* dtype: 佔位符數據類型
* name: 佔位符名稱（可選）

示例：
```python
>>> from keras import backend as K
>>> input_ph = K.placeholder(shape=(2, 4, 5))
>>> input_ph._keras_shape
(2, 4, 5)
>>> input_ph
<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
```

### shape
```python
shape(x)
```
返回一個張量的符號shape，符號shape的意思是返回值本身也是一個tensor，示例：
```python
>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> input = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(input)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
__To get integer shape (Instead, you can use K.int_shape(x))__

>>> K.shape(kvar).eval(session=tf_session)
array([2, 2], dtype=int32)
>>> K.shape(input).eval(session=tf_session)
array([2, 4, 5], dtype=int32)
```

### int_shape
```python
int_shape(x)
```
以整數Tuple或None的形式返回張量shape，示例：
```python
>>> from keras import backend as K
>>> input = K.placeholder(shape=(2, 4, 5))
>>> K.int_shape(input)
(2, 4, 5)
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.int_shape(kvar)
(2, 2)
```

### ndim
```python
ndim(x)
```
返回張量的階數，為整數，示例：
```python
>>> from keras import backend as K
>>> input = K.placeholder(shape=(2, 4, 5))
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.ndim(input)
3
>>> K.ndim(kvar)
2
```

### dtype
```python
dtype(x)
```
返回張量的數據類型，為字符串，示例：
```python
>>> from keras import backend as K
>>> K.dtype(K.placeholder(shape=(2,4,5)))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
'float64'
__Keras variable__

>>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
>>> K.dtype(kvar)
'float32_ref'
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.dtype(kvar)
'float32_ref'
```

### eval
```python
eval(x)
```
求得張量的值，返回一個Numpy array，示例：
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1., 2.],
   [ 3., 4.]], dtype=float32)
```

### zeros
```python
zeros(shape, dtype='float32', name=None)
```
生成一個全0張量，示例：
```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0., 0., 0., 0.],
   [ 0., 0., 0., 0.],
   [ 0., 0., 0., 0.]], dtype=float32)
```

### ones
```python
ones(shape, dtype='float32', name=None)
```
生成一個全1張量，示例
```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1., 1., 1., 1.],
   [ 1., 1., 1., 1.],
   [ 1., 1., 1., 1.]], dtype=float32)
```

### eye
```python
eye(size, dtype='float32', name=None)
```
生成一個單位矩陣，示例：
```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1., 0., 0.],
   [ 0., 1., 0.],
   [ 0., 0., 1.]], dtype=float32)
```


### zeros_like
```python
zeros_like(x, name=None)
```
生成與另一個張量x的shape相同的全0張量，示例：
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0., 0., 0.],
   [ 0., 0., 0.]], dtype=float32)
```

### ones_like
```python
ones_like(x, name=None)
```
生成與另一個張量shape相同的全1張量，示例：
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1., 1., 1.],
   [ 1., 1., 1.]], dtype=float32)
```

### random_uniform_variable
```python
random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```
初始化一個Keras變量，其數值為從一個均勻分佈中採樣的樣本，返回之。

參數：

- shape：張量shape
- low：浮點數，均勻分佈之下界
- high：浮點數，均勻分佈之上界
- dtype：數據類型
- name：張量名
- seed：隨機數種子

示例：
```python
>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075, 0.10047495, 0.476143 ],
   [ 0.66137183, 0.00869417, 0.89220798]], dtype=float32)
```

### count_params
```python
count_params(x)
```
返回張量中標量的個數，示例：
```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0., 0., 0.],
   [ 0., 0., 0.]], dtype=float32)
```


###cast
```python
cast(x, dtype)
```
改變張量的數據類型，dtype只能是`float16`, `float32`或`float64`之一，示例：
```python
>>> from keras import backend as K
>>> input = K.placeholder((2, 3), dtype='float32')
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
__It doesn't work in-place as below.__

>>> K.cast(input, dtype='float16')
<tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
__you need to assign it.__

>>> input = K.cast(input, dtype='float16')
>>> input
<tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>```
```

### update
```python
update(x, new_x)
```
用new_x更新x

### update_add
```python
update_add(x, increment)
```
通過將x增加increment更新x

### update_sub
```python
update_sub(x, decrement)
```
通過將x減少decrement更新x


### moving_average_update
```python
moving_average_update(x, value, momentum)
```
含義暫不明確

### dot
```python
dot(x, y)
```
求兩個張量的乘積。當試圖計算兩個N階張量的乘積時，與Theano行為相同，如```(2, 3).(4, 3, 5) = (2, 4, 5))```，示例：
```python
>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

Theano-like的行為示例：
```python
>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```
### batch_dot
```python
batch_dot(x, y, axes=None)
```
按批進行張量乘法，該函數用於計算x和y的點積，其中x和y都是成batch出現的數據。即它的數據shape形如`(batch_size,:)`。 batch_dot將產生比輸入張量維度低的張量，如果張量的維度被減至1，則通過```expand_dims```保證其維度至少為2
例如，假設```x = [[1, 2],[3,4]]``` ， ```y = [[5, 6],[7, 8]]```，則`` ` batch_dot(x, y, axes=1) = [[17, 53]] ```，即```x.dot(yT)```的主對角元素，此過程中我們沒有計算過反對角元素的值

參數：

* x,y：階數大於等於2的張量，在tensorflow下，只支持大於等於3階的張量
* axes：目標結果的維度，為整數或整數列表，`axes[0]`和`axes[1]`應相同

示例：
假設`x=[[1,2],[3,4]]`，`y=[[5,6],[7,8]]`，則`batch_dot(x, y, axes=1) `為`[[17, 53]]`，恰好為`x.dot(yT)`的主對角元，整個過程沒有計算反對角元的元素。

我們做一下shape的推導，假設x是一個shape為(100,20)的tensor，y是一個shape為(100,30,20)的tensor，假設`axes=(1,2)`，則輸出tensor的shape通過循環x.shape和y.shape確定：

- `x.shape[0]`：值為100，加入到輸入shape裡
- `x.shape[1]`：20，不加入輸出shape裡，因為該維度的值會被求和(dot_axes[0]=1)
- `y.shape[0]`：值為100，不加入到輸出shape裡，y的第一維總是被忽略
- `y.shape[1]`：30，加入到輸出shape裡
- `y.shape[2]`：20，不加到output shape裡，y的第二個維度會被求和(dot_axes[1]=2)

- 結果為(100, 30)

```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```
### transpose
```python
transpose(x)
```
張量轉置，返迴轉置後的tensor，示例：
```python
>>> var = K.variable([[1, 2, 3], [4, 5, 6]])
>>> K.eval(var)
array([[ 1., 2., 3.],
   [ 4., 5., 6.]], dtype=float32)
>>> var_transposed = K.transpose(var)
>>> K.eval(var_transposed)
array([[ 1., 4.],
   [ 2., 5.],
   [ 3., 6.]], dtype=float32)

>>> input = K.placeholder((2, 3))
>>> input
<tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
>>> input_transposed = K.transpose(input)
>>> input_transposed
<tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

```

### gather
```python
gather(reference, indices)
```
在給定的張量中檢索給定下標的向量

參數：

* reference：張量
* indices：整數張量，其元素為要查詢的下標

返回值：一個與```reference```數據類型相同的張量

### max
```python
max(x, axis=None, keepdims=False)
```
求張量中的最大值

###min
```python
min(x, axis=None, keepdims=False)
```
求張量中的最小值

###sum
```python
sum(x, axis=None, keepdims=False)
```
在給定軸上計算張量中元素之和

### prod
```python
prod(x, axis=None, keepdims=False)
```
在給定軸上計算張量中元素之積

### cumsum
```python
cumsum(x, axis=0)
```
在給定軸上求張量的累積和

### cumprod
```python
cumprod(x, axis=0)
```
在給定軸上求張量的累積積
### var
```python
var(x, axis=None, keepdims=False)
```
在給定軸上計算張量方差

### std
```python
std(x, axis=None, keepdims=False)
```
在給定軸上求張量元素之標準差

### mean
```python
mean(x, axis=None, keepdims=False)
```
在給定軸上求張量元素之均值

### any
```python
any(x, axis=None, keepdims=False)
```
按位或，返回數據類型為uint8的張量（元素為0或1）

### all
```python
any(x, axis=None, keepdims=False)
```
按位與，返回類型為uint8de tensor

### argmax
```python
argmax(x, axis=-1)
```
在給定軸上求張量之最大元素下標

### argmin
```python
argmin(x, axis=-1)
```
在給定軸上求張量之最小元素下標

###square
```python
square(x)
```
逐元素平方

### abs
```python
abs(x)
```
逐元素絕對值

###sqrt
```python
sqrt(x)
```
逐元素開方

###exp
```python
exp(x)
```
逐元素求自然指數

###log
```python
log(x)
```
逐元素求自然對數

###logsumexp
```python
logsumexp(x, axis=None, keepdims=False)
```
在給定軸上計算log(sum(exp()))，該函數在數值穩定性上超過直接計算log(sum(exp()))，可以避免由exp和log導致的上溢和下溢

###round
```python
round(x)
```
逐元素四捨五入

###sign
```python
sign(x)
```
逐元素求元素的符號（+1或-1）


###pow
```python
pow(x, a)
```
逐元素求x的a次方

###clip
```python
clip(x, min_value, max_value)
```
逐元素clip（將超出指定範圍的數強制變為邊界值）

###equal
```python
equal(x, y)
```
逐元素判相等關係，返回布爾張量

###not_equal
```python
not_equal(x, y)
```
逐元素判不等關係，返回布爾張量

### greater
```python
greater(x,y)
```
逐元素判斷x>y關係，返回布爾張量

### greater_equal
```python
greater_equal(x,y)
```
逐元素判斷x>=y關係，返回布爾張量

### lesser
```python
lesser(x,y)
```
逐元素判斷x<y關係，返回布爾張量

### lesser_equal
```python
lesser_equal(x,y)
```
逐元素判斷x<=y關係，返回布爾張量

###maximum
```python
maximum(x, y)
```
逐元素取兩個張量的最大值

###minimum
```python
minimum(x, y)
```
逐元素取兩個張量的最小值

###sin
```python
sin(x)
```
逐元素求正弦值

###cos
```python
cos(x)
```
逐元素求餘弦值

### normalize_batch_in_training
```python
normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.0001)
```
對一個batch數據先計算其均值和方差，然後再進行batch_normalization

### batch_normalization
```python
batch_normalization(x, mean, var, beta, gamma, epsilon=0.0001)
```
對一個batch的數據進行batch_normalization，計算公式為：
output = (x-mean)/(sqrt(var)+epsilon)*gamma+beta

###concatenate
```python
concatenate(tensors, axis=-1)
```
在給定軸上將一個列表中的張量串聯爲一個張量 specified axis

###reshape
```python
reshape(x, shape)
```
將張量的shape變換為指定shape

###permute_dimensions
```python
permute_dimensions(x, pattern)
```
按照給定的模式重排一個張量的軸

參數：

* pattern：代表維度下標的tuple如```(0, 2, 1)```

###resize_images
```python
resize_images(X, height_factor, width_factor, dim_ordering)
```
依據給定的縮放因子，改變一個batch圖片的shape，參數中的兩個因子都為正整數，圖片的排列順序與維度的模式相關，如‘th’和‘tf’

###resize_volumes
```python
resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering)
```
依據給定的縮放因子，改變一個5D張量數據的shape，參數中的兩個因子都為正整數，圖片的排列順序與維度的模式相關，如‘th’和‘tf’。 5D數據的形式是[batch, channels, depth, height, width](th)或[batch, depth, height, width, channels](tf)

###repeat_elements
```python
repeat_elements(x, rep, axis)
```
在給定軸上重複張量元素```rep```次，與```np.repeat```類似。例如，若xshape```(s1, s2, s3) ```並且給定軸為```axis=1`，輸出張量的shape為`(s1, s2 * rep, s3)```

###repeat
```python
repeat(x, n)
```
重複2D張量，例如若xshape是```(samples, dim)```且n為2，則輸出張量的shape是```(samples, 2, dim)```

###arange
```python
arange(start, stop=None, step=1, dtype='int32')
```
生成1D的整數序列張量，該函數的參數與Theano的arange函數含義相同，如果只有一個參數被提供了，那麼它實際上就是`stop`參數的值

為了與tensorflow的默認保持匹配，函數返回張量的默認數據類型是`int32`

### tile
```python
tile(x, n)
```
將x在各個維度上重複n次，x為張量，n為與x維度數目相同的列表

### batch_flatten
```python
batch_flatten(x)
```
將一個n階張量轉變為2階張量，其第一維度保留不變

### expand_dims
```python
expand_dims(x, dim=-1)
```
在下標為```dim```的軸上增加一維

### squeeze
```python
squeeze(x, axis)
```
將下標為```axis```的一維從張量中移除

###temporal_padding
```python
temporal_padding(x, padding=1)
```
向3D張量中間的那個維度的左右兩端填充```padding```個0值
###asymmetric_temporal_padding
```python
asymmetric_temporal_padding(x, left_pad=1, right_pad=1)
```
向3D張量中間的那個維度的一端填充```padding```個0值

###spatial_2d_padding
```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```
向4D張量第二和第三維度的左右兩端填充```padding[0]```和```padding[1]```個0值


###spatial_3d_padding
```python
spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='th')
```
向5D張量深度、高度和寬度三個維度上填充```padding[0]```，```padding[1]```和```padding[2]```個0值

### stack
```python
stack(x, axis=0)
```
將一個列表中維度數目為R的張量堆積起來形成維度為R+1的新張量

### one-hot
```python
one_hot(indices, nb_classes)
```
輸入為n維的整數張量，形如(batch_size, dim1, dim2, ... dim(n-1))，輸出為(n+1)維的one-hot編碼，形如(batch_size, dim1, dim2, ... dim(n-1), nb_classes)

### reverse
```python
reverse(x, axes)
```
將一個張量在給定軸上反轉

###get_value
```python
get_value(x)
```
以Numpy array的形式返回張量的值

###batch_get_value
```python
batch_get_value(x)
```
以Numpy array list的形式返回多個張量的值

###set_value
```python
set_value(x, value)
```
從numpy array將值載入張量中

###batch_set_value
```python
batch_set_value(tuples)
```
將多個值載入多個張量變量中

參數：

* tuples: 列表，其中的元素形如```(tensor, value)```。 ```value```是要載入的Numpy array數據

### print_tensor
```
print_tensor(x, message='')
```
在求值時打印張量的信息，並返回原張量

###function
```python
function(inputs, outputs, updates=[])
```
實例化一個Keras函數

參數：

* inputs:：列表，其元素為佔位符或張量變量
* outputs：輸出張量的列表
* updates：列表，其元素是形如```(old_tensor, new_tensor)```的tuple.

###gradients
```python
gradients(loss, variables)
```
返回loss函數關於variables的梯度，variables為張量變量的列表

### stop_gradient
```python
stop_gradient(variables)
```
Returns `variables` but with zero gradient with respect to every other variables.

###rnn
```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```
在張量的時間維上迭代

參數：

* inputs： 形如```(samples, time, ...) ```的時域信號的張量，階數至少為3
* step_function：每個時間步要執行的函數
  其參數：
  * input：形如```(samples, ...)```的張量，不含時間維，代表某個時間步時一個batch的樣本
  * states：張量列表
    其返回值：
  * output：形如```(samples, ...)```的張量
    * new_states：張量列表，與‘states’的長度相同
* initial_states：形如```(samples, ...)```的張量，包含了```step_function```狀態的初始值。
* go_backwards：布爾值，若設為True，則逆向迭代序列
* mask：形如```(samples, time, 1) ```的二值張量，需要屏蔽的數據元素上值為1
* constants：按時間步傳遞給函數的常數列表
* unroll：當使用TensorFlow時，RNN總是展開的。當使用Theano時，設置該值為```True```將展開遞歸網絡
* input_length：使用TensorFlow時不需要此值，在使用Theano時，如果要展開遞歸網絡，必須指定輸入序列

返回值：形如```(last_output, outputs, new_states)```的tuple

* last_output：rnn最後的輸出，形如```(samples, ...)```
* outputs：形如```(samples, time, ...) ```的張量，每個在\[s,t\]點的輸出對應於樣本s在t時間的輸出
* new_states: 列表，其元素為形如```(samples, ...)```的張量，代表每個樣本的最後一個狀態

###switch
```python
switch(condition, then_expression, else_expression)
```
依據給定的條件‘condition’（整數或布爾值）在兩個表達式之間切換，注意兩個表達式都應該是具有同樣shape的符號化張量表達式

參數：

* condition：標量張量
* then_expression：TensorFlow表達式
* else_expression: TensorFlow表達式

###in_train_phase
```python
in_train_phase(x, alt)
```
如果處於訓練模式，則選擇x，否則選擇alt，注意alt應該與x的shape相同

###in_test_phase
```python
in_test_phase(x, alt)
```
如果處於測試模式，則選擇x，否則選擇alt，注意alt應該與x的shape相同

###relu
```python
relu(x, alpha=0.0, max_value=None)
```
修正線性單元

參數：

* alpha：負半區斜率
* max_value: 飽和門限

###elu
```python
elu(x, alpha=1.0)
```
指數線性單元

參數：

* x：輸入張量
* alpha: 標量

### softmax
```python
softmax(x)
```
返回張量的softmax值

###softplus
```python
softplus(x)
```
返回張量的softplus值

###softsign
```python
softsign(x)
```
返回張量的softsign值

###categorical_crossentropy
```python
categorical_crossentropy(output, target, from_logits=False)
```
計算輸出張量和目標張量的Categorical crossentropy（類別交叉熵），目標張量與輸出張量必須shape相同

###sparse_categorical_crossentropy
```python
sparse_categorical_crossentropy(output, target, from_logits=False)
```
計算輸出張量和目標張量的Categorical crossentropy（類別交叉熵），目標張量必須是整型張量

###binary_crossentropy
```python
binary_crossentropy(output, target, from_logits=False)
```
計算輸出張量和目標張量的交叉熵

###sigmoid
```python
sigmoid(x)
```
逐元素計算sigmoid值

###hard_sigmoid
```python
hard_sigmoid(x)
```
該函數是分段線性近似的sigmoid，計算速度更快

###tanh
```python
tanh(x)
```
逐元素計算sigmoid值

###dropout
```python
dropout(x, level, seed=None)
```
隨機將x中一定比例的值設置為0，並放縮整個tensor

參數：

* x：張量
* level：x中設置成0的元素比例
* seed：隨機數種子

###l2_normalize
```python
l2_normalize(x, axis)
```
在給定軸上對張量進行L2範數規範化

###in_top_k
```python
in_top_k(predictions, targets, k)
```
判斷目標是否在predictions的前k大值位置

參數：

* predictions：預測值張量, shape為(batch_size, classes), 數據類型float32
* targets：真值張量, shape為(batch_size,),數據類型為int32或int64
* k：整數

###conv1d
```python
conv1d(x, kernel, strides=1, border_mode='valid', image_shape=None, filter_shape=None)
```
1D卷積

參數：

* kernel：卷積核張量
* strides：步長，整型
* border_mode：“same”，“valid”之一的字符串

###conv2d
```python
conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```
2D卷積
參數：

* kernel：卷積核張量
* strides：步長，長為2的tuple
* border_mode：“same”，“valid”之一的字符串
* dim_ordering：“tf”和“th”之一，維度排列順序

### deconv2d
```python
deconv2d(x, kernel, output_shape, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```
2D反捲積（轉置卷積）

參數：

* x：輸入張量
* kernel：卷積核張量
* output_shape: 輸出shape的1D的整數張量
* strides：步長，tuple類型
* border_mode：“same”或“valid”
* dim_ordering：“tf”或“th”

### conv3d
```python
conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', volume_shape=None, filter_shape=None)
```
3D卷積

參數：

* x：輸入張量
* kernel：卷積核張量
* strides：步長，tuple類型
* border_mode：“same”或“valid”
* dim_ordering：“tf”或“th”

### pool2d
```python
pool2d(x, pool_size, strides=(1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```
2D池化

參數：

* pool_size：含有兩個整數的tuple，池的大小
* strides：含有兩個整數的tuple，步長
* border_mode：“same”，“valid”之一的字符串
* dim_ordering：“tf”和“th”之一，維度排列順序
* pool_mode: “max”，“avg”之一，池化方式

### pool3d
```python
pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```
3D池化

參數：

* pool_size：含有3個整數的tuple，池的大小
* strides：含有3個整數的tuple，步長
* border_mode：“same”，“valid”之一的字符串
* dim_ordering：“tf”和“th”之一，維度排列順序
* pool_mode: “max”，“avg”之一，池化方式

### bias_add
```python
bias_add(x, bias, data_format=None)
```
為張量增加一個偏置項

### random_normal
```python
random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```
返回具有正態分佈值的張量，mean和stddev為均值和標準差


### random_uniform
```python
random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```
返回具有均勻分佈值的張量，minval和maxval是均勻分佈的下上界

### random_binomial
```python
random_binomial(shape, p=0.0, dtype=None, seed=None)
```
返回具有二項分佈值的張量，p是二項分佈參數

### truncated_normall
```python
truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```
返回具有截尾正態分佈值的張量，在距離均值兩個標準差之外的數據將會被截斷並重新生成


### ctc_label_dense_to_sparse
```python
ctc_label_dense_to_sparse(labels, label_lengths)
```
將ctc標籤從稠密形式轉換為稀疏形式

### ctc_batch_cost
```python
ctc_batch_cost(y_true, y_pred, input_length, label_length)
```
在batch上運行CTC損失算法

參數：

* y_true：形如(samples，max_tring_length)的張量，包含標籤的真值
* y_pred：形如(samples，time_steps，num_categories)的張量，包含預測值或輸出的softmax值
* input_length：形如(samples，1)的張量，包含y_pred中每個batch的序列長
* label_length：形如(samples，1)的張量，包含y_true中每個batch的序列長

返回值：形如(samoles，1)的tensor，包含了每個元素的CTC損失

### ctc_decode
```python
ctc_decode(y_pred, input_length, greedy=True, beam_width=None, dict_seq_lens=None, dict_values=None)
```
使用貪婪算法或帶約束的字典搜索算法解碼softmax的輸出

參數：

* y_pred：形如(samples，time_steps，num_categories)的張量，包含預測值或輸出的softmax值
* input_length：形如(samples，1)的張量，包含y_pred中每個batch的序列長
* greedy：設置為True使用貪婪算法，速度快
* dict_seq_lens：dic_values列表中各元素的長度
* dict_values：列表的列表，代表字典

返回值：形如(samples，time_steps，num_catgories)的張量，包含了路徑可能性（以softmax概率的形式）。注意仍然需要一個用來取出argmax和處理空白標籤的函數

### map_fn
```python
map_fn(fn, elems, name=None)
```
元素elems在函數fn上的映射，並返回結果

參數：

* fn：函數
* elems：張量
* name：節點的名字

返回值：返回一個張量，該張量的第一維度等於elems，第二維度取決於fn

### foldl
```python
foldl(fn, elems, initializer=None, name=None)
```
減少elems，用fn從左到右連接它們

參數：

* fn：函數，例如：lambda acc, x: acc + x
* elems：張量
* initializer：初始化的值(elems[0])
* name：節點名

返回值：與initializer的類型和形狀一致

### foldr
```python
foldr(fn, elems, initializer=None, name=None)
```
減少elems，用fn從右到左連接它們

參數：

* fn：函數，例如：lambda acc, x: acc + x
  * elems：張量
* initializer：初始化的值（elems[-1]）
* name：節點名

返回值：與initializer的類型和形狀一致
