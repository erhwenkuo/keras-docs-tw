# utils 工具

本模塊提供了一系列有用工具


##CustomObjectScope
```python
keras.utils.generic_utils.CustomObjectScope()
```
提供定制類的作用域，在該作用域內全局定制類能夠被更改，但在作用域結束後將回到初始狀態。
以```with```聲明開頭的代碼將能夠通過名字訪問定制類的實例，在with的作用範圍，這些定制類的變動將一直持續，在with作用域結束後，全局定制類的實例將回歸其在with作用域前的狀態。

```python
with CustomObjectScope({"MyObject":MyObject}):
    layer = Dense(..., W_regularizer="MyObject")
    # save, load, etc. will recognize custom object by name
```
***

## HDF5Matrix

```python
keras.utils.io_utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

這是一個使用HDF5數據集代替Numpy數組的方法

提供```start```和```end```參數可以進行切片，另外，還可以提供一個正規化函數或匿名函數，該函數將會在每片數據檢索時自動調用。

```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

* datapath: 字符串，HDF5文件的路徑
* dataset: 字符串，在datapath路徑下HDF5數據庫名字
* start: 整數，想要的數據切片起點
* end: 整數，想要的數據切片終點
* normalizer: 在每個切片數據檢索時自動調用的函數對象

***

## Sequence
```
keras.utils.data_utils.Sequence()
```
序列數據的基類，例如一個數據集。
每個Sequence必須實現`__getitem__`和`__len__`方法

下面是一個例子：
```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

__Here, `x_set` is list of path to the images__

# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):
def __init__(self, x_set, y_set, batch_size):
    self.X,self.y = x_set,y_set
    self.batch_size = batch_size

def __len__(self):
    return len(self.X) // self.batch_size

def __getitem__(self,idx):
    batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
    batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

    return np.array([
    resize(imread(file_name), (200,200))
       for file_name in batch_x]), np.array(batch_y)

```

***

## to_categorical
```python
to_categorical(y, num_classes=None)
```

將類別向量(從0到nb_classes的整數向量)映射為二值類別矩陣, 用於應用到以`categorical_crossentropy`為目標函數的模型中.

###參數

* y: 類別向量
* num_classes:總共類別數

***

## normalize
```python
normalize(x, axis=-1, order=2)
```

對numpy數組規範化，返回規範化後的數組

###參數
* x：待規範化的數據
* axis: 規範化的軸
* order：規範化方法，如2為L2範數

***

### custom_object_scope
```python
custom_object_scope()
```
提供定制類的作用域，在該作用域內全局定制類能夠被更改，但在作用域結束後將回到初始狀態。
以```with```聲明開頭的代碼將能夠通過名字訪問定制類的實例，在with的作用範圍，這些定制類的變動將一直持續，在with作用域結束後，全局定制類的實例將回歸其在with作用域前的狀態。

本函數返回```CustomObjectScope```對象

```python
with custom_object_scope({"MyObject":MyObject}):
layer = Dense(..., W_regularizer="MyObject")
# save, load, etc. will recognize custom object by name
```

***

### get_custom_objects
```python
get_custom_objects()
```

檢索全局定制類，推薦利用custom_object_scope更新和清理定制對象，但```get_custom_objects```可被直接用於訪問```_GLOBAL_CUSTOM_OBJECTS```。本函數返回從名稱到類別映射的全局字典。

```python
get_custom_objects().clear()
get_custom_objects()["MyObject"] = MyObject
```
***

## convert_all_kernels_in_model
```python
convert_all_kernels_in_model(model)
```

將模型中全部卷積核在Theano和TensorFlow模式中切換

***

### plot_model
```python
plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)
```
繪製模型的結構圖

***

### serialize_keras_object
```python
serialize_keras_object(instance)
```
將keras對象序列化

***

### deserialize_keras_object
```python
eserialize_keras_object(identifier, module_objects=None, custom_objects=None, printable_module_name='object')
```
從序列中恢復keras對象

***

### get_file

```python
get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```

從給定的URL中下載文件, 可以傳遞MD5值用於數據校驗(下載後或已經緩存的數據均可)

默認情況下文件會被下載到`~/.keras`中的`cache_subdir`文件夾，並將其文件名設為`fname`，因此例如一個文件`example.txt`最終將會被存放在`~ /.keras/datasets/example.txt~

tar,tar.gz.tar.bz和zip格式的文件可以被提取，提供哈希碼可以在下載後校驗文件。命令喊程序`shasum`和`sha256sum`可以計算哈希值。


### 參數

* fname: 文件名，如果指定了絕對路徑`/path/to/file.txt`,則文件將會保存到該位置。

* origin: 文件的URL地址

* untar: 布爾值,是否要進行解壓

* md5_hash: MD5哈希值,用於數據校驗，支持`sha256`和`md5`哈希

* cache_subdir: 用於緩存數據的文件夾，若指定絕對路徑`/path/to/folder`則將存放在該路徑下。

* hash_algorithm: 選擇文件校驗的哈希算法，可選項有'md5', 'sha256', 和'auto'. 默認'auto'自動檢測使用的哈希算法
* extract: 若為True則試圖提取文件，例如tar或zip tries extracting the file as an Archive, like tar or zip.
* archive_format: 試圖提取的文件格式，可選為'auto', 'tar', 'zip', 和None. 'tar' 包括tar, tar.gz, tar.bz文件. 默認'auto'是['tar ', 'zip']. None或空列表將返回沒有匹配。
* cache_dir: 緩存文件存放地在，參考[FAQ](for_beginners/FAQ/#where_config)
### 返回值

下載後的文件地址