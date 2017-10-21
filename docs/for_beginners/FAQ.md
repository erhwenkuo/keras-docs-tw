# Keras FAQ：常見問題

* [如何引用Keras? ](#citation)
* [如何使Keras調用GPU? ](#GPU)
* ["batch", "epoch"和"sample"都是啥意思? ](#batch)
* [如何保存Keras模型? ](#save_model)
* [為什麼訓練誤差(loss)比測試誤差高很多? ](#loss)
* [如何獲取中間層的輸出? ](#intermediate_layer)
* [如何利用Keras處理超過機器內存的數據集? ](#dataset)
* [當驗證集的loss不再下降時，如何中斷訓練? ](#stop_train)
* [驗證集是如何從訓練集中分割出來的? ](#validation_spilt)
* [訓練數據在訓練時會被隨機洗亂嗎? ](#shuffle)
* [如何在每個epoch後記錄訓練/測試的loss和正確率? ](#history)
* [如何使用狀態RNN（statful RNN）? ](#statful_RNN)
* [如何“凍結”網絡的層? ](#freeze)
* [如何從Sequential模型中去除一個層? ](#pop)
* [如何在Keras中使用預訓練的模型](#pretrain)
* [如何在Keras中使用HDF5輸入? ](#hdf5)
* [Keras的配置文件存儲在哪裡? ](#where_config)
* [在使用Keras開發過程中，我如何獲得可複現的結果? ](#reproduce)
***

<a name='citation'>
<font color='#404040'>
## 如何引用Keras?
</font>
</a>

如果Keras對你的研究有幫助的話，請在你的文章中引用Keras。這裡是一個使用BibTex的例子

```python
@misc{chollet2015keras,
  author = {Chollet, François and others},
  title = {Keras},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fchollet/keras}}
}
```

***

<a name='GPU'>
<font color='#404040'>
## 如何使Keras調用GPU?
</font>
</a>

如果採用TensorFlow作為後端，當機器上有可用的GPU時，代碼會自動調用GPU進行並行計算。如果使用Theano作為後端，可以通過以下方法設置：

方法1：使用Theano標記

在執行python腳本時使用下面的命令：

```python
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

方法2：設置```.theano```文件

點擊[這裡](http://deeplearning.net/software/theano/library/config.html)查看指導教程

方法3：在代碼的開頭處手動設置```theano.config.device```和```theano.config.floatX```

```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```


***

<a name='batch'>
<font color='#404040'>
## "batch", "epoch"和"sample"都是啥意思? ?
</font>
</a>

下面是一些使用keras時常會遇到的概念，我們來簡單解釋。

- Sample：樣本，數據集中的一條數據。例如圖片數據集中的一張圖片，語音數據中的一段音頻。
- Batch：中文為批，一個batch由若干條數據構成。 batch是進行網絡優化的基本單位，網絡參數的每一輪優化需要使用一個batch。 batch中的樣本是被並行處理的。與單個樣本相比，一個batch的數據能更好的模擬數據集的分佈，batch越大則對輸入數據分佈模擬的越好，反應在網絡訓練上，則體現為能讓網絡訓練的方向“更加正確”。但另一方面，一個batch也只能讓網絡的參數更新一次，因此網絡參數的迭代會較慢。在測試網絡的時候，應該在條件的允許的範圍內盡量使用更大的batch，這樣計算效率會更高。
- Epoch，epoch可譯為“輪次”。如果說每個batch對應網絡的一次更新的話，一個epoch對應的就是網絡的一輪更新。每一輪更新中網絡更新的次數可以隨意，但通常會設置為遍歷一遍數據集。因此一個epoch的含義是模型完整的看了一遍數據集。
設置epoch的主要作用是把模型的訓練的整個訓練過程分為若干個段，這樣我們可以更好的觀察和調整模型的訓練。 Keras中，當指定了驗證集時，每個epoch執行完後都會運行一次驗證集以確定模型的性能。另外，我們可以使用回調函數在每個epoch的訓練前後執行一些操作，如調整學習率，打印目前模型的一些信息等，詳情請參考Callback一節。
***

<a name='save_model'>
<font color='#404040'>
## 如何保存Keras模型?
</font>
</a>

我們不推薦使用pickle或cPickle來保存Keras模型

你可以使用```model.save(filepath)```將Keras模型和權重保存在一個HDF5文件中，該文件將包含：

* 模型的結構，以便重構該模型
* 模型的權重
* 訓練配置（損失函數，優化器等）
* 優化器的狀態，以便於從上次訓練中斷的地方開始

使用```keras.models.load_model(filepath)```來重新實例化你的模型，如果文件中存儲了訓練配置的話，該函數還會同時完成模型的編譯

例子：
```python
from keras.models import load_model

model.save('my_model.h5') # creates a HDF5 file 'my_model.h5'
del model # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```
如果你只是希望保存模型的結構，而不包含其權重或配置信息，可以使用：
```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```
這項操作將把模型序列化為json或yaml文件，這些文件對人而言也是友好的，如果需要的話你甚至可以手動打開這些文件並進行編輯。

當然，你也可以從保存好的json文件或yaml文件中載入模型：

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

如果需要保存模型的權重，可通過下面的代碼利用HDF5進行保存。注意，在使用前需要確保你已安裝了HDF5和其Python庫h5py

```
model.save_weights('my_model_weights.h5')
```

如果你需要在代碼中初始化一個完全相同的模型，請使用：
```python
model.load_weights('my_model_weights.h5')
```
如果你需要加載權重到不同的網絡結構（有些層一樣）中，例如fine-tune或transfer-learning，你可以通過層名字來加載模型：

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

例如：
```python
"""
假如原模型為：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1")) # will be loaded
model.add(Dense(10, name="new_dense")) # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)

```
***

<a name='loss'>
<font color='#404040'>
## 為什麼訓練誤差比測試誤差高很多?
</font>
</a>

一個Keras的模型有兩個模式：訓練模式和測試模式。一些正則機制，如Dropout，L1/L2正則項在測試模式下將不被啟用。

另外，訓練誤差是訓練數據每個batch的誤差的平均。在訓練過程中，每個epoch起始時的batch的誤差要大一些，而後面的batch的誤差要小一些。另一方面，每個epoch結束時計算的測試誤差是由模型在epoch結束時的狀態決定的，這時候的網絡將產生較小的誤差。

【Tips】可以通過定義回調函數將每個epoch的訓練誤差和測試誤差並作圖，如果訓練誤差曲線和測試誤差曲線之間有很大的空隙，說明你的模型可能有過擬合的問題。當然，這個問題與Keras無關。

***

<a name='intermediate_layer'>
<font color='#404040'>
## 如何獲取中間層的輸出?
</font>
</a>

一種簡單的方法是創建一個新的`Model`，使得它的輸出是你想要的那個輸出
```python
from keras.models import Model

model = ... # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data
```

此外，我們也可以建立一個Keras的函數來達到這一目的：

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
[model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]
```
當然，我們也可以直接編寫Theano和TensorFlow的函數來完成這件事

注意，如果你的模型在訓練和測試兩種模式下不完全一致，例如你的模型中含有Dropout層，批規範化（BatchNormalization）層等組件，你需要在函數中傳遞一個learning_phase的標記，像這樣：
```
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
[model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([X, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X, 1])[0]
```

***

<a name='dataset'>
<font color='#404040'>
## 如何利用Keras處理超過機器內存的數據集?
</font>
</a>

可以使用```model.train_on_batch(X,y)```和```model.test_on_batch(X,y)```。請參考[模型](../models/sequential.md)

另外，也可以編寫一個每次產生一個batch樣本的生成器函數，並調用```model.fit_generator(data_generator, samples_per_epoch, nb_epoch)```進行訓練

這種方式在Keras代碼包的example文件夾下CIFAR10例子裡有示範，也可點擊[這裡](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)在github上瀏覽。

***

<a name='early_stopping'>
<font color='#404040'>
## 當驗證集的loss不再下降時，如何中斷訓練?
</font>
</a>

可以定義```EarlyStopping```來提前終止訓練
```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```
請參考[回調函數](../other/callbacks)


***

<a name='validation_spilt'>
<font color='#404040'>
## 驗證集是如何從訓練集中分割出來的?
</font>
</a>

如果在```model.fit```中設置```validation_spilt```的值，則可將數據分為訓練集和驗證集，例如，設置該值為0.1，則訓練集的最後10%數據將作為驗證集，設置其他數字同理。注意，原數據在進行驗證集分割前並沒有被shuffle，所以這裡的驗證集嚴格的就是你輸入數據最末的x%。


***

<a name='shuffle'>
<font color='#404040'>
## 訓練數據在訓練時會被隨機洗亂嗎?
</font>
</a>

是的，如果```model.fit```的```shuffle```參數為真，訓練的數據就會被隨機洗亂。不設置時默認為真。訓練數據會在每個epoch的訓練中都重新洗亂一次。

驗證集的數據不會被洗亂


***

<a name='history'>
<font color='#404040'>
## 如何在每個epoch後記錄訓練/測試的loss和正確率?
</font>
</a>

```model.fit```在運行結束後返回一個```History```對象，其中含有的```history```屬性包含了訓練過程中損失函數的值以及其他度量指標。
```python
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```

***

<a name='statful_RNN'>
<font color='#404040'>
## 如何使用狀態RNN（statful RNN）?
</font>
</a>

一個RNN是狀態RNN，意味著訓練時每個batch的狀態都會被重用於初始化下一個batch的初始狀態。

當使用狀態RNN時，有如下假設

* 所有的batch都具有相同數目的樣本

* 如果```X1```和```X2```是兩個相鄰的batch，那麼對於任何```i```，```X2[i]```都是`` `X1[i]```的後續序列

要使用狀態RNN，我們需要

* 顯式的指定每個batch的大小。可以通過模型的首層參數```batch_input_shape```來完成。 ```batch_input_shape```是一個整數tuple，例如\(32,10,16\)代表一個具有10個時間步，每步向量長為16，每32個樣本構成一個batch的輸入數據格式。

* 在RNN層中，設置```stateful=True```

要重置網絡的狀態，使用：

* ```model.reset_states()```來重置網絡中所有層的狀態

* ```layer.reset_states()```來重置指定層的狀態

例子：
```python
X # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(X[:, :10, :], np.reshape(X[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(X[:, 10:20, :], np.reshape(X[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```
注意，```predict```，```fit```，```train_on_batch```
，```predict_classes```等方法都會更新模型中狀態層的狀態。這使得你可以不但可以進行狀態網絡的訓練，也可以進行狀態網絡的預測。

***



<a name='freeze'>
<font color='#404040'>
## 如何“凍結”網絡的層?
</font>
</a>

“凍結”一個層指的是該層將不參加網絡訓練，即該層的權重永不會更新。在進行fine-tune時我們經常會需要這項操作。
在使用固定的embedding層處理文本輸入時，也需要這個技術。

可以通過向層的構造函數傳遞```trainable```參數來指定一個層是不是可訓練的，如：

```python
frozen_layer = Dense(32,trainable=False)
```

此外，也可以通過將層對象的```trainable```屬性設為```True```或```False```來為已經搭建好的模型設置要凍結的層。
在設置完後，需要運行```compile```來使設置生效，例如：

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels) # this does NOT update the weights of `layer`
trainable_model.fit(data, labels) # this updates the weights of `layer`
```

***

<a name='pop'>
<font color='#404040'>
## 如何從Sequential模型中去除一個層?
</font>
</a>
可以通過調用```.pop()```來去除模型的最後一個層，反複調用n次即可去除模型後面的n個層

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers)) # "2"

model.pop()
print(len(model.layers)) # "1"
```


***

<a name='pretrain'>
<font color='#404040'>
## 如何在Keras中使用預訓練的模型?
</font>
</a>

我們提供了下面這些圖像分類的模型代碼及預訓練權重：

- VGG16
- VGG19
- ResNet50
- Inception v3

可通過```keras.applications```載入這些模型：

```python
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

model = VGG16(weights='imagenet', include_top=True)
```

這些代碼的使用示例請參考```.Application```模型的[文檔](../other/application.md)


使用這些預訓練模型進行特徵抽取或fine-tune的例子可以參考[此博客](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html )

VGG模型也是很多Keras例子的基礎模型，如：

* [<font color='#FF0000'>Style-transfer</font>](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py)
* [<font color='#FF0000'>Feature visualization</font>](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py)
* [<font color='#FF0000'>Deep dream</font>](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py)

<a name='hdf5'>
<font color='#404040'>
## 如何在Keras中使用HDF5輸入?
</font>
</a>

你可以使用keras.utils中的```HDF5Matrix```類來讀取HDF5輸入，參考[這裡](../utils.md)

可以直接使用HDF5數據庫，示例
```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    X_data = f['X_data']
    model.predict(X_data)
```

***
<a name='where_config'>
<font color='#404040'>
## Keras的配置文件存儲在哪裡?
</font>
</a>

所有的Keras數據默認存儲在：
```bash
$HOME/.keras/
```

對windows用戶而言，`$HOME`應替換為`%USERPROFILE%`

當Keras無法在上面的位置創建文件夾時（例如由於權限原因），備用的地址是`/tmp/.keras/`

Keras配置文件為JSON格式的文件，保存在`$HOME/.keras/keras.json`。默認的配置文件長這樣：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
該文件包含下列字段：


- 默認的圖像數據格式`channels_last`或`channels_first`
- 用於防止除零錯誤的`epsilon`
- 默認的浮點數類型
- 默認的後端

類似的，緩存的數據集文件，即由`get_file()`下載的文件，默認保存在`$HOME/.keras/datasets/`

***
<a name='reproduce'>
<font color='#404040'>
## 在使用Keras開發過程中，我如何獲得可複現的結果?
</font>
</a>

在開發模型中，有時取得可複現的結果是很有用的。例如，這可以幫助我們定位模型性能的改變是由模型本身引起的還是由於數據上的變化引起的。下面的代碼展示瞭如何獲得可複現的結果，該代碼基於Python3的tensorflow後端

```python
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
```