# Application應用

Kera的應用模塊Application提供了帶有預訓練權重的Keras模型，這些模型可以用來進行預測、特徵提取和finetune

模型的預訓練權重將下載到```~/.keras/models/```並在載入模型時自動載入

## 可用的模型

應用於圖像分類的模型,權重訓練自ImageNet：
* [Xception](#xception)
* [VGG16](#vgg16)
* [VGG19](#vgg19)
* [ResNet50](#resnet50)
* [InceptionV3](#inceptionv3)
* [MobileNet](#mobilenet)

所有的這些模型(除了Xception和MobileNet)都兼容Theano和Tensorflow，並會自動基於```~/.keras/keras.json```的Keras的圖像維度進行自動設置。例如，如果你設置```data_format="channel_last"```，則加載的模型將按照TensorFlow的維度順序來構造，即“Width-Height-Depth”的順序

Xception模型僅在TensorFlow下可用，因為它依賴的SeparableConvolution層僅在TensorFlow可用。 MobileNet僅在TensorFlow下可用，因為它依賴的DepethwiseConvolution層僅在TF下可用。

以上模型（暫時除了MobileNet）的預訓練權重可以在我的[百度網盤](http://pan.baidu.com/s/1geHmOpH)下載，如果有更新的話會在這里報告

***

## 圖片分類模型的示例

### 利用ResNet50網絡進行ImageNet分類
```python
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

```

### 利用VGG16提取特徵
```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### 從VGG19的任意中間層中抽取特徵
```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

###在新類別上fine-tune inceptionV3
```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will tr​​ain
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well tr​​ained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
```

### 在定制的輸入tensor上構建InceptionV3
```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3)) # this assumes K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

***
## 模型文檔
* [Xception](#xception)
* [VGG16](#vgg16)
* [VGG19](#vgg19)
* [ResNet50](#resnet50)
* [InceptionV3](#inceptionv3)
* [MobileNet](#mobilenet)


***

<a name='xception'>
<font color='#404040'>
## Xception模型
</font>
</a>
```python
keras.applications.xception.Xception(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None, classes=1000)
```

Xception V1 模型, 權重由ImageNet訓練而言

在ImageNet上,該模型取得了驗證集top1 0.790和top5 0.945的正確率

注意,該模型目前僅能以TensorFlow為後端使用,由於它依賴於"SeparableConvolution"層,目前該模型只支持channels_last的維度順序(width, height, channels)

預設輸入圖片大小為299x299

### 參數
* include_top：是否保留頂層的3個全連接網絡
* weights：None代表隨機初始化，即不加載預訓練權重。 'imagenet'代表加載預訓練權重
* input_tensor：可填入Keras tensor作為模型的圖像輸出tensor
* input_shape：可選，僅當`include_top=False`有效，應為長為3的tuple，指明輸入圖片的shape，圖片的寬高必須大於71，如(150,150,3)
* pooling：當include_top=False時，該參數指定了池化方式。 None代表不池化，最後一個卷積層的輸出為4D張量。 ‘avg’代表全局平均池化，‘max’代表全局最大值池化。

* classes：可選，圖片分類的類別數，僅當`include_top=True`並且不加載預訓練權重時可用。


### 返回值

Keras 模型對象

### 參考文獻

* [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### License
預訓練權重由我們自己訓練而來，基於MIT license發布

***

<a name='vgg16'>
<font color='#404040'>
## VGG16模型
</font>
</a>
```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000)
```

VGG16模型,權重由ImageNet訓練而來

該模型再Theano和TensorFlow後端均可使用,並接受channels_first和channels_last兩種輸入維度順序

模型的預設輸入尺寸時224x224

### 參數
* include_top：是否保留頂層的3個全連接網絡
* weights：None代表隨機初始化，即不加載預訓練權重。 'imagenet'代表加載預訓練權重
* input_tensor：可填入Keras tensor作為模型的圖像輸出tensor
* input_shape：可選，僅當`include_top=False`有效，應為長為3的tuple，指明輸入圖片的shape，圖片的寬高必須大於48，如(200,200,3)
### 返回值
* pooling：當include_top=False時，該參數指定了池化方式。 None代表不池化，最後一個卷積層的輸出為4D張量。 ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
* classes：可選，圖片分類的類別數，僅當`include_top=True`並且不加載預訓練權重時可用。

Keras 模型對象

### 參考文獻

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)：如果在研究中使用了VGG，請引用該文

### License
預訓練權重由[牛津VGG組](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)發布的預訓練權重移植而來，基於[Creative Commons Attribution License](https ://creativecommons.org/licenses/by/4.0/)

***

<a name='vgg19'>
<font color='#404040'>
## VGG19模型
</font>
</a>
```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000)
```
VGG19模型,權重由ImageNet訓練而來

該模型在Theano和TensorFlow後端均可使用,並接受channels_first和channels_last兩種輸入維度順序

模型的預設輸入尺寸時224x224
### 參數
* include_top：是否保留頂層的3個全連接網絡
* weights：None代表隨機初始化，即不加載預訓練權重。 'imagenet'代表加載預訓練權重
* input_tensor：可填入Keras tensor作為模型的圖像輸出tensor
* input_shape：可選，僅當`include_top=False`有效，應為長為3的tuple，指明輸入圖片的shape，圖片的寬高必須大於48，如(200,200,3)
* pooling：當include_top=False時，該參數指定了池化方式。 None代表不池化，最後一個卷積層的輸出為4D張量。 ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
* classes：可選，圖片分類的類別數，僅當`include_top=True`並且不加載預訓練權重時可用。
### 返回值
### 返回值

Keras 模型對象

### 參考文獻

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)：如果在研究中使用了VGG，請引用該文

### License
預訓練權重由[牛津VGG組](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)發布的預訓練權重移植而來，基於[Creative Commons Attribution License](https ://creativecommons.org/licenses/by/4.0/)

***

<a name='resnet50'>
<font color='#404040'>
## ResNet50模型
</font>
</a>
```python
keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000)
```

50層殘差網絡模型,權重訓練自ImageNet

該模型在Theano和TensorFlow後端均可使用,並接受channels_first和channels_last兩種輸入維度順序

模型的預設輸入尺寸時224x224

### 參數
* include_top：是否保留頂層的全連接網絡
* weights：None代表隨機初始化，即不加載預訓練權重。 'imagenet'代表加載預訓練權重
* input_tensor：可填入Keras tensor作為模型的圖像輸出tensor
* input_shape：可選，僅當`include_top=False`有效，應為長為3的tuple，指明輸入圖片的shape，圖片的寬高必須大於197，如(200,200,3)
* pooling：當include_top=False時，該參數指定了池化方式。 None代表不池化，最後一個卷積層的輸出為4D張量。 ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
* classes：可選，圖片分類的類別數，僅當`include_top=True`並且不加載預訓練權重時可用。
### 返回值

Keras 模型對象

### 參考文獻

* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)：如果在研究中使用了ResNet50，請引用該文

### License
預訓練權重由[Kaiming He](https://github.com/KaimingHe/deep-residual-networks)發布的預訓練權重移植而來，基於[MIT License](https://github.com/KaimingHe/ deep-residual-networks/blob/master/LICENSE)

***

<a name='inceptionv3'>
<font color='#404040'>
## InceptionV3模型
</font>
</a>
```python
keras.applications.inception_v3.InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000)
```
InceptionV3網絡,權重訓練自ImageNet

該模型在Theano和TensorFlow後端均可使用,並接受channels_first和channels_last兩種輸入維度順序

模型的預設輸入尺寸時299x299
### 參數
* include_top：是否保留頂層的全連接網絡
* weights：None代表隨機初始化，即不加載預訓練權重。 'imagenet'代表加載預訓練權重
* input_tensor：可填入Keras tensor作為模型的圖像輸出tensor
* input_shape：可選，僅當`include_top=False`有效，應為長為3的tuple，指明輸入圖片的shape，圖片的寬高必須大於197，如(200,200,3)
* pooling：當include_top=False時，該參數指定了池化方式。 None代表不池化，最後一個卷積層的輸出為4D張量。 ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
* classes：可選，圖片分類的類別數，僅當`include_top=True`並且不加載預訓練權重時可用。
### 返回值

Keras 模型對象

### 參考文獻

* [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)：如果在研究中使用了InceptionV3，請引用該文

### License
預訓練權重由我們自己訓練而來，基於[MIT License](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)

***


<a name='mobilenet'>
<font color='#404040'>
## MobileNet模型
</font>
</a>
```python
keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```
MobileNet網絡,權重訓練自ImageNet

該模型僅在TensorFlow後端均可使用,因此僅channels_last維度順序可用。當需要以`load_model()`加載MobileNet時，需要在`custom_object`中傳入`relu6`和`DepthwiseConv2D`，即：

```python
model = load_model('mobilenet.h5', custom_objects={
                   'relu6': mobilenet.relu6,
                   'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
```

模型的預設輸入尺寸時224x224
### 參數
* include_top：是否保留頂層的全連接網絡
* weights：None代表隨機初始化，即不加載預訓練權重。 'imagenet'代表加載預訓練權重
* input_tensor：可填入Keras tensor作為模型的圖像輸出tensor
* input_shape：可選，僅當`include_top=False`有效，應為長為3的tuple，指明輸入圖片的shape，圖片的寬高必須大於197，如(200,200,3)
* pooling：當include_top=False時，該參數指定了池化方式。 None代表不池化，最後一個卷積層的輸出為4D張量。 ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
* classes：可選，圖片分類的類別數，僅當`include_top=True`並且不加載預訓練權重時可用。
* alpha: 控製網絡的寬度：
  * 如果alpha<1，則同比例的減少每層的濾波器個數
  * 如果alpha>1，則同比例增加每層的濾波器個數
  * 如果alpha=1，使用預設的濾波器個數
* depth_multiplier：depthwise卷積的深度乘子，也稱為（分辨率乘子）
* dropout：dropout比例
### 返回值

Keras 模型對象

### 參考文獻

* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)：如果在研究中使用了MobileNet，請引用該文

### License
預訓練基於[Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)發布