# 圖片預處理

## 圖片生成器ImageDataGenerator
```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizo​​ntal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```
用以生成一個batch的圖像數據，支持實時數據提升。訓練時該函數會無限生成數據，直到達到規定的epoch次數為止。

### 參數

* featurewise_center：布爾值，使輸入數據集去中心化（均值為0）, 按feature執行

* samplewise_center：布爾值，使輸入數據的每個樣本均值為0

* featurewise_std_normalization：布爾值，將輸入除以數據集的標準差以完成標準化, 按feature執行

* samplewise_std_normalization：布爾值，將輸入的每個樣本除以其自身的標準差

* zca_whitening：布爾值，對輸入數據施加ZCA白化

* zca_epsilon: ZCA使用的eposilon，預設1e-6

* rotation_range：整數，數據提升時圖片隨機轉動的角度

* width_shift_range：浮點數，圖片寬度的某個比例，數據提升時圖片水平偏移的幅度

* height_shift_range：浮點數，圖片高度的某個比例，數據提升時圖片豎直偏移的幅度

* shear_range：浮點數，剪切強度（逆時針方向的剪切變換角度）

* zoom_range：浮點數或形如```[lower,upper]```的列表，隨機縮放的幅度，若為浮點數，則相當於```[lower,upper] = [1 - zoom_range, 1 +zoom_range]```

* channel_shift_range：浮點數，隨機通道偏移的幅度

* fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，當進行變換時超出邊界的點將根據本參數給定的方法進行處理

* cval：浮點數或整數，當```fill_mode=constant```時，指定要向超出邊界的點填充的值

* horizo​​ntal_flip：布爾值，進行隨機水平翻轉

* vertical_flip：布爾值，進行隨機豎直翻轉

* rescale: 重放縮因子,預設為None. 如果為None或0則不進行放縮,否則會將該數值乘到數據上(在應用其他變換之前)
* preprocessing_function: 將被應用於每個輸入的函數。該函數將在任何其他修改之前運行。該函數接受一個參數，為一張圖片（秩為3的numpy array），並且輸出一個具有相同shape的numpy array

* data_format：字符串，“channel_first”或“channel_last”之一，代表圖像的通道維的位置。該參數是Keras 1.x中的image_dim_ordering，“channel_last”對應原本的“tf”，“channel_first”對應原本的“th”。以128x128的RGB圖像為例，“channel_first”應將數據組織為（3,128,128），而“channel_last”應將數據組織為（128,128,3）。該參數的預設值是```~/.keras/keras.json```中設置的值，若從未設置過，則為“channel_last”

***

### 方法

* fit(x, augment=False, rounds=1)：計算依賴於數據的變換所需要的統計信息(均值方差等),只有使用```featurewise_center```，```featurewise_std_normalization```或` ``zca_whitening```時需要此函數。

    * X：numpy array，樣本數據，秩應為4.在黑白圖像的情況下channel軸的值為1，在彩色圖像情況下值為3

    * augment：布爾值，確定是否使用隨即提升過的數據

    * round：若設```augment=True```，確定要在數據上進行多少輪數據提升，預設值為1

    * seed: 整數,隨機數種子

* flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')：接收numpy數組和標籤為參數,生成經過數據提升或標準化後的batch數據,並在一個無限循環中不斷的返回batch數據

    * x：樣本數據，秩應為4.在黑白圖像的情況下channel軸的值為1，在彩色圖像情況下值為3

    * y：標籤

    * batch_size：整數，預設32

    * shuffle：布爾值，是否隨機打亂數據，預設為True

    * save_to_dir：None或字符串，該參數能讓你將提升後的圖片保存起來，用以可視化

    * save_prefix：字符串，保存提升後圖片時使用的前綴, 僅當設置了```save_to_dir```時生效

    * save_format："png"或"jpeg"之一，指定保存圖片的數據格式,預設"jpeg"

    * yields:形如(x,y)的tuple,x是代表圖像數據的numpy數組.y是代表標籤的numpy數組.該迭代器無限循環.

    * seed: 整數,隨機數種子

* flow_from_directory(directory): 以文件夾路徑為參數,生成經過數據提升/歸一化後的數據,在一個無限循環中無限產生batch數據

    * directory: 目標文件夾路徑,對於每一個類,該文件夾都要包含一個子文件夾.子文件夾中任何JPG、PNG和BNP的圖片都會被生成器使用.詳情請查看[<font color= '#FF0000'>此腳本</font>](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
    * target_size: 整數tuple,預設為(256, 256). 圖像將被resize成該尺寸
    * color_mode: 顏色模式,為"grayscale","rgb"之一,預設為"rgb".代表這些圖片是否會被轉換為單通道或三通道的圖片.
    * classes: 可選參數,為子文件夾的列表,如['dogs','cats']預設為None. 若未提供,則該類別列表將從`directory`下的子文件夾名稱/結構自動推斷。每一個子文件夾都會被認為是一個新的類。 (類別的順序將按照字母表順序映射到標籤值)。通過屬性`class_indices`可獲得文件夾名與類的序號的對應字典。
    * class_mode: "categorical", "binary", "sparse"或None之一. 預設為"categorical. 該參數決定了返回的標籤數組的形式, "categorical"會返回2D的one-hot編碼標籤,"binary "返回1D的二值標籤."sparse"返回1D的整數標籤,如果為None則不返回任何標籤, 生成器將僅僅生成batch數據, 這種情況在使用```model.predict_generator()```和```model.evaluate_generator()```等函數時會用到.
    * batch_size: batch數據的大小,預設32
    * shuffle: 是否打亂數據,預設為True
    * seed: 可選參數,打亂數據和進行變換時的隨機數種子
    * save_to_dir: None或字符串，該參數能讓你將提升後的圖片保存起來，用以可視化
    * save_prefix：字符串，保存提升後圖片時使用的前綴, 僅當設置了```save_to_dir```時生效
    * save_format："png"或"jpeg"之一，指定保存圖片的數據格式,預設"jpeg"
    * flollow_links: 是否訪問子文件夾中的軟鏈接
### 例子

使用```.flow()```的例子
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizo​​ntal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train), epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print 'Epoch', e
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        loss = model.train(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

使用```.flow_from_directory(directory)```的例子
```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizo​​ntal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

同時變換圖像和mask

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```