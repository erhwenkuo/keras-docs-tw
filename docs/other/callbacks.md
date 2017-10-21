# 回調函數Callbacks

回調函數是一組在訓練的特定階段被調用的函數集，你可以使用回調函數來觀察訓練過程中網絡內部的狀態和統計信息。通過傳遞回調函數列表到模型的```.fit()```中，即可在給定的訓練階段調用該函數集中的函數。

【Tips】雖然我們稱之為回調“函數”，但事實上Keras的回調函數是一個類，回調函數只是習慣性稱呼


## Callback
```python
keras.callbacks.Callback()
```
這是回調函數的抽像類，定義新的回調函數必須繼承自該類

### 類屬性

* params：字典，訓練參數集（如信息顯示方法verbosity，batch大小，epoch數）

* model：```keras.models.Model```對象，為正在訓練的模型的引用

回調函數以字典```logs```為參數，該字典包含了一系列與當前batch或epoch相關的信息。

目前，模型的```.fit()```中有下列參數會被記錄到```logs```中：

* 在每個epoch的結尾處（on_epoch_end），```logs```將包含訓練的正確率和誤差，```acc```和```loss```，如果指定了驗證集，還會包含驗證集正確率和誤差```val_acc)```和```val_loss```，```val_acc```還額外需要在```.compile```中啟用``` metrics=['accuracy']```。

* 在每個batch的開始處（on_batch_begin）：```logs```包含```size```，即當前batch的樣本數

* 在每個batch的結尾處（on_batch_end）：```logs```包含```loss```，若啟用```accuracy```則還包含```acc```

***

## BaseLogger
```python
keras.callbacks.BaseLogger()
```
該回調函數用來對每個epoch累加```metrics```指定的監視指標的epoch平均值

該回調函數在每個Keras模型中都會被自動調用

***

## ProgbarLogger
```python
keras.callbacks.ProgbarLogger()
```
該回調函數用來將```metrics```指定的監視指標輸出到標準輸出上

***

## History
```python
keras.callbacks.History()
```
該回調函數在Keras模型上會被自動調用，```History```對象即為```fit```方法的返回值

***

## ModelCheckpoint
```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```
該回調函數將在每個epoch後保存模型到```filepath```

```filepath```可以是格式化的字符串，裡面的佔位符將會被```epoch```值和傳入```on_epoch_end```的```logs```關鍵字所填入

例如，```filepath```若為```weights.{epoch:02d-{val_loss:.2f}}.hdf5```，則會生成對應epoch和驗證集loss的多個文件。

### 參數

* filename：字符串，保存模型的路徑

* monitor：需要監視的值

* verbose：信息展示模式，0或1

* save_best_only：當設置為```True```時，將只保存在驗證集上性能最好的模型

* mode：'auto'，'min'，'max'之一，在```save_best_only=True```時決定性能最佳模型的評判準則，例如，當監測值為```val_acc```時，模式應為```max```，當檢測值為```val_loss```時，模式應為```min```。在```auto```模式下，評價準則由被監測值的名字自動推斷。

* save_weights_only：若設置為True，則只保存模型權重，否則將保存整個模型（包括模型結構，配置信息等）

* period：CheckPoint之間的間隔的epoch數

***

## EarlyStopping
```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
```
當監測值不再改善時，該回調函數將中止訓練

### 參數

* monitor：需要監視的量

* patience：當early stop被激活（如發現loss相比上一個epoch訓練沒有下降），則經過```patience```個epoch後停止訓練。

* verbose：信息展示模式

* mode：‘auto’，‘min’，‘max’之一，在```min```模式下，如果檢測值停止下降則中止訓練。在```max```模式下，當檢測值不再上升則停止訓練。

***

## RemoteMonitor
```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000')
```
該回調函數用於向服務器發送事件流，該回調函數需要```requests```庫

### 參數

* root：該參數為根url，回調函數將在每個epoch後把產生的事件流發送到該地址，事件將被發往```root + '/publish/epoch/end/'```。發送方法為HTTP POST，其```data```字段的數據是按JSON格式編碼的事件字典。

***

## LearningRateScheduler
```python
keras.callbacks.LearningRateScheduler(schedule)
```
該回調函數是學習率調度器

### 參數

* schedule：函數，該函數以epoch號為參數（從0算起的整數），返回一個新學習率（浮點數）

***

## TensorBoard
```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
```
該回調函數是一個可視化的展示器

TensorBoard是TensorFlow提供的可視化工具，該回調函數將日誌信息寫入TensorBorad，使得你可以動態的觀察訓練和測試指標的圖像以及不同層的激活值直方圖。

如果已經通過pip安裝了TensorFlow，我們可通過下面的命令啟動TensorBoard：

```python
tensorboard --logdir=/full_path_to_your_logs
```
更多的參考信息，請點擊[這裡](https://www.tensorflow.org/get_started/summaries_and_tensorboard)

### 參數

* log_dir：保存日誌文件的地址，該文件將被TensorBoard解析以用於可視化

* histogram_freq：計算各個層激活值直方圖的頻率（每多少個epoch計算一次），如果設置為0則不計算。

* write_graph: 是否在Tensorboard上可視化圖，當設為True時，log文件可能會很大
* write_images: 是否將模型權重以圖片的形式可視化
* embeddings_freq: 依據該頻率(以epoch為單位)篩選保存的embedding層
* embeddings_layer_names:要觀察的層名稱的列表，若設置為None或空列表，則所有embedding層都將被觀察。
* embeddings_metadata: 字典，將層名稱映射為包含該embedding層元數據的文件名，參考[這裡](https://keras.io/https__://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)獲得元數據文件格式的細節。如果所有的embedding層都使用相同的元數據文件，則可傳遞字符串。

***

## ReduceLROnPlateau
```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```
當評價指標不在提升時，減少學習率

當學習停滯時，減少2倍或10倍的學習率常常能獲得較好的效果。該回調函數檢測指標的情況，如果在`patience`個epoch中看不到模型性能提升，則減少學習率

### 參數

- monitor：被監測的量
- factor：每次減少學習率的因子，學習率將以`lr = lr*factor`的形式被減少
- patience：當patience個epoch過去而模型性能不提升時，學習率減少的動作會被觸發
- mode：‘auto’，‘min’，‘max’之一，在```min```模式下，如果檢測值觸發學習率減少。在```max```模式下，當檢測值不再上升則觸發學習率減少。
- epsilon：閾值，用來確定是否進入檢測值的“平原區”
- cooldown：學習率減少後，會經過cooldown個epoch才重新進行正常操作
- min_lr：學習率的下限


##CSVLogger
```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```
將epoch的訓練結果保存在csv文件中，支持所有可被轉換為string的值，包括1D的可迭代數值如np.ndarray.
###參數

- fiename：保存的csv文件名，如`run/log.csv`
- separator：字符串，csv分隔符
- append：默認為False，為True時csv文件如果存在則繼續寫入，為False時總是覆蓋csv文件


## LambdaCallback
```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```
用於創建簡單的callback的callback類

該callback的匿名函數將會在適當的時候調用，注意，該回調函數假定了一些位置參數`on_eopoch_begin`和`on_epoch_end`假定輸入的參數是`epoch, logs`. `on_batch_begin`和`on_batch_end`假定輸入的參數是`batch, logs`，`on_train_begin`和`on_train_end`假定輸入的參數是`logs`

### 參數

- on_epoch_begin: 在每個epoch開始時調用
- on_epoch_end: 在每個epoch結束時調用
- on_batch_begin: 在每個batch開始時調用
- on_batch_end: 在每個batch結束時調用
- on_train_begin: 在訓練開始時調用
- on_train_end: 在訓練結束時調用

### 示例

```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Plot the loss after every epoch.
import numpy as np
import matplotlib.pyplot as plt
plot_loss_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
                      logs['loss']))

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
    p.terminate() for p in processes if p.is_alive()])

model.fit(...,
      callbacks=[batch_print_callback,
         plot_loss_callback,
         cleanup_callback])
```

## 編寫自己的回調函數

我們可以通過繼承```keras.callbacks.Callback```編寫自己的回調函數，回調函數通過類成員```self.model```訪問訪問，該成員是模型的一個引用。

這裡是一個簡單的保存每個batch的loss的回調函數：

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

### 例子：記錄損失函數的歷史數據
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
```

### 例子：模型檢查點
```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```