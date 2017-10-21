# Sequential模型接口

如果剛開始學習Sequential模型，請首先移步[這裡](../getting_started/sequential_model.md)閱讀文檔，本節內容是Sequential的API和參數介紹。

## 常用Sequential屬性

* ```model.layers```是添加到​​模型上的層的list

***

## Sequential模型方法

### add
```python
add(self, layer)
```
向模型中添加一個層

* layer: Layer對象

***

### pop
```python
pop(self)
```
彈出模型最後的一層，無返回值


***

### compile

```python
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
```
編譯用來配置模型的學習過程，其參數有

* optimizer：字符串（預定義優化器名）或優化器對象，參考[優化器](../other/optimizers.md)

* loss：字符串（預定義損失函數名）或目標函數，參考[損失函數](../other/objectives.md)

* metrics：列表，包含評估模型在訓練和測試時的網絡性能的指標，典型用法是```metrics=['accuracy']```

* sample_weight_mode：如果你需要按時間步為樣本賦權（2D權矩陣），將該值設為“temporal”。默認為“None”，代表按樣本賦權（1D權）。在下面```fit```函數的解釋中有相關的參考內容。

* kwargs：使用TensorFlow作為後端請忽略該參數，若使用Theano作為後端，kwargs的值將會傳遞給 K.function

```python
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
```

模型在使用前必須編譯，否則在調用fit或evaluate時會拋出異常。

### fit

```python
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```
本函數將模型訓練```nb_epoch```輪，其參數有：

* x：輸入數據。如果模型只有一個輸入，那麼x的類型是numpy array，如果模型有多個輸入，那麼x的類型應當為list，list的元素是對應於各個輸入的numpy array

* y：標籤，numpy array

* batch_size：整數，指定進行梯度下降時每個batch包含的樣本數。訓練時一個batch的樣本會被計算一次梯度下降，使目標函數優化一步。

* epochs：整數，訓練的輪數，每個epoch會把訓練集輪一遍。

* verbose：日誌顯示，0為不在標準輸出流輸出日誌信息，1為輸出進度條記錄，2為每個epoch輸出一行記錄

* callbacks：list，其中的元素是```keras.callbacks.Callback```的對象。這個list中的回調函數將會在訓練過程中的適當時機被調用，參考[回調函數](../other/callbacks.md)

* validation_split：0~1之間的浮點數，用來指定訓練集的一定比例數據作為驗證集。驗證集將不參與訓練，並在每個epoch結束後測試的模型的指標，如損失函數、精確度等。注意，validation_split的劃分在shuffle之前，因此如果你的數據本身是有序的，需要先手工打亂再指定validation_split，否則可能會出現驗證集樣本不均勻。

* validation_data：形式為（X，y）的tuple，是指定的驗證集。此參數將覆蓋validation_spilt。

* shuffle：布爾值或字符串，一般為布爾值，表示是否在訓練過程中隨機打亂輸入樣本的順序。若為字符串“batch”，則是用來處理HDF5數據的特殊情況，它將在batch內部將數據打亂。

* class_weight：字典，將不同的類別映射為不同的權值，該參數用來在訓練過程中調整損失函數（只能用於訓練）

* sample_weight：權值的numpy array，用於在訓練時調整損失函數（僅用於訓練）。可以傳遞一個1D的與樣本等長的向量用於對樣本進行1對1的加權，或者在面對時序數據時，傳遞一個的形式為（samples，sequence_length）的矩陣來為每個時間步上的樣本賦不同的權。這種情況下請確定在編譯模型時添加了```sample_weight_mode='temporal'```。

* initial_epoch: 從該參數指定的epoch開始訓練，在繼續之前的訓練時有用。

```fit```函數返回一個```History```的對象，其```History.history```屬性記錄了損失函數和其他指標的數值隨epoch變化的情況，如果有驗證集的話，也包含了驗證集的這些指標變化情況

***
<a name='evaluate'>
<font color='#404040'>
### evaluate
</font>
</a>
```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```
本函數按batch計算在某些輸入數據上模型的誤差，其參數有：

* x：輸入數據，與```fit```一樣，是numpy array或numpy array的list

* y：標籤，numpy array

* batch_size：整數，含義同```fit```的同名參數

* verbose：含義同```fit```的同名參數，但只能取0或1

* sample_weight：numpy array，含義同```fit```的同名參數

本函數返回一個測試誤差的標量值（如果模型沒有其他評價指標），或一個標量的list（如果模型還有其他的評價指標）。 ```model.metrics_names```將給出list中各個值的含義。

如果沒有特殊說明，以下函數的參數均保持與```fit```的同名參數相同的含義

如果沒有特殊說明，以下函數的verbose參數（如果有）均只能取0或1


***

### predict

```python
predict(self, x, batch_size=32, verbose=0)
```
本函數按batch獲得輸入數據對應的輸出，其參數有：

函數的返回值是預測值的numpy array

***

### train_on_batch
```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```
本函數在一個batch的數據上進行一次參數更新

函數返回訓練誤差的標量值或標量值的list，與[evaluate](#evaluate)的情形相同。

***

### test_on_batch
```python
test_on_batch(self, x, y, sample_weight=None)
```
本函數在一個batch的樣本上對模型進行評估

函數的返回與[evaluate](#evaluate)的情形相同

***

### predict_on_batch
```python
predict_on_batch(self, x)
```
本函數在一個batch的樣本上對模型進行測試

函數返回模型在一個batch上的預測結果

***

### fit_generator
```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
```
利用Python的生成器，逐個生成數據的batch並進行訓練。生成器與模型將並行執行以提高效率。例如，該函數允許我們在CPU上進行實時的數據提升，同時在GPU上進行模型訓練

函數的參數是：

* generator：生成器函數，生成器的輸出應該為：
* 一個形如（inputs，targets）的tuple

* 一個形如（inputs, targets,sample_weight）的tuple。所有的返回值都應該包含相同數目的樣本。生成器將無限在數據集上循環。每個epoch以經過模型的樣本數達到```samples_per_epoch```時，記一個epoch結束

* steps_per_epoch：整數，當生成器返回```steps_per_epoch```次數據時計一個epoch結束，執行下一個epoch

* epochs：整數，數據迭代的輪數

* verbose：日誌顯示，0為不在標準輸出流輸出日誌信息，1為輸出進度條記錄，2為每個epoch輸出一行記錄

* validation_data：具有以下三種形式之一
* 生成驗證集的生成器

* 一個形如（inputs,targets）的tuple

* 一個形如（inputs,targets，sample_weights）的tuple
* validation_steps: 當validation_data為生成器時，本參數指定驗證集的生成器返回次數

* class_weight：規定類別權重的字典，將類別映射為權重，常用於處理樣本不均衡問題。

* sample_weight：權值的numpy array，用於在訓練時調整損失函數（僅用於訓練）。可以傳遞一個1D的與樣本等長的向量用於對樣本進行1對1的加權，或者在面對時序數據時，傳遞一個的形式為（samples，sequence_length）的矩陣來為每個時間步上的樣本賦不同的權。這種情況下請確定在編譯模型時添加了```sample_weight_mode='temporal'```。

* workers：最大進程數

* max_q_size：生成器隊列的最大容量

* pickle_safe: 若為真，則使用基於進程的線程。由於該實現依賴多進程，不能傳遞non picklable（無法被pickle序列化）的參數到生成器中，因為無法輕易將它們傳入子進程中。

* initial_epoch: 從該參數指定的epoch開始訓練，在繼續之前的訓練時有用。


函數返回一個```History```對象

例子：

```python
def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create Numpy arrays of input data
        # and labels, from each line in the file
        x, y = process_line(line)
        yield (x, y)
    f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, epochs=10)
```

***

### evaluate_generator
```python
evaluate_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False)
```
本函數使用一個生成器作為數據源評估模型，生成器應返回與```test_on_batch```的輸入數據相同類型的數據。該函數的參數與```fit_generator```同名參數含義相同，steps是生成器要返回數據的輪數。

***

### predcit_generator
```python
predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
```
本函數使用一個生成器作為數據源預測模型，生成器應返回與```test_on_batch```的輸入數據相同類型的數據。該函數的參數與```fit_generator```同名參數含義相同，steps是生成器要返回數據的輪數。

***