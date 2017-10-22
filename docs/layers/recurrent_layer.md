# 循環層Recurrent

## Recurrent層
```python
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```

這是循環層的抽像類，請不要在模型中直接應用該層（因為它是抽像類，無法實例化任何對象）。請使用它的子類```LSTM```，```GRU```或```SimpleRNN```。

所有的循環層（```LSTM```,```GRU```,```SimpleRNN```）都服從本層的性質，並接受本層指定的所有關鍵字參數。

### 參數

* weights：numpy array的list，用以初始化權重。該list形如```[(input_dim, output_dim),(output_dim, output_dim),(output_dim,)]```

* return_sequences：布爾值，預設```False```，控制返回類型。若為```True```則返回整個序列，否則僅返回輸出序列的最後一個輸出

* go_backwards：布爾值，預設為```False```，若為```True```，則逆向處理輸入序列並返回逆序後的序列

* stateful：布爾值，預設為```False```，若為```True```，則一個batch中下標為i的樣本的最終狀態將會用作下一個batch同樣下標的樣本的初始狀態。

* unroll：布爾值，預設為```False```，若為```True```，則循環層將被展開，否則就使用符號化的循環。當使用TensorFlow為後端時，循環網絡本來就是展開的，因此該層不做任何事情。層展開會佔用更多的記憶體，但會加速RNN的運算。層展開只適用於短序列。

* implementation：0，1或2， 若為0，則RNN將以更少但是更大的矩陣乘法實現，因此在CPU上運行更快，但消耗更多的記憶體。如果設為1，則RNN將以更多但更小的矩陣乘法實現，因此在CPU上運行更慢，在GPU上運行更快，並且消耗更少的記憶體。如果設為2（僅LSTM和GRU可以設為2），則RNN將把輸入門、遺忘門和輸出門合併為單個矩陣，以獲得更加在GPU上更加高效的實現。注意，RNN dropout必須在所有門上共享，並導致正則效果性能微弱降低。

* input_dim：輸入維度，當使用該層為模型首層時，應指定該值（或等價的指定input_shape)

* input_length：當輸入序列的長度固定時，該參數為輸入序列的長度。當需要在該層後連接```Flatten```層，然後又要連接```Dense```層時，需要指定該參數，否則全連接的輸出無法計算出來。注意，如果循環層不是網絡的第一層，你需要在網絡的第一層中指定序列的長度（通過```input_shape```指定）。

### 輸入shape

形如（samples，timesteps，input_dim）的3D張量

### 輸出shape

如果```return_sequences=True```：返回形如（samples，timesteps，output_dim）的3D張量

否則，返回形如（samples，output_dim）的2D張量

### 例子
```python
# as the first layer in a Sequential model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# now model.output_shape == (None, 32)
# note: `None` is the batch dimension.

# the following is identical:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# for subsequent layers, no need to specify the input size:
         model.add(LSTM(16))

# to stack recurrent layers, you must use return_sequences=True
# on any recurrent layer that feeds into another recurrent layer.
# note that you only need to specify the input size on the first layer.
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

### 指定RNN初始狀態的注意事項

可以通過設置`initial_state`用符號式的方式指定RNN層的初始狀態。即，`initial_stat`的值應該為一個tensor或一個tensor列表，代表RNN層的初始狀態。

也可以通過設置`reset_states`參數用數值的方法設置RNN的初始狀態，狀態的值應該為numpy數組或numpy數組的列表，代表RNN層的初始狀態。

### 屏蔽輸入數據（Masking）

循環層支持通過時間步變量對輸入數據進行Masking，如果想將輸入數據的一部分屏蔽掉，請使用[Embedding](embedding_layer)層並將參數```mask_zero```設為```True`` `。


### 使用狀態RNN的注意事項

可以將RNN設置為‘stateful’，意味著由每個batch計算出的狀態都會被重用於初始化下一個batch的初始狀態。狀態RNN假設連續的兩個batch之中，相同下標的元素有一一映射關係。

要啟用狀態RNN，請在實例化層對象時指定參數```stateful=True```，並在Sequential模型使用固定大小的batch：通過在模型的第一層傳入```batch_size=(. ..)```和```input_shape```來實現。在函數式模型中，對所有的輸入都要指定相同的```batch_size```。

如果要將循環層的狀態重置，請調用```.reset_states()```，對模型調用將重置模型中所有狀態RNN的狀態。對單個層調用則只重置該層的狀態。


***

## SimpleRNN層
```python
keras.layers.recurrent.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
全連接RNN網絡，RNN的輸出會被回饋到輸入

### 參數

* units：輸出維度

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)）

* use_bias: 布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* recurrent_initializer：循環核的初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* recurrent_regularizer：施加在循環核上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* recurrent_constraints：施加在循環核上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象

* dropout：0~1之間的浮點數，控制輸入線性變換的神經元斷開比例

* recurrent_dropout：0~1之間的浮點數，控制循環狀態的線性變換的神經元斷開比例

### 參考文獻

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

***
## GRU層
```python
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer= None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
門限循環單元（詳見參考文獻）

### 參數

* units：輸出維度

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)）

* use_bias: 布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* recurrent_initializer：循環核的初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* recurrent_regularizer：施加在循環核上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* recurrent_constraints：施加在循環核上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象

* dropout：0~1之間的浮點數，控制輸入線性變換的神經元斷開比例

* recurrent_dropout：0~1之間的浮點數，控制循環狀態的線性變換的神經元斷開比例

### 參考文獻

* [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)

* [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

***

## LSTM層
```python
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer= None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
Keras長短期記憶模型，關於此算法的詳情，請參考[本教程](http://deeplearning.net/tutorial/lstm.html)

### 參數

* units：輸出維度

* activation：激活函數，為預定義的激活函數名（參考[激活函數](../other/activations)）

* recurrent_activation: 為循環步施加的激活函數（參考[激活函數](../other/activations)）

* use_bias: 布爾值，是否使用偏置項

* kernel_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* recurrent_initializer：循環核的初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* bias_initializer：權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。參考[initializers](../other/initializations)

* kernel_regularizer：施加在權重上的正則項，為[Regularizer](../other/regularizers)對象

* bias_regularizer：施加在偏置向量上的正則項，為[Regularizer](../other/regularizers)對象

* recurrent_regularizer：施加在循環核上的正則項，為[Regularizer](../other/regularizers)對象

* activity_regularizer：施加在輸出上的正則項，為[Regularizer](../other/regularizers)對象

* kernel_constraints：施加在權重上的約束項，為[Constraints](../other/constraints)對象

* recurrent_constraints：施加在循環核上的約束項，為[Constraints](../other/constraints)對象

* bias_constraints：施加在偏置上的約束項，為[Constraints](../other/constraints)對象

* dropout：0~1之間的浮點數，控制輸入線性變換的神經元斷開比例

* recurrent_dropout：0~1之間的浮點數，控制循環狀態的線性變換的神經元斷開比例

### 參考文獻

* [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)（original 1997 paper）

* [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)

* [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)